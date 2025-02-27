"""
main.py

Main driver code for processing bridge inspection reports using agents.
This script reads input text, segments it, performs extraction, validation, correction of bridge inspection entities and relationships,
and construction and reviewing of the constructed bridge maintenance knowledge graphs.
"""

import os
import json
from agentscope.message import Msg
import agentscope
from agentscope.rag import KnowledgeBank
from agentscope import init as agents_init
from agentscope.pipelines import sequentialpipeline
from agents.decomposer import DecomposerAgent
from agents.extractor import ExtractorAgent
from agents.validator import ValidatorAgent
from agents.corrector import CorrectorAgent
from agents.constructor import ConstructorAgent
from agents.reviewer import ReviewerAgent
from knowledge.knowledge_base import deduplicate_within_data, update_base, load_json, save_json
from utils.neo4j_utils import Neo4jHandler, find_and_merge_duplicate_nodes, combine_paths, save_results_to_json,review_correction
from neo4j import GraphDatabase

# Initialize AgentScope with model configurations
configs = {
    "config_name": "zhipuai",
    "model_type": "zhipuai_chat",
    "model_name": "glm-4-flash",
    "api_key": "your_API_key"
}

configs_emb ={
    "config_name": "zhipuai_emb_config",
    "model_type": "zhipuai_embedding",
    "model_name": "embedding-3",
    "api_key": "your_API_key"
  }

# Initialize AgentScope with the model configurations
agentscope.init(model_configs=[configs,configs_emb])


def main():
    # Read the input text from the file for processing
    file_path = './data/inspection_report.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    # Initialize agents responsible for different stages of processing
    decomposer = DecomposerAgent("decomposer", "zhipuai")
    extractor = ExtractorAgent("extractor",
                                   "You are an expert in bridge engineering. Ensure the extracted triples adhere to domain knowledge.",
                                   "zhipuai",
                                    knowledge_id_list=["bridge_rag"])
    validator = ValidatorAgent("validator", "zhipuai")
    corrector = CorrectorAgent("modifier", "zhipuai")
    constructor = ConstructorAgent("constructor", "zhipuai")
    reviewer = ReviewerAgent("reviewer", "zhipuai")

    # Initialize KnowledgeBank and equip it with the extractor knowledge
    knowledge_bank = KnowledgeBank(configs="configs/knowledge_config.json")
    knowledge_bank.equip(extractor, extractor.knowledge_id_list)

    # Create a message with file content for further processing
    msg = Msg(name='user', content=file_content)

    # Decompose the input file content into relevant keywords
    Seg_results = decomposer(msg)
    keywords = json.loads(Seg_results.content)
    keywords.append("其他")

    for keyword in keywords:
        index = 1

        while True:
            # Construct the filename based on keyword and index
            filename = f"data/{keyword}.txt" if index == 1 else f"data/{keyword}-{index}.txt"

            # Check if the file exists
            if os.path.exists(filename):
                with open(filename, "r", encoding='utf-8') as file:
                    content = file.read()

                # Print the content of the file
                print(content)
                msg_seg = Msg(name='user', content=content)
                print(f'Processing file: {filename}')

                # Extract knowledge from the segmented content
                try:
                    Extraction_result = extractor(msg_seg)

                    Check_result, score = validator(Extraction_result)
                    print(score)

                    # Combine extraction and validation results
                    a = "{'提取结果':{" + Extraction_result.content + "}" + Check_result.content
                    Check_results = Msg(name='user', content=a)

                    max_iterations = 5  # Set maximum iteration count for correction
                    iteration_count = 0 # Initialize iteration counter

                    # If validation score is not 1, attempt to correct the result
                    if score != 1:
                        while score != 1 and iteration_count < max_iterations:
                            modify_result = corrector(Check_results)
                            Check_results, score = validator(modify_result)
                            print(f"Iteration {iteration_count + 1}, score: {score}")
                            iteration_count += 1

                        # If maximum iterations are reached, log the error
                        if iteration_count >= max_iterations:
                            print(f"迭代次数过多 ({max_iterations})，中止操作")
                            # 保存迭代过程的log
                            # 保存错误的句子到error文件中
                            with open("error_sentences.log", "a", encoding="utf-8") as error_file:
                                error_file.write(f"错误句子: {Check_results}\n")
                                error_file.write(f"评分: {score}\n")
                                error_file.write(f"迭代次数: {iteration_count}\n")
                            construction_result = constructor(modify_result)

                        else:
                            # If correction succeeded, proceed with construction
                            construction_result = constructor(modify_result)

                    else:
                        # If the initial score is 1, directly use the Extraction_results for construction
                        construction_result = constructor(Extraction_result)

                    # Only save JSON and perform construction if iteration count is within the limit
                    if iteration_count < max_iterations:
                        data_json = load_json('./knowledge/graph_data.json')
                        base_json = load_json('./knowledge/knowledge_base.json')

                        # Deduplicate data before updating knowledge base
                        deduplicated_data_json = deduplicate_within_data(data_json)
                        updated_base_json = update_base(deduplicated_data_json, base_json)

                        # Save the updated knowledge base
                        save_json(updated_base_json, './knowledge/knowledge_base.json')

                    # Print the construction results for the file
                    print(f'Construction results for {filename}: {construction_result}')

                except Exception as e:
                    # Log any errors that occur during file processing
                    print(f'Error processing file {filename}: {e}')
            else:
                break  # Exit the loop if the file doesn't exist
            index += 1

    # Initialize the Neo4j handler for graph database interaction
    handler = Neo4jHandler()

    with handler.driver.session() as session:
        # Execute write operations to merge duplicate nodes in Neo4j
        session.execute_write(find_and_merge_duplicate_nodes)

    # Getting the results of constructed KG using extractor, validator, corrector, and constructor
    try:
        # Find root nodes in the graph
        root_nodes = handler.find_root_nodes()
        all_results = []

        # For each root node, find and combine all paths
        for root_node in root_nodes:
            root_id = root_node["id"]
            paths = handler.find_full_paths_from_root(root_id)
            combined_results = combine_paths(paths)
            all_results.extend(combined_results)

        # Save the combined results as a JSON file
        save_results_to_json(all_results)
    finally:
        # Close the Neo4j handler session
        handler.close()

    # Load the KG results for further review in the KG level
    results = load_json("results.json")
    sentence = [content["sentence"] for content in results]
    msg_seg = Msg(name='user', content=sentence)

    # Execute a sequential pipeline of reviewer
    Reviwer_result = sequentialpipeline([reviewer], msg_seg)

    # Perform review corrections of knowledge graph on the results
    review_correction(results)



if __name__ == "__main__":
    main()



