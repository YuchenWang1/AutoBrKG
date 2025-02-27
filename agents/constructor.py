"""
ConstructionAgent module.

This agent integrates extraction results into a structured graph data file.
It appends new data to the graph data JSON file.
"""

import json
import os
import uuid
import re
import threading
from agentscope.agents import AgentBase
from agentscope.models import ModelResponse
from agentscope.message import Msg
from agentscope.prompt import PromptEngine
from typing import Optional
from utils.neo4j_utils import Neo4jHandler

# Converts extracted data into the target format (graph data format)
def convert_to_target_format(data):
    result = []

    for item in data:
        # Process the triples
        relation_results = []
        for triple in item["三元组"]:
            # Split the triple components
            parts = triple.split(">")
            主实体信息, 关系, 宾实体信息 = parts[0], parts[1], parts[2]

            主实体类型, 主实体 = 主实体信息.split(":")
            宾实体类型, 宾实体 = 宾实体信息.split(":")

            # Add the processed relation to the results
            relation_results.append({
                "关系": 关系,
                "主实体类型": 主实体类型,
                "宾实体类型": 宾实体类型,
                "主实体": 主实体,
                "宾实体": 宾实体
            })

        # Process the attributes
        attribute_results = []
        for attribute in item["属性"]:
            # Split the attribute components
            parts = attribute.split(">")
            实体, 属性类型, 属性值 = parts[0], parts[1], parts[2]

            # Add the processed attribute to the results
            attribute_results.append({
                "实体": 实体,
                "属性类型": 属性类型,
                "属性值": 属性值
            })

        # Append the final result for each item
        result.append({
            "文本": item["文本"],
            "关系提取结果": relation_results,
            "属性提取结果": attribute_results
        })

    return result

class ConstructorAgent(AgentBase):
    """
    The ConstructorAgent class handles the process of integrating extracted data into a structured format and saving it.
    """

    def __init__(
            self,
            name: str,
            model_config_name: str,
            use_memory: bool = True,
            memory_config: Optional[dict] = None,
    ) -> None:

        super().__init__(
            name=name,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )

    def parse_json(self, response: ModelResponse) -> ModelResponse:
        pattern = r"```json(.*)```"
        match = re.search(pattern, response.text, re.DOTALL)
        json_text = match.group(1) if match else response.text
        resu = ModelResponse(text=json_text)
        return resu

    def clean_json_string(self, data: str) -> str:  # 使用正则表达式匹配并删除前面除 [ 之外的全部内容，以及最后一个 ] 之后的全部内容
        match = re.search(r'\[.*\]', data, re.DOTALL)
        if match:
            return match.group(0)
        else:
            raise ValueError("No valid JSON array found in the string")

    def create_graph(self, data):
        """
        Create the graph data by assigning unique IDs to entities and relations, then saving the data to a Neo4j graph.
        """
        data = self.clean_json_string(data)
        data = json.loads(data)
        lock = threading.Lock()

        with lock:
            entity_ids = {}

            # Assign unique IDs and update the JSON data
            for entry in data:
                for relation in entry["关系提取结果"]:
                    主实体 = relation.get('主实体')
                    宾实体 = relation.get('宾实体')

                    if 主实体 not in entity_ids:
                        entity_ids[主实体] = str(uuid.uuid4())
                    if 宾实体 not in entity_ids:
                        entity_ids[宾实体] = str(uuid.uuid4())
                    if relation.get('宾实体类型') in ['病害', '构件部位']:
                        entity_ids[宾实体] = str(uuid.uuid4())

                    relation['主实体_id'] = entity_ids[主实体]
                    relation['宾实体_id'] = entity_ids[宾实体]

                for attribute in entry["属性提取结果"]:
                    实体 = attribute.get('实体')
                    if 实体 not in entity_ids:
                        entity_ids[实体] = str(uuid.uuid4())
                    attribute['实体_id'] = entity_ids[实体]

            handler = Neo4jHandler()
            with handler.driver.session() as session:
                for entry in data:
                    for relation in entry["关系提取结果"]:
                        主实体类型 = relation.get('主实体类型')
                        宾实体类型 = relation.get('宾实体类型')
                        主实体 = relation.get('主实体')
                        宾实体 = relation.get('宾实体')
                        关系 = relation.get('关系')
                        主实体_id = relation.get('主实体_id')
                        宾实体_id = relation.get('宾实体_id')

                        query = (
                            f"MERGE (a:{主实体类型} {{name: '{主实体}', unique_id: '{主实体_id}'}}) "
                            f"MERGE (b:{宾实体类型} {{name: '{宾实体}', unique_id: '{宾实体_id}'}}) "
                            f"MERGE (a)-[r:{关系}]->(b) "
                        )
                        session.run(query)

                    for attribute in entry["属性提取结果"]:
                        实体 = attribute.get('实体')
                        属性类型 = attribute.get('属性类型')
                        属性值 = attribute.get('属性值', '')
                        实体_id = attribute.get('实体_id')

                        query = (
                            f"MATCH (a {{name: '{实体}', unique_id: '{实体_id}'}}) "
                            f"SET a.{属性类型} = '{属性值}'"
                        )
                        session.run(query)

        handler.driver.close()

    def reply(self, x: dict = None) -> dict:
        if x is not None:
            self.memory.add(x)

        self.engine = PromptEngine(self.model)

        # Define the base prompt for construction tasks, recommending using Chinese prompt for coding test.
        baseprompt = """
        待检查内容：{context}
        请检查上述内容格式是否为json格式,注意一定保证最外层为中括号[]，且保证均为双引号，不允许单引号生成
        特别注意，只需要最终返回正确的内容```json your_check_here ``` 不需要给我其他任何内容。
        """

        # For english report. Recommending using Chinese prompt for coding test.
        baseprompt_en = """
        Content to be checked: {context}
        Please check the above content format is json format, note that must ensure that the outermost parentheses [], and ensure that all double quotes, do not allow single quotes generated!
        Special note, only need to return the correct content ```json your_check_here ``` do not need to give me any other content.
        """

        input_data = json.loads(x.content)

        # Check if the graph data file exists and load the existing data
        if os.path.exists('./knowledge/graph_data.json'):
            with open('./knowledge/graph_data.json', 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Add the new data to the existing data
        if isinstance(input_data, list):
            existing_data.extend(input_data)
        else:
            existing_data.append(input_data)

        # Save the updated data back to the JSON file
        with open('./knowledge/graph_data.json', 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        output_data = convert_to_target_format(input_data)

        variables = {"context": output_data}

        finalprompt = self.engine.join(
            baseprompt, format_map=variables
        )

        prompt = self.model.format(
            Msg("system", finalprompt, role="system"),
            self.memory
            and self.memory.get_memory()
            or x,
        )

        response = self.model(
            prompt,
            parse_func=self.parse_json,
        ).text

        msg = Msg(self.name, response, role="constructor")

        self.speak(msg)

        self.create_graph(response)

        if self.memory:
            self.memory.add(msg)

        return msg
