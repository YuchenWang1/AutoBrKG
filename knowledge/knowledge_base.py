"""
KnowledgeBase module.

This module defines a KnowledgeBase class for storing and retrieving knowledge examples.
"""
import json
from sentence_transformers import SentenceTransformer, util
from zhipuai import ZhipuAI


# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to save JSON data to a file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_embedding(text):
    client = ZhipuAI(api_key="your_API_key")
    # Request the embedding for the provided text using the specified model
    response = client.embeddings.create(
        model="embedding-3",  # Specify the model to use for embedding
        input=text,
    )
    emb=response.data[0].embedding
    return emb

# Function to calculate the cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    cosine_sim = util.cos_sim(embedding1, embedding2)
    return cosine_sim.item()

# Function to check if a given text embedding is similar to any in the list of embeddings
def is_similar(text_embedding, embeddings_list, threshold=0.85):
    for existing_embedding in embeddings_list:
        similarity = calculate_similarity(text_embedding, existing_embedding)
        if similarity >= threshold:
            return True
    return False

# Function to deduplicate data by comparing the text embeddings
def deduplicate_within_data(data_json, threshold=0.85):
    unique_data = []
    embeddings_list = []

    for entry in data_json:
        text = entry["文本"]
        text_embedding = get_embedding(text)

        if not is_similar(text_embedding, embeddings_list, threshold):
            unique_data.append(entry)
            embeddings_list.append(text_embedding)

    return unique_data

# Function to update the knowledge base with new data if not already present
def update_base(data_json, base_json, threshold=0.85):
    base_texts = [entry["文本"] for entry in base_json]
    base_embeddings = [get_embedding(text) for text in base_texts]

    for entry in data_json:
        text = entry["文本"]
        text_embedding = get_embedding(text)

        if not is_similar(text_embedding, base_embeddings, threshold):
            base_json.append(entry)
            base_embeddings.append(text_embedding)  # 更新嵌入列表

    return base_json

# Class to represent the Knowledge Base
class KnowledgeBase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_knowledge()
        # 初始化 SentenceTransformer 模型

    def load_knowledge(self):
        """
        Load the knowledge base from a JSON file.
        If the file is not found, initialize with an empty list.
        """
        try:
            with open(self.file_path, 'r') as f:
                self.knowledge = json.load(f)
        except FileNotFoundError:
            self.knowledge = []

    def save_knowledge(self):
        """
        Save the current knowledge to the JSON file.
        """
        with open(self.file_path, 'w') as f:
            json.dump(self.knowledge, f, ensure_ascii=False, indent=4)

    def add_example(self, example):
        """
        Add a new example to the knowledge base.

        Args:
            example (dict): The example data to be added.
        """
        self.knowledge.append(example)
        self.save_knowledge()

    def compute_similarity(self, text1, text2):
        """
        Compute the cosine similarity between two pieces of text.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The cosine similarity between the two texts.
        """
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)
        cosine_sim = util.cos_sim(embedding1, embedding2)
        return cosine_sim.item()

    def search_similar(self, query, threshold=0.85):
        """
        Search for similar examples in the knowledge base based on a query.

        Args:
            query (str): The query text to search for.
            threshold (float): The similarity threshold for matching.

        Returns:
            dict or None: The most similar example if found, else None.
        """
        for example in self.knowledge:
            similarity = self.compute_similarity(example['文本'], query)
            if similarity > threshold:
                return example
        return None
