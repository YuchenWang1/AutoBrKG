"""
Neo4j Utilities module.

Provides a Neo4jHandler class and related functions for processing graph data.
"""

import json
from neo4j import GraphDatabase
from utils.config_loader import NEO4J_CONFIG
from knowledge.knowledge_base import load_json

class Neo4jHandler:
    """
    Neo4jHandler provides methods to interact with a Neo4j database.
    """

    def __init__(self):
        # Initialize the Neo4j database driver using the configuration
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG["neo4j_uri"],
            auth=(NEO4J_CONFIG["neo4j_username"], NEO4J_CONFIG["neo4j_password"])
        )

    def close(self):
        # Close the connection to the Neo4j driver
        self.driver.close()

    def find_root_nodes(self):
        """
        Find root nodes (nodes with no incoming relationships).

        Returns:
            list: List of root node records.
        """
        with self.driver.session() as session:
            result = session.execute_read(self._find_and_return_root_nodes)
            return result

    def find_full_paths_from_root(self, root_id):
        """
        Find full paths from a given root node.

        Args:
            root_id (int): The ID of the root node.

        Returns:
            list: List of paths (nodes and relationships).
        """
        with self.driver.session() as session:
            result = session.execute_read(self._find_and_return_full_paths, root_id)
            return result

    def update_node_property(self, node_id, correction, suggestion):
        """
        Update a node's property by replacing a specified text.

        Args:
            node_id (int): Node ID.
            correction (str): Text to be corrected.
            suggestion (str): Replacement text.
        """
        with self.driver.session() as session:
            session.execute_write(self._update_node_property, node_id, correction, suggestion)

    def update_relationship_property(self, rel_id, correction, suggestion):
        """
        Update a relationship's property by replacing a specified text.

        Args:
            rel_id (int): Relationship ID.
            correction (str): Text to be corrected.
            suggestion (str): Replacement text.
        """
        with self.driver.session() as session:
            session.execute_write(self._update_relationship_property, rel_id, correction, suggestion)

    @staticmethod
    def _find_and_return_root_nodes(tx):
        """
        Helper function to find root nodes in the graph.

        Args:
            tx: Neo4j transaction.

        Returns:
            list: List of root nodes and their properties.
        """
        query = (
            "MATCH (n) WHERE NOT (n)<--() RETURN id(n) as id, n"
        )
        result = tx.run(query)
        return [{"id": record["id"], "properties": record["n"]} for record in result]

    @staticmethod
    def _find_and_return_full_paths(tx, root_id):
        """
        Helper function to find full paths starting from a root node.

        Args:
            tx: Neo4j transaction.
            root_id (int): ID of the root node.

        Returns:
            list: List of paths (nodes and relationships).
        """
        query = (
            f"MATCH p=(n)-[*]->(m) WHERE id(n) = {root_id} AND NOT (m)-->() RETURN p"
        )
        result = tx.run(query)
        paths = []
        for record in result:
            path = record["p"]
            nodes = [{"id": node.id, "labels": list(node.labels), "properties": dict(node)} for node in path.nodes]
            relationships = [{"id": rel.id, "type": rel.type, "properties": dict(rel)} for rel in path.relationships]
            paths.append((nodes, relationships))
        return paths

    @staticmethod
    def _update_node_property(tx, node_id, correction, suggestion):
        """
        Helper function to update a node's property in the graph.

        Args:
            tx: Neo4j transaction.
            node_id (int): Node ID.
            correction (str): Text to be corrected.
            suggestion (str): Replacement text.
        """
        query = (
            "MATCH (n) WHERE id(n) = $node_id "
            "SET n.name = apoc.text.replace(n.name, $correction, $suggestion) "
            "RETURN n"
        )
        tx.run(query, node_id=node_id, correction=correction, suggestion=suggestion)

    @staticmethod
    def _update_relationship_property(tx, rel_id, correction, suggestion):
        """
        Helper function to update a relationship's property in the graph.

        Args:
            tx: Neo4j transaction.
            rel_id (int): Relationship ID.
            correction (str): Text to be corrected.
            suggestion (str): Replacement text.
        """
        query = (
            "MATCH ()-[r]->() WHERE id(r) = $rel_id "
            "SET r.name = apoc.text.replace(r.name, $correction, $suggestion) "
            "RETURN r"
        )
        tx.run(query, rel_id=rel_id, correction=correction, suggestion=suggestion)

def combine_paths(paths):
    """
    Combine Neo4j paths into a sentence format.

    Args:
        paths (list): List of paths, each containing nodes and relationships.

    Returns:
        list: List of dictionaries with combined sentence and details.
    """
    combined_results = []
    for nodes, relationships in paths:
        sentence = ""
        for i in range(len(relationships)):
            labels = ", ".join(f"{label}: {nodes[i]['properties'].get('name', str(nodes[i]['id']))}" for label in nodes[i]["labels"])
            sentence += f"{labels} - {relationships[i]['type']} - "
        last_node = nodes[-1]
        last_labels = ", ".join(f"{label}: {last_node['properties'].get('name', str(last_node['id']))}" for label in last_node["labels"])
        sentence += last_labels
        combined_results.append({
            "sentence": sentence.strip(" - "),
            "nodes": nodes,
            "relationships": relationships
        })
    return combined_results

def save_results_to_json(results, file_path="results.json"):
    """
    Save results to a JSON file.

    Args:
        results (list): Results data.
        file_path (str): Destination file path.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results have been saved to {file_path}")

def find_and_merge_duplicate_nodes(tx):
    """
    Find and merge duplicate nodes in the Neo4j graph.

    Args:
        tx: Neo4j transaction.
    """
    # Query to find root nodes that are not connected to other nodes (no incoming relationships)
    root_nodes_query = """
    MATCH (n)
    WHERE NOT (n)<--() AND (n)-->()
    RETURN n, labels(n) as labels, elementId(n) as id
    """
    root_nodes = tx.run(root_nodes_query).data()
    node_groups = {}

    # Group nodes by label and name
    for record in root_nodes:
        node = record['n']
        labels = record['labels']
        label = labels[0]  # assume one label per node
        name = node['name']
        if (label, name) not in node_groups:
            node_groups[(label, name)] = []
        node_groups[(label, name)].append({'node': node, 'id': record['id']})

    # Merge nodes with the same label and name
    for (label, name), nodes in node_groups.items():
        if len(nodes) > 1:
            ids = [node['id'] for node in nodes]
            merge_multiple_nodes(tx, ids)

def merge_multiple_nodes(tx, node_ids):
    """
    Merge multiple nodes into one.

    Args:
        tx: Neo4j transaction.
        node_ids (list): List of node IDs to merge.
    """
    merge_query = """
    UNWIND $node_ids as id
    MATCH(n)
    WHERE elementId(n) = id
    WITH collect(n) as nodes
    CALL apoc.refactor.mergeNodes(nodes, {
         properties: {
         id: 'discard',
         otherProperties: 'combine'
         },
         mergeRels: true
    }) YIELD node
    SET node.id = nodes[0].id
    RETURN node
    """
    result = tx.run(merge_query, node_ids=node_ids).single()
    merged_node_id = result['node'].element_id
    child_nodes_query = """
    MATCH (parent)-[]->(child)
    WHERE elementId(parent) = $id
    RETURN child, elementId(child) as id
    """

    # Find child nodes related to the merged node
    child_nodes = tx.run(child_nodes_query, id=merged_node_id).data()
    child_groups = {}

    # Group child nodes by their name
    for child in child_nodes:
        name = child['child']['name']
        if name not in child_groups:
            child_groups[name] = []
        child_groups[name].append({'node': child['child'], 'id': child['id']})

    # Merge child nodes with the same name
    for name, children in child_groups.items():
        if len(children) > 1:
            child_ids = [child['id'] for child in children]
            merge_multiple_nodes(tx, child_ids)

def review_correction(results):
    """
    Review and correct errors in the results based on predefined check results.

    Args:
        results (list): List of results data to be reviewed and corrected.
    """
    checks_json_path = "check_results.json"
    checks = load_json(checks_json_path)
    handler=Neo4jHandler()

    # Step 1: Check if there are any sequence errors
    for item in checks:
        if "顺序错误" in item["存在错误"]:
            # Step 1: Get the corrected order from the check result
            correct_order = checks[0]['修正结果'].split(" - ")

            # Step 2: Separate entities and relationships from the corrected order
            entities = [item.split(":")[1] for item in correct_order if ":" in item]
            relationships = [item for item in correct_order if ":" not in item]

            # Step 3: Get all nodes and relationships from the results
            node_map = {node['properties']['name']: node['id'] for node in results[0]['nodes']}

            # Step 4: Delete existing relationships
            relationships_to_delete = [rel['id'] for rel in results[0]['relationships']]

            with handler.driver.session() as session:
                for rel_id in relationships_to_delete:
                    session.run(f"MATCH ()-[r]->() WHERE r.id = {rel_id} DELETE r")

            # Step 5: Recreate relationships based on the corrected order
            with handler.driver.session() as session:
                for i in range(len(entities) - 1):
                    start_entity = entities[i]
                    end_entity = entities[i + 1]
                    start_entity = start_entity.strip()
                    end_entity = end_entity.strip()

                    # Get the node IDs of the start and end entities
                    start_node_id = node_map[start_entity]
                    end_node_id = node_map[end_entity]

                    # Get the relationship type
                    relationship_type = relationships[i]

                    # Create new relationships based on the corrected order
                    session.run(f"""
                        MATCH (a), (b)
                        WHERE a.id = {start_node_id} AND b.id = {end_node_id}
                        CREATE (a)-[r:{relationship_type}]->(b)
                    """)

        elif "实体错误" in item["存在错误"]:
            # Step: Handle entity errors
            corrected_entity = item["修正结果"].split(":")[1].strip()  # Corrected entity name
            incorrect_entity = item["句子"].split(":")[1].strip()  # Original entity name

            # Step: Find the corresponding node ID for the entity
            with handler.driver.session() as session:
                session.run(f"""
                    MATCH (n {{name: '{incorrect_entity}'}})
                    SET n.name = '{corrected_entity}'
                """)

        elif "关系错误" in item["存在错误"]:
            # Step: Handle relationship errors
            corrected_relationship = item["修正结果"].split(" - ")[1]  # Corrected relationship type
            incorrect_relationship = item["句子"].split(" - ")[1]  # Original relationship type

            # Step: Find the corresponding relationship ID and update its type
            with handler.driver.session() as session:
                session.run(f"""
                    MATCH ()-[r:{incorrect_relationship}]->()
                    SET r.type = '{corrected_relationship}'
                """)

