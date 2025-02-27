"""
config_loader.py

This module loads configuration settings from JSON files in the `config` directory.
"""

import json
import os

CONFIG_DIR = "configs"

def load_config(filename):
    """
    Load a JSON configuration file from the config directory.

    Args:
        filename (str): Name of the JSON config file.

    Returns:
        dict or list: Parsed JSON content.
    """
    file_path = os.path.join(CONFIG_DIR, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file {filename} not found in {CONFIG_DIR}.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {e}")
        return None

# Load all configurations
MODEL_CONFIG = load_config("model_config.json")
KNOWLEDGE_CONFIG = load_config("knowledge_config.json")
NEO4J_CONFIG = load_config("neo4j_config.json")
