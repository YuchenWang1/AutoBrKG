# AutoBrKG-Bridge Inspection Knowledge Graph Construction

This project automates the construction of a bridge maintenance knowledge graph from inspection reports. It covers the entire process from text segmentation, information extraction, validation, correction, and knowledge graph construction to final review. The system employs dedicated agents for each stage, ensuring a modular and flexible pipeline.

-----

## Overview

The system extracts entities and relationships from raw bridge inspection reports, constructs a knowledge graph, and validates the extracted data. The key components include:

  - **DecomposerAgent**: Segments the input text based on specified keywords.
  - **ExtractorAgent**: Extracts entities and relationships from the segmented content.
  - **ValidatorAgent**: Validates the extracted content against predefined rules.
  - **CorrectorAgent**: Corrects any issues identified during validation.
  - **ConstructorAgent**: Integrates the corrected data into a structured graph and stores it in the knowledge base.
  - **ReviewerAgent**: Reviews the final results for logical consistency, linguistic accuracy, and structural correctness.

-----

## File Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── constructor.py           # ConstructorAgent
│   ├── corrector.py             # CorrectorAgent
│   ├── decomposer.py            # DecomposerAgent
│   ├── extractor.py             # ExtractorAgent
│   ├── reviewer.py              # ReviewerAgent
│   └── validator.py             # ValidatorAgent
├── configs/
│   └── model_config.json        # Model-related configurations (chat model, embedding API, etc.)
├── data/
│   └── inspection_report.txt    # Raw bridge inspection report
├── evaluation/
│   ├── data/
│   │   └── bridge/
│   │       ├── bridge.jsonl
│   │       ├── bridge-pre.jsonl
│   │       ├── bridge_ontology.json
│   │       ├── bridge_output.jsonl
│   │       ├── bridge_selected.txt
│   │       └── ouput.jsonl
│   ├── data-class/
│   │   └── bridge/
│   │       ├── bridge.jsonl
│   │       ├── bridge-pre.jsonl
│   │       └── output_class.json
│   ├── preprocessing/
│   │   └── data_transfer_pre.py # Script to generate evaluation-ready files
│   └── src/
│       ├── config/
│       │   ├── bridge.json      # Configuration file for triple-level evaluation
│       │   └── bridge.jsonl     # Configuration file for ontology-level evaluation
│       ├── evl.py               # Main script for triple-level evaluation
│       └── evl-class.py         # Main script for ontology-level evaluation
├── knowledge/
│   ├── bridge.pdf               # RAG knowledge file for bridge data
│   ├── knowledge_base.json      # Stored knowledge base data
│   ├── knowledge_base.py        # Knowledge base management
│   └── rag_utils.py             # RAG utility functions
├── run/                         # Directory for runtime files
├── utils1/
│   ├── __init__.py
│   ├── config_loader.py         # Utility for loading configurations
│   ├── instance_data.ttl        # Instance data in Turtle format
│   ├── ontology.py              # Ontology-related utilities
│   └── ontology.ttl             # Ontology definition in Turtle format
├── __init__.py
├── main.py                      # Main entry point of the application
├── metric_information.md        # Documentation for evaluation metrics
└── readme.md                    # This file
```

-----

## Configuration

### API Configuration

In **`main.py`**, set up the **chat** and **embedding** API configurations based on your chosen service. For example:

```python
configs = {
    "config_name": "zhipuai",
    "model_type": "zhipuai_chat",
    "model_name": "glm-4-flash",
    "api_key": "your-chat-api-key"
}

configs_emb = {
    "config_name": "zhipuai_emb_config",
    "model_type": "zhipuai_embedding",
    "model_name": "embedding-3",
    "api_key": "your-embedding-api-key"
}
```

In **`knowledge_base.py`**, you also need to configure the embedding API used for similarity calculations and knowledge retrieval.

-----

## Pipeline

1.  **Text Segmentation**: `DecomposerAgent` segments the `inspection_report.txt` based on predefined keywords.
2.  **Information Extraction**: `ExtractorAgent` extracts entities and relationships from the segmented content.
3.  **Validation**: `ValidatorAgent` checks the extracted data against predefined rules.
4.  **Correction**: `CorrectorAgent` corrects any issues identified during validation.
5.  **Knowledge Graph Construction**: `ConstructorAgent` structures the corrected data into a knowledge graph and appends it to the knowledge base.
6.  **Review**: `ReviewerAgent` examines the final results, ensuring logical consistency, linguistic correctness, and structural accuracy.

-----

## Usage

1.  Place your input bridge inspection report (`inspection_report.txt`) in the `data` folder.
2.  Ensure all configuration files in the `configs` folder are correctly set up.
3.  Run the main script:
    ```bash
    python main.py
    ```
4.  After execution, a corresponding knowledge graph will be generated, and a JSON file containing the structured data will also be created locally.

-----

## Evaluation

This project provides an evaluation tool for the extracted results, located in the `evaluation` folder:

  - **`data/bridge`**: Contains files for evaluating triple-level extraction results.
  - **`data-class/bridge`**: Contains files for evaluating ontology-level extraction results.
  - **`preprocessing/`**: Contains scripts (e.g., `data_transfer_pre.py`) to convert raw results into an evaluation-ready format.
  - **`src/`**: Contains the main evaluation code and configuration files.
      - **`config/bridge.json`**: Configuration file for evaluation.
      - **`evl.py`**: Main script for triple-level evaluation.
      - **`evl-class.py`**: Main script for ontology-level evaluation.

### How to Run the Evaluation

1.  **Preprocessing**: First, run the script(s) in the `evaluation/preprocessing` folder (e.g., `data_transfer_pre.py`) to generate the files needed for evaluation.
2.  **Triple-Level Evaluation**:
    ```bash
    cd evaluation/src
    python evl.py --eval_config_path config/bridge.json
    ```
3.  **Ontology-Level Evaluation**:
    ```bash
    cd evaluation/src
    python evl-class.py
    ```
4.  All evaluation outputs will be stored in the respective `bridge` folders.

-----

**Note**: If you wish to use English prompts and processing, you need to translate `bridge.pdf` (and other Chinese knowledge files) into English, and modify the prompts accordingly. By default, the system operates in Chinese for information extraction and evaluation.

-----

## Fact Construction Performance

This method evaluates fact construction performance using precision, recall, and F1 score. 

### Overall Metrics

  - **Precision:** Calculated by dividing the number of correct triples (those matching the ground truth) by the total number of generated triples.
  - **Recall:** Calculated by dividing the number of correct triples by the total number of ground truth triples.
  - **F1 Score:** The harmonic mean of precision and recall, providing a comprehensive and reliable evaluation metric, especially for imbalanced data.

The following formulas are presented in plain text:

  - **Precision = TP / (TP + FP)**
  - **Recall = TP / (TP + FN)**
  - **F1 score = (2 × Precision × Recall) / (Precision + Recall)**

Where:

  - **TP:** Number of correctly predicted triples (True Positive)
  - **FP:** Number of incorrectly predicted triples (False Positive)
  - **FN:** Number of ground truth triples that were not extracted (False Negative)
-----

## Domain Consistency Evaluation

Since general large language models (LLMs) are used instead of domain-specific LLMs, the constructed knowledge graph may include non-domain information. To evaluate domain consistency, the **Hallucination Metric** is employed.

The hallucination metric measures the extent of generated content that is nonsensical or unfaithful to the source information. The following plain text formulas are used:

  - **Subject Hallucination (SH) = 1 - (T\_sub / T)**
  - **Relation Hallucination (RH) = 1 - (T\_rel / T)**
  - **Object Hallucination (OH) = 1 - (T\_obj / T)**

Where:

  - **T\_sub:** Number of triples with correct subjects.
  - **T\_rel:** Number of triples with correct relations.
  - **T\_obj:** Number of triples with correct objects.
  - **T:** Total number of generated triples.

All metrics are computed using exact matches in accordance with the domain requirements.