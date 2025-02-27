
import json
import os
import re
from typing import List, Dict, Set, Tuple


def calculate_precision_recall_f1(gold: Set, pred: Set) -> Tuple[float, float, float]:
    if len(pred) == 0:
        return 0, 0, 0
    p = len(gold.intersection(pred)) / len(pred)
    r = len(gold.intersection(pred)) / len(gold)
    f1 = 2 * ((p * r) / (p + r)) if (p + r) > 0 else 0
    return p, r, f1


def normalize_triple(sub_label: str, rel_label: str, obj_label: str) -> str:
    sub_label = re.sub(r"(\s+)", '', sub_label).lower()
    rel_label = re.sub(r"(\s+)", '', rel_label).lower()
    obj_label = re.sub(r"(\s+)", '', obj_label).lower()
    return f"{sub_label}_{rel_label}_{obj_label}"


def read_jsonl(jsonl_path: str) -> List:
    data = []
    with open(jsonl_path, encoding='utf-8') as in_file:
        for line in in_file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} for line: {line.strip()}")
    return data


def convert_to_dict(data: List[Dict], id_name: str = "id") -> Dict:
    return {item[id_name]: item for item in data}


def evaluate_entities_and_relationships(ground_truth: List[Dict], system_output: List[Dict]) -> Dict:
    entity_metrics = {}
    relation_metrics = {}

    for item in ground_truth:
        for triple in item['triples']:
            sub = triple['sub']['name']
            obj = triple['obj']['name']
            entity_type_sub = triple['sub']['type']
            entity_type_obj = triple['obj']['type']

            # Collect ground truth entities
            if entity_type_sub not in entity_metrics:
                entity_metrics[entity_type_sub] = {'gold': set(), 'pred': set()}
            if entity_type_obj not in entity_metrics:
                entity_metrics[entity_type_obj] = {'gold': set(), 'pred': set()}

            entity_metrics[entity_type_sub]['gold'].add(sub)
            entity_metrics[entity_type_obj]['gold'].add(obj)

            # Collect ground truth relationships
            rel = triple['rel']['name']
            if rel not in relation_metrics:
                relation_metrics[rel] = {'gold': set(), 'pred': set()}
            relation_metrics[rel]['gold'].add(normalize_triple(sub, rel, obj))

    for item in system_output:
        for triple in item['triples']:
            sub = triple['sub']['name']
            obj = triple['obj']['name']
            entity_type_sub = triple['sub']['type']
            entity_type_obj = triple['obj']['type']

            # Collect predicted entities
            if entity_type_sub in entity_metrics:
                entity_metrics[entity_type_sub]['pred'].add(sub)
            if entity_type_obj in entity_metrics:
                entity_metrics[entity_type_obj]['pred'].add(obj)

            # Collect predicted relationships
            rel = triple['rel']['name']
            if rel in relation_metrics:
                relation_metrics[rel]['pred'].add(normalize_triple(sub, rel, obj))

    evaluation_results = {}

    # Evaluate entity metrics
    for entity_type, metrics in entity_metrics.items():
        precision, recall, f1 = calculate_precision_recall_f1(metrics['gold'], metrics['pred'])
        evaluation_results[f"entity_{entity_type}"] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Evaluate relation metrics
    for rel, metrics in relation_metrics.items():
        precision, recall, f1 = calculate_precision_recall_f1(metrics['gold'], metrics['pred'])
        evaluation_results[f"relation_{rel}"] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return evaluation_results


def main():
    eval_config_path = "config/bridge.jsonl"  # 指定 JSONL 配置文件路径
    output_file_path = "../data-class/bridge/output_class.json"  # 指定输出结果文件路径

    if not os.path.exists(eval_config_path):
        print(f"Evaluation config file is not found in path: {eval_config_path}")
        return

    eval_inputs = read_jsonl(eval_config_path)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for onto in eval_inputs:
            system_output = convert_to_dict(read_jsonl(onto['sys']))
            ground_truth = convert_to_dict(read_jsonl(onto['gt']))

            evaluation_results = evaluate_entities_and_relationships(ground_truth.values(), system_output.values())

            output_file.write(f"Evaluation results for {onto['id']}:\n")
            for item, metrics in evaluation_results.items():
                if item.startswith("entity_"):
                    output_file.write(
                        f"Entity: {item[7:]}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}\n"
                    )
                elif item.startswith("relation_"):
                    output_file.write(
                        f"Relation: {item[9:]}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}\n"
                    )
            output_file.write("\n")  # 添加空行以分隔不同的评估结果


if __name__ == "__main__":
    main()
