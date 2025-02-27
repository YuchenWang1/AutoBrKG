
# #### 主要功能概述：
# 1. **命令行参数解析**：通过`argparse`模块解析传入的`--eval_config_path`，该路径指向包含评价配置的JSON文件，包含系统输出、本体、人工标注文件等路径。
# 2. **加载配置文件**：通过`load_config`函数读取配置文件，并解析出评价所需的各种文件路径。
# 3. **评估三元组**：
#     - `calculate_precision_recall_f1`：计算系统输出与标准答案的精确率、召回率和F1值。
#     - `get_subject_object_hallucinations`：检查系统输出中的主语和宾语是否符合本体或出现在句子中。
#     - `get_ontology_conformance`：检查关系是否符合本体，计算本体一致性和关系幻觉。
# 4. **迭代评估**：遍历每个本体，逐一比较系统输出与人工标注的三元组，计算各项指标并将其保存到JSONL文件中。
# 5. **全局评估**：通过全局的precision, recall, F1值对所有本体进行汇总，并生成最终评估报告。
#
# ### 改进为支持中文的方法：
# 1. **去除或替换英语词干提取**：因为原代码中使用了 `PorterStemmer` 来做英文单词的词干提取，而中文不需要词干提取，建议直接去掉 `PorterStemmer` 的使用。
# 2. **替换分词器**：`nltk.tokenize.word_tokenize` 是用于英语的分词工具，但在中文中需要使用专门的中文分词工具，如 `jieba`。
# 3. **修改正则表达式**：处理空格或下划线的正则表达式需要检查是否适用于中文字符。
#

import argparse
import sys
import os
import json
import re
from typing import List, Dict, Set, Tuple
import jieba  # 中文分词工具

def calculate_precision_recall_f1(gold: Set, pred: Set) -> (float, float, float):
    if len(pred) == 0:
        return 0, 0, 0
    p = len(gold.intersection(pred)) / len(pred)
    r = len(gold.intersection(pred)) / len(gold)
    if p + r > 0:
        f1 = 2 * ((p * r) / (p + r))
    else:
        f1 = 0
    return p, r, f1

def get_subject_object_hallucinations(ontology, test_sentence, triples) -> (float, float):
    if len(triples) == 0:
        return 0, 0

    # 对句子和本体概念进行中文分词
    test_sentence += " ".join([c["label"] for c in ontology['concepts']])
    segmented_sentence = "".join(jieba.cut(test_sentence))  # 使用jieba进行中文分词
    normalized_sentence = re.sub(r"(\s+)", '', segmented_sentence).lower()

    num_subj_hallucinations, num_obj_hallucinations = 0, 0
    for triple in triples:
        normalized_subject = clean_entity_string(triple[0])
        normalized_object = clean_entity_string(triple[2])

        if normalized_sentence.find(normalized_subject) == -1:
            num_subj_hallucinations += 1
        if normalized_sentence.find(normalized_object) == -1:
            num_obj_hallucinations += 1

    subj_hallucination = num_subj_hallucinations / len(triples)
    obj_hallucination = num_obj_hallucinations / len(triples)
    return subj_hallucination, obj_hallucination

def get_ontology_conformance(ontology: Dict, triples: List) -> (float, float):
    if len(triples) == 0:
        return 1, 0
    ont_rels = [rel['label'].replace(" ", "_") for rel in ontology['relations']]
    num_rels_conformant = len([tr for tr in triples if tr[1] in ont_rels])
    ont_conformance = num_rels_conformant / len(triples)
    rel_hallucination = 1 - ont_conformance
    return ont_conformance, rel_hallucination

def normalize_triple(sub_label: str, rel_label: str, obj_label: str) -> str:
    sub_label = re.sub(r"(\s+)", '', sub_label).lower()
    rel_label = re.sub(r"(\s+)", '', rel_label).lower()
    obj_label = re.sub(r"(\s+)", '', obj_label).lower()
    tr_key = f"{sub_label}{rel_label}{obj_label}"
    return tr_key

def clean_entity_string(entity: str) -> str:
    segmented_entity = "".join(jieba.cut(entity))  # 中文分词
    normalized_entity = re.sub(r"(\s+)", '', segmented_entity).lower()
    return normalized_entity

def read_jsonl(jsonl_path: str, is_json: bool = True) -> List:
    data = []
    with open(jsonl_path, encoding='utf-8') as in_file:
        for line in in_file:
            if is_json:
                data.append(json.loads(line))
            else:
                data.append(line.strip())
    return data

def load_config(eval_config_path: str) -> Dict:
    raw_config = read_json(eval_config_path)
    onto_list = raw_config['onto_list']
    path_patterns = raw_config["path_patterns"]
    new_config = dict()
    expanded_onto_list = list()
    for onto in onto_list:
        onto_data = dict()
        onto_data["id"] = onto
        for key in path_patterns:
            onto_data[key] = path_patterns[key].replace("$$onto$$", onto)
        expanded_onto_list.append(onto_data)
    new_config["onto_list"] = expanded_onto_list
    new_config["avg_out_file"] = raw_config["avg_out_file"]
    return new_config

def save_jsonl(data: List, jsonl_path: str) -> None:
    with open(jsonl_path, "w", encoding='utf-8') as out_file:
        for item in data:
            out_file.write(f"{json.dumps(item, ensure_ascii=False)}\n")

def append_jsonl(data: Dict, jsonl_path: str) -> None:
    with open(jsonl_path, "a+", encoding='utf-8') as out_file:
        out_file.write(f"{json.dumps(data, ensure_ascii=False)}\n")

def read_json(json_path: str) -> Dict:
    with open(json_path, encoding='utf-8') as in_file:
        return json.load(in_file)

def convert_to_dict(data: List[Dict], id_name: str = "id") -> Dict:
    return {item[id_name]: item for item in data}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config_path', type=str, required=True)
    args = parser.parse_args()

    eval_config_path = args.eval_config_path
    if not os.path.exists(eval_config_path):
        print(f"Evaluation config file is not found in path: {eval_config_path}")
    eval_inputs = load_config(eval_config_path)

    # 初始化全局评估指标
    global_p, global_r, global_f1, global_onto_conf, global_rel_halluc, global_sub_halluc, global_obj_halluc = 0, 0, 0, 0, 0, 0, 0
    # 评估每个本体输出
    for onto in eval_inputs['onto_list']:
        # 初始化本体评估指标
        t_p, t_r, t_f1, t_onto_conf, t_rel_halluc, t_sub_halluc, t_obj_halluc = 0, 0, 0, 0, 0, 0, 0
        sel_t_p, sel_t_r, sel_t_f1, sel_t_onto_conf, sel_t_rel_halluc, sel_t_sub_halluc, sel_t_obj_halluc = 0, 0, 0, 0, 0, 0, 0
        eval_metrics_list = list()
        onto_id = onto['id']
        system_output = convert_to_dict(read_jsonl(onto['sys']))
        ground_truth = convert_to_dict(read_jsonl(onto['gt']))
        ontology = read_json(onto['onto'])
        selected_ids = read_jsonl(onto['selected_ids'], is_json=False) if 'selected_ids' in onto else []

        for sent_id in list(ground_truth.keys()):
            gt_triples = [[tr['sub'], tr['rel'], tr['obj']] for tr in ground_truth[sent_id]['triples']]
            sentence = ground_truth[sent_id]["sent"]

            if sent_id in system_output:
                system_triples = system_output[sent_id]['triples']
                gt_relations = {tr[1].replace(" ", "_") for tr in gt_triples}
                filtered_system_triples = [tr for tr in system_triples if tr[1] in gt_relations]

                # 将系统输出和人工标注的三元组进行归一化以便比较
                normalized_system_triples = {normalize_triple(tr[0], tr[1], tr[2]) for tr in filtered_system_triples}
                normalized_gt_triples = {normalize_triple(tr[0], tr[1], tr[2]) for tr in gt_triples}

                # 计算精确率、召回率和F1值
                precision, recall, f1 = calculate_precision_recall_f1(normalized_gt_triples, normalized_system_triples)

                # 计算本体一致性和关系幻觉
                ont_conformance, rel_hallucination = get_ontology_conformance(ontology, system_triples)

                # 计算主语和宾语的幻觉
                subj_hallucination, obj_hallucination = get_subject_object_hallucinations(ontology, sentence,
                                                                                          system_triples)

                # 如果F1小于1且没有主语和宾语幻觉，输出诊断信息
                if f1 < 1 and len(filtered_system_triples) > 0 and subj_hallucination == 0 and obj_hallucination == 0:
                    print(
                        f"句子: {sentence}\nF1: {f1}\n系统输出: {filtered_system_triples}\n标准答案: {gt_triples}\n\n")

                # 将评价指标保存到列表中
                eval_metrics = {
                    "id": sent_id,
                    "precision": f"{precision:.2f}",
                    "recall": f"{recall:.2f}",
                    "f1": f"{f1:.2f}",
                    "onto_conf": f"{ont_conformance:.5f}",
                    "rel_halluc": f"{rel_hallucination:.5f}",
                    "sub_halluc": f"{subj_hallucination:.5f}",
                    "obj_halluc": f"{obj_hallucination:.5f}",
                    "llm_triples": system_triples,
                    "filtered_llm_triples": filtered_system_triples,
                    "gt_triples": gt_triples,
                    "sent": sentence
                }
                eval_metrics_list.append(eval_metrics)

                # 累积本体的评价指标
                t_p += precision
                t_r += recall
                t_f1 += f1
                t_onto_conf += ont_conformance
                t_rel_halluc += rel_hallucination
                t_sub_halluc += subj_hallucination
                t_obj_halluc += obj_hallucination

                # 如果句子是已选择的句子，累积选择的评价指标
                if sent_id in selected_ids:
                    sel_t_p += precision
                    sel_t_r += recall
                    sel_t_f1 += f1
                    sel_t_onto_conf += ont_conformance
                    sel_t_rel_halluc += rel_hallucination
                    sel_t_sub_halluc += subj_hallucination
                    sel_t_obj_halluc += obj_hallucination

                # 保存每个本体的评估结果
            save_jsonl(eval_metrics_list, onto['output'])

            # 计算本体的平均评价指标
            total_test_cases = len(ground_truth)
            total_selected_test_cases = len(selected_ids)
            average_metrics = {
                "onto": onto_id,
                "type": "all_test_cases",
                "avg_precision": f"{t_p / total_test_cases:.5f}",
                "avg_recall": f"{t_r / total_test_cases:.5f}",
                "avg_f1": f"{t_f1 / total_test_cases:.5f}",
                "avg_onto_conf": f"{t_onto_conf / total_test_cases:.5f}",
                "avg_sub_halluc": f"{t_sub_halluc / total_test_cases:.5f}",
                "avg_rel_halluc": f"{t_rel_halluc / total_test_cases:.5f}",
                "avg_obj_halluc": f"{t_obj_halluc / total_test_cases:.5f}"
            }
            append_jsonl(average_metrics, eval_inputs['avg_out_file'])

            # 累积全局评价指标
            global_p = (t_p / total_test_cases)
            global_r = (t_r / total_test_cases)
            global_f1 = (t_f1 / total_test_cases)
            global_onto_conf = (t_onto_conf / total_test_cases)
            global_sub_halluc = (t_sub_halluc / total_test_cases)
            global_rel_halluc = (t_rel_halluc / total_test_cases)
            global_obj_halluc = (t_obj_halluc / total_test_cases)

            # 如果有选择的测试案例，计算选择的平均评价指标
            if total_selected_test_cases > 0:
                selected_average_metrics = {
                    "onto": onto_id,
                    "type": "selected_test_cases",
                    "avg_precision": f"{sel_t_p / total_selected_test_cases:.5f}",
                    "avg_recall": f"{sel_t_r / total_selected_test_cases:.5f}",
                    "avg_f1": f"{sel_t_f1 / total_selected_test_cases:.5f}",
                    "avg_onto_conf": f"{sel_t_onto_conf / total_selected_test_cases:.5f}",
                    "avg_sub_halluc": f"{sel_t_sub_halluc / total_selected_test_cases:.5f}",
                    "avg_rel_halluc": f"{sel_t_rel_halluc / total_selected_test_cases:.5f}",
                    "avg_obj_halluc": f"{sel_t_obj_halluc / total_selected_test_cases:.5f}"
                }
                append_jsonl(selected_average_metrics, eval_inputs['avg_out_file'])

            # 计算全局的平均评价指标
        num_ontologies = len(eval_inputs['onto_list'])
        global_metrics = {
            "id": "global",
            "type": "global",
            "avg_precision": f"{global_p / num_ontologies:.5f}",
            "avg_recall": f"{global_r / num_ontologies:.5f}",
            "avg_f1": f"{global_f1 / num_ontologies:.5f}",
            "avg_onto_conf": f"{global_onto_conf / num_ontologies:.5f}",
            "avg_sub_halluc": f"{global_sub_halluc / num_ontologies:.5f}",
            "avg_rel_halluc": f"{global_rel_halluc / num_ontologies:.5f}",
            "avg_obj_halluc": f"{global_obj_halluc / num_ontologies:.5f}",
            "onto_list": eval_inputs['onto_list']
        }
        append_jsonl(global_metrics, eval_inputs['avg_out_file'])

if __name__ == "__main__":
        sys.exit(main())
