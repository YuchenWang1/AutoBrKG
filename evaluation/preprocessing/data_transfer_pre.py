import json

# 从文件加载原始 JSON 数据
with open('../../knowledge/graph_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# 转换函数
def convert_data(data):
    result = []
    for idx, item in enumerate(data, start=1):
        triples = []

        # 处理三元组
        for triple in item['三元组']:
            sub, rel, obj = map(str.strip, triple.split('>'))
            triples.append([sub.split(':')[1].strip(), rel, obj.split(':')[1].strip()])

        # 处理属性
        for attr in item['属性']:
            sub, rel, obj = map(str.strip, attr.split('>'))
            triples.append([sub, "病害性状类别是", rel])
            triples.append([rel, "性状数值是", obj.strip()])

        result.append({
            "id": f"ont_bridge_test_{idx}",
            "response": item['文本'],
            "triples": triples
        })

    return result


# 转换数据
converted_data = convert_data(data)

# 导出到 JSONL 文件
with open('../data/bridge/bridge-pre.jsonl', 'w', encoding='utf-8') as f:
    for item in converted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


