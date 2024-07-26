import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gcn_rsult', default="dataset/result/gcn_result.json")
parser.add_argument('--ml_result',default='dataset/result/xgboost_result.json')
parser.add_argument('--merge_result',default='merge_all.json')
args = parser.parse_args()

# 读取第一个 JSON 文件
with open(args.gcn_rsult, 'r') as f:
    data1 = json.load(f)

# 读取第二个 JSON 文件
with open(args.ml_result, 'r') as f:
    data2 = json.load(f)



# 初始化一个空字典用于存储加和后的值
summed_data = {}

# 遍历第一个 JSON 文件的键值对
for key, value in data1.items():
    if key in data2:  # 检查第二个 JSON 文件是否有相同的键
        # 将对应的值相加，并存储到新字典中
        author = {}
        auther1 =  value
        auther2 = data2[key]
        for key_2,value_2 in auther1.items():
            author[key_2] = value_2*0.1 + auther2[key_2]*0.9
        summed_data[key] = author

# 将结果保存到新的 JSON 文件中
with open(args.merge_result, 'w') as f:
    json.dump(summed_data, f)