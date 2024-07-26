import os
import torch
from tqdm import tqdm
import json
import pickle

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_dir = "../model/gcn_model/model_7.pt"
test_data = "../dataset/graph/graph_submit_test.pkl"

encoder = torch.load(model_dir)
encoder.eval()
with open(test_data, "rb") as f:
    test_data = pickle.load(f)
result = {}

with torch.no_grad():
    for tmp_test in tqdm(test_data):
        each_sub, _, author_id, pub_id = tmp_test
        each_sub = each_sub.cuda()
        node_outputs, adj_matrix, adj_weight, batch_item = (
            each_sub.x,
            each_sub.edge_index,
            each_sub.edge_attr.squeeze(-1),
            each_sub.batch,
        )

        adj_weight = adj_weight.mean(dim=-1)
        # 获取大于等于阈值的边的索引
        mask = adj_weight >= 0.01

        # 过滤掉权重小于阈值的边
        filtered_adj_matrix = adj_matrix[:, mask]
        filtered_adj_weight = adj_weight[mask]

        logit = encoder(node_outputs, filtered_adj_matrix, filtered_adj_weight)
        logit = logit.squeeze(-1)

        result[author_id] = {}
        for i in range(len(pub_id)):
            result[author_id][pub_id[i]] = logit[i].item()

with open("../dataset/result/gcn_result.json", "w") as f:
    json.dump(result, f, indent=4)
