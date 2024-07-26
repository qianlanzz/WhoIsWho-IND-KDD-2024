import json
import numpy as np
import pickle as pk
from unidecode import unidecode
import torch
from torch_geometric.data.batch import Batch
import multiprocessing as mp
from tqdm import tqdm


def clean_name(name):
    name = unidecode(name)
    name = name.lower()
    new_name = ""
    for a in name:
        if a.isalpha():
            new_name += a
        else:
            new_name = new_name.strip()
            new_name += " "
    return new_name.strip()


def simple_name_match(n1, n2):
    n1_set = set(n1.split())
    n2_set = set(n2.split())

    if len(n1_set) != len(n2_set):
        return False
    com_set = n1_set & n2_set
    if len(com_set) == len(n1_set):
        return True
    return False


# 计算两篇文章的oag_bert文本相似度
def oag_bert_similarity(paper1_id, paper2_id):
    def calculate_similarity(vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    paper1_paper_vec = np.array(torch.tensor(oag_bert_paper_embedding[paper1_id]["paper_oagbert_vec"]).squeeze(0))
    paper2_paper_vec = np.array(torch.tensor(oag_bert_paper_embedding[paper2_id]["paper_oagbert_vec"]).squeeze(0))
    paper_similarity = calculate_similarity(paper1_paper_vec, paper2_paper_vec)

    return paper_similarity


def co_occurance(core_name, paper1_id, paper2_id):
    paper1 = papers_info[paper1_id]
    paper2 = papers_info[paper2_id]
    core_name = clean_name(core_name)
    coauthor_weight = 0
    coorg_weight = 0
    covenue_weight = 0
    ori_n1_authors = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper1["authors"]), 50))]
    ori_n2_authors = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper2["authors"]), 50))]

    for name in ori_n1_authors:
        if simple_name_match(core_name, name):
            ori_n1_authors.remove(name)

    for name in ori_n2_authors:
        if simple_name_match(core_name, name):
            ori_n2_authors.remove(name)

    whole_authors = max(len(set(ori_n1_authors + ori_n2_authors)), 1)

    for per_n1 in ori_n1_authors:
        for per_n2 in ori_n2_authors:
            if simple_name_match(per_n1, per_n2):
                coauthor_weight += 1
                break
    coauthor_weight = coauthor_weight / whole_authors

    def jaccard_similarity(list1, list2):
        if not list1 or not list2:
            return 0
        intersection = len(set(list1) & set(list2))
        union = len(set(list1)) + len(set(list2)) - intersection
        return intersection / union if union != 0 else 0

    n1_title = paper1["title"].split()
    n2_title = paper2["title"].split()

    n1_org = " ".join([i["org"] for i in paper1["authors"] if i["org"] != ""]).split()
    n2_org = " ".join([i["org"] for i in paper2["authors"] if i["org"] != ""]).split()

    n1_abstract = paper1["abstract"].split()
    n2_abstract = paper2["abstract"].split()

    n1_keywords = " ".join(paper1["keywords"]).split()
    n2_keywords = " ".join(paper2["keywords"]).split()

    n1_venue = paper1["venue"].split()
    n2_venue = paper2["venue"].split()

    cotitle_weight = jaccard_similarity(n1_title, n2_title)
    coorg_weight = jaccard_similarity(n1_org, n2_org)
    coabstract_weight = jaccard_similarity(n1_abstract, n2_abstract)
    cokeywords_weight = jaccard_similarity(n1_keywords, n2_keywords)
    covenue_weight = jaccard_similarity(n1_venue, n2_venue)
    oag_bert_sim = oag_bert_similarity(paper1_id, paper2_id)
    return coauthor_weight, cotitle_weight, coorg_weight, coabstract_weight, cokeywords_weight, covenue_weight, oag_bert_sim


def getdata(orcid):
    trainset = True
    if "normal_data" in author_names[orcid]:
        normal_papers_id = author_names[orcid]["normal_data"]
        outliers_id = author_names[orcid]["outliers"]
        all_pappers_id = normal_papers_id + outliers_id
    elif "papers" in author_names[orcid]:
        all_pappers_id = author_names[orcid]["papers"]
        trainset = False
    total_matrix, total_weight = [], []

    for ii in range(len(all_pappers_id)):
        paper1_id = all_pappers_id[ii]
        for jj in range(len(all_pappers_id)):
            paper2_id = all_pappers_id[jj]
            if paper1_id == paper2_id:
                continue

            weight = list(co_occurance(author_names[orcid]["name"], paper1_id, paper2_id))
            if sum(weight) == 0:
                continue
            total_matrix.append([paper1_id, paper2_id])  # 两篇论文有关系 A矩阵
            total_weight.append(weight)  # 存权重 H矩阵

    num_papers = len(all_pappers_id)

    # 从新编号
    re_num = dict(zip(all_pappers_id, list(range(num_papers))))
    # edge_index
    if trainset:
        set_out = set(outliers_id)
        list_edge_y = [0 if (i in set_out) or (j in set_out) else 1 for i, j in total_matrix]

    else:
        list_edge_y = [1] * len(total_matrix)

    total_matrix = [[re_num[i], re_num[j]] for i, j in total_matrix]
    edge_index = np.array(total_matrix, dtype=np.int64).T

    # 提取点特征(embedding+相似度)
    list_x = []

    if trainset:
        for x in all_pappers_id:
            paper_emb_oag = oag_bert_paper_embedding[x]["paper_oagbert_vec"][0]
            co_occurance_sim = co_occurance_sim_train_feature[orcid][x]
            bge_sim = bge_sim_train_feature[orcid][x]
            oagbert_sim = oagbert_sim_train_feature[orcid][x]
            paper_feature_all = paper_emb_oag + co_occurance_sim + bge_sim + oagbert_sim
            list_x.append(paper_feature_all)
    else:
        for x in all_pappers_id:
            paper_emb_oag = oag_bert_paper_embedding[x]["paper_oagbert_vec"][0]
            co_occurance_sim = co_occurance_sim_test_feature[orcid][x]
            bge_sim = bge_sim_test_feature[orcid][x]
            oagbert_sim = oagbert_sim_test_feature[orcid][x]
            paper_feature_all = paper_emb_oag + co_occurance_sim + bge_sim + oagbert_sim
            list_x.append(paper_feature_all)

    features = np.stack(list_x)

    # node labels
    if trainset:
        list_y = len(normal_papers_id) * [1] + len(outliers_id) * [0]
    else:
        list_y = None

    # build batch
    batch = [0] * num_papers

    if edge_index.size == 0:  # if no edge, for rare cases, add default self loop with low weight
        e = [[], []]
        for i in range(len(all_pappers_id)):
            for j in range(len(all_pappers_id)):
                if i != j:
                    e[0].append(i)
                    e[1].append(j)
        edge_index = e
        total_weight = [[0.0001, 0.0001, 0.0001]] * len(e[0])
        if trainset:
            list_edge_y = []
            for i in range(len(edge_index[0])):
                if list_y[edge_index[0][i]] == 1 and list_y[edge_index[1][i]] == 1:
                    list_edge_y.append(1)
                else:
                    list_edge_y.append(0)
    # build data
    data = Batch(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(total_weight, dtype=torch.float32),
        y=torch.tensor(list_y) if list_y is not None else None,
        batch=torch.tensor(batch),
    )
    assert torch.any(torch.isnan(data.x)) == False
    edge_label = torch.tensor(list_edge_y) if trainset else None

    return (data, edge_label, orcid, all_pappers_id)


def build_dataset(path):
    keys_list = list(author_names.keys())  # 所有作者id

    print(f"total_len:{len(keys_list)}")
    with mp.Pool(processes=10) as pool:
        results = list(tqdm(pool.imap_unordered(getdata, keys_list), total=len(keys_list)))

    with open(path, "wb") as f:
        pk.dump(results, f)


if __name__ == "__main__":
    with open("../dataset/IND-WhoIsWho/norm_pid_to_info_all.json", "r", encoding="utf-8") as f:
        papers_info = json.load(f)

    # oag_bert特征向量
    with open("../dataset/embedding/oagbert_paper_embedding_all_info.pk", "rb") as file:
        oag_bert_paper_embedding = pk.load(file)

    with open("../dataset/feature/co_occurance_sim_train_feature.pk", "rb") as file:
        co_occurance_sim_train_feature = pk.load(file)
    with open("../dataset/feature/bge_sim_train_feature.pk", "rb") as file:
        bge_sim_train_feature = pk.load(file)
    with open("../dataset/feature/oagbert_sim_train_feature.pk", "rb") as file:
        oagbert_sim_train_feature = pk.load(file)
    with open("../dataset/feature/co_occurance_sim_test_feature.pk", "rb") as file:
        co_occurance_sim_test_feature = pk.load(file)
    with open("../dataset/feature/bge_sim_test_feature.pk", "rb") as file:
        bge_sim_test_feature = pk.load(file)
    with open("../dataset/feature/oagbert_sim_test_feature.pk", "rb") as file:
        oagbert_sim_test_feature = pk.load(file)

    # # 构建训练数据gcn图
    # with open("../dataset/IND-WhoIsWho/train_author.json", "r", encoding="utf-8") as f:
    #     author_names = json.load(f)
    # build_dataset("../dataset/graph/graph_submit_train.pkl")

    # 构建测试数据gcn图
    with open("../dataset/IND-test-public/ind_test_author_filter_public.json", "r", encoding="utf-8") as f:
        author_names = json.load(f)
    build_dataset("../dataset/graph/graph_submit_test.pkl")
