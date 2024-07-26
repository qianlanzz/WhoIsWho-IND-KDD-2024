import numpy as np
import json
from unidecode import unidecode
from tqdm import tqdm
import pickle as pk

with open("../dataset/IND-WhoIsWho/train_author.json", "r", encoding="utf-8") as file:
    train_author = json.load(file)
with open("../dataset/IND-WhoIsWho/norm_pid_to_info_all.json", "r", encoding="utf-8") as file:
    pid_to_info = json.load(file)
with open("../dataset/IND-test-public/ind_test_author_filter_public.json", "r", encoding="utf-8") as file:
    test_author = json.load(file)
with open("../dataset/embedding/bge_small_paper_embedding_all_info.pk", "rb") as file:
    bge_paper_embedding = pk.load(file)
with open("../dataset/embedding/scibert_paper_embedding_all_info.pk", "rb") as file:
    scibert_paper_embedding = pk.load(file)
with open("../dataset/embedding/oagbert_paper_embedding_all_info.pk", "rb") as file:
    oagbert_paper_embedding = pk.load(file)


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


def co_occurance(core_name, paper1, paper2):
    core_name = clean_name(core_name)
    coauthor_weight = 0
    coorg_weight = 0
    covenue_weight = 0
    ori_n1_authors = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper1["authors"]), 50))]
    ori_n2_authors = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper2["authors"]), 50))]

    # remove disambiguate author
    for name in ori_n1_authors:
        if simple_name_match(core_name, name):
            ori_n1_authors.remove(name)

    for name in ori_n2_authors:
        if simple_name_match(core_name, name):
            ori_n2_authors.remove(name)

    whole_authors = max(len(set(ori_n1_authors + ori_n2_authors)), 1)

    matched = []
    for per_n1 in ori_n1_authors:
        for per_n2 in ori_n2_authors:
            if simple_name_match(per_n1, per_n2):
                matched.append((per_n1, per_n2))
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
    return cotitle_weight, coauthor_weight, coorg_weight, coabstract_weight, cokeywords_weight, covenue_weight


def co_cal(core_name, paper1, paper_list):
    core_name = clean_name(core_name)
    auther_list_1 = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in range(len(paper1["authors"]))]
    org_1 = " ".join([i["org"] for i in paper1["authors"] if i["org"] != ""]).split()
    n1_venue = paper1["venue"].split()
    n1_keyword = set((" ".join(paper1["keywords"]).split()))

    if core_name in auther_list_1:
        auther_list_1.remove(core_name)

    num = 0
    num_org = 0
    num_venue = 0
    num_keyword = 0

    for paper2 in paper_list:
        if paper1["id"] != paper2["id"]:
            auther_list_2 = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in range(len(paper2["authors"]))]
            org_2 = " ".join([i["org"] for i in paper2["authors"] if i["org"] != ""]).split()
            n2_venue = paper2["venue"].split()
            n2_keyword = set((" ".join(paper2["keywords"]).split()))

            if core_name in auther_list_2:
                auther_list_2.remove(core_name)
            if len(set(auther_list_1) & set(auther_list_2)) > 0:
                num += 1
            if len(set(org_1) & set(org_2)) > 0:
                num_org += 1
            if len(set(n1_venue) & set(n2_venue)) > 0:
                num_venue += 1
            if len(set(n1_keyword) & set(n2_keyword)) > 0:
                num_keyword += 1

    return (num + 1) / len(paper_list), (num_org + 1) / len(paper_list), (num_venue + 1) / len(paper_list), (num_keyword + 1) / len(paper_list)


def get_similarity_batch(paper1_array, others_pappers_array):
    similarity = paper1_array @ others_pappers_array.T
    similarity_sum = np.sum(similarity)
    return similarity_sum


def get_total_similarity(paper_embedding, paper1_id, other_papers):
    total_title_sim = get_similarity_batch(np.array(paper_embedding[paper1_id]["title_vec"]), np.array(other_papers[0]))
    total_abstract_sim = get_similarity_batch(np.array(paper_embedding[paper1_id]["abstract_vec"]), np.array(other_papers[1]))
    total_title_and_abstract_sim = get_similarity_batch(np.array(paper_embedding[paper1_id]["title_and_abstract_vec"]), np.array(other_papers[2]))
    total_keyword_sim = get_similarity_batch(np.array(paper_embedding[paper1_id]["keywords_vec"]), np.array(other_papers[3]))
    total_venue_sim = get_similarity_batch(np.array(paper_embedding[paper1_id]["venue_vec"]), np.array(other_papers[4]))
    return total_title_sim, total_abstract_sim, total_title_and_abstract_sim, total_keyword_sim, total_venue_sim


def get_other_paper_inf(paper_embedding, paper1_id, all_pappers_id):
    other_papers = []
    other_papers_title = []
    other_papers_abstract = []
    other_papers_title_and_abstract = []
    other_papers_keywords = []
    other_papers_venue = []
    for jj in range(len(all_pappers_id)):
        paper2_id = all_pappers_id[jj]
        if paper1_id == paper2_id:
            continue
        other_papers_title.append(paper_embedding[paper2_id]["title_vec"])
        other_papers_abstract.append(paper_embedding[paper2_id]["abstract_vec"])
        other_papers_title_and_abstract.append(paper_embedding[paper2_id]["title_and_abstract_vec"])
        other_papers_keywords.append(paper_embedding[paper2_id]["keywords_vec"])
        other_papers_venue.append(paper_embedding[paper2_id]["venue_vec"])
    other_papers.append(other_papers_title)
    other_papers.append(other_papers_abstract)
    other_papers.append(other_papers_title_and_abstract)
    other_papers.append(other_papers_keywords)
    other_papers.append(other_papers_venue)
    return other_papers


def get_oagbert_similarity(oagbert_paper_embedding, paper1_id, other_papers):
    total_oagbert_sim = get_similarity_batch(np.array(oagbert_paper_embedding[paper1_id]["paper_oagbert_vec"][0]), np.array(other_papers))
    return total_oagbert_sim


def get_len_feature(pid):
    return [
        len(pid_to_info[pid]["title"]),
        len(pid_to_info[pid]["abstract"]),
        len(pid_to_info[pid]["keywords"]),
        len(pid_to_info[pid]["authors"]),
        len(pid_to_info[pid]["venue"]),
        int(pid_to_info[pid]["year"]) if pid_to_info[pid]["year"] != None and pid_to_info[pid]["year"] != "" else 2000,
    ]


def add_feature(orcid, paper1_id, author_names):
    if "normal_data" in author_names[orcid]:
        normal_papers_id = author_names[orcid]["normal_data"]
        outliers_id = author_names[orcid]["outliers"]
        all_pappers_id = normal_papers_id + outliers_id
    elif "papers" in author_names[orcid]:
        all_pappers_id = author_names[orcid]["papers"]

    other_oagbert_papers = []
    paper1_inf = pid_to_info[paper1_id]
    papers_list = []
    total_w_cotitle, total_w_coauthor, total_w_coorg, total_w_coabstract, total_w_cokeywords, total_w_covenue = 0, 0, 0, 0, 0, 0
    for jj in range(len(all_pappers_id)):
        paper2_id = all_pappers_id[jj]
        if paper1_id == paper2_id:
            continue
        paper2_inf = pid_to_info[paper2_id]
        papers_list.append(paper2_inf)
        other_oagbert_papers.append(oagbert_paper_embedding[paper2_id]["paper_oagbert_vec"][0])
        # feature kind one: total_w_cotitle, total_w_coauthor, total_w_coabstract, total_w_coorg, total_w_cokeywords, total_w_covenue
        w_cotitle, w_coauthor, w_coorg, w_coabstract, w_cokeywords, w_covenue = co_occurance(author_names[orcid]["name"], paper1_inf, paper2_inf)
        total_w_cotitle += w_cotitle
        total_w_coauthor += w_coauthor
        total_w_coorg += w_coorg
        total_w_coabstract += w_coabstract
        total_w_cokeywords += w_cokeywords
        total_w_covenue += w_covenue

    other_bge_papers = get_other_paper_inf(bge_paper_embedding, paper1_id, all_pappers_id)
    other_scibert_papers = get_other_paper_inf(scibert_paper_embedding, paper1_id, all_pappers_id)

    # feature kind two:  total_title_bge_sim, total_abstract_bge_sim, total_title_and_abstract_bge_sim, total_keyword_bge_smi, total_venue_bge_smi
    total_title_bge_sim, total_abstract_bge_sim, total_title_and_abstract_bge_sim, total_keyword_bge_sim, total_venue_bge_sim = get_total_similarity(bge_paper_embedding, paper1_id, other_bge_papers)
    # feature kind three:  total_title_scibert_sim, total_abstract_scibert_sim, total_title_and_abstract_scibert_sim, total_keyword_scibert_smi, total_venue_scibert_smi
    total_title_scibert_sim, total_abstract_scibert_sim, total_title_and_abstract_scibert_sim, total_keyword_scibert_sim, total_venue_scibert_sim = get_total_similarity(
        scibert_paper_embedding, paper1_id, other_scibert_papers
    )
    # feature kind four: total_oagbert_sim
    total_oagbert_sim = get_oagbert_similarity(oagbert_paper_embedding, paper1_id, other_oagbert_papers)
    # feature kind five: co_auther, co_auther_org, co_venu, co_keywords
    co_auther, co_auther_org, co_venu, co_keywords = co_cal(author_names[orcid]["name"], paper1_inf, papers_list)

    return [
        total_w_cotitle,
        total_w_coauthor,
        total_w_coorg,
        total_w_coabstract,
        total_w_cokeywords,
        total_w_covenue,
        co_auther,
        co_auther_org,
        co_venu,
        co_keywords,
        total_title_bge_sim,
        total_abstract_bge_sim,
        total_title_and_abstract_bge_sim,
        total_keyword_bge_sim,
        total_venue_bge_sim,
        total_title_scibert_sim,
        total_abstract_scibert_sim,
        total_title_and_abstract_scibert_sim,
        total_keyword_scibert_sim,
        total_venue_scibert_sim,
        total_oagbert_sim,
    ]


# -------------------------------构造训练数据特征----------------------------------------------------

len_train_feature = {}
co_occurance_sim_train_feature = {}
bge_sim_train_feature = {}
scibert_sim_train_feature = {}
oagbert_sim_train_feature = {}

for aid, person_info in tqdm(train_author.items(), desc="process line", total=len(train_author)):
    tmp1 = {}
    tmp2 = {}
    tmp3 = {}
    tmp4 = {}
    tmp5 = {}
    for pid in person_info["normal_data"]:
        w_list = add_feature(aid, pid, train_author)
        tmp1.update({pid: w_list[:10]})
        tmp2.update({pid: w_list[10:15]})
        tmp3.update({pid: w_list[15:20]})
        tmp4.update({pid: w_list[20:]})
        tmp5.update({pid: get_len_feature(pid)})
    for pid in person_info["outliers"]:
        w_list = add_feature(aid, pid, train_author)
        tmp1.update({pid: w_list[:10]})
        tmp2.update({pid: w_list[10:15]})
        tmp3.update({pid: w_list[15:20]})
        tmp4.update({pid: w_list[20:]})
        tmp5.update({pid: get_len_feature(pid)})
    co_occurance_sim_train_feature.update({aid: tmp1})
    bge_sim_train_feature.update({aid: tmp2})
    scibert_sim_train_feature.update({aid: tmp3})
    oagbert_sim_train_feature.update({aid: tmp4})
    len_train_feature.update({aid: tmp5})
with open("../dataset/feature/len_train_feature.pk", "wb") as file:
    pk.dump(len_train_feature, file)
with open("../dataset/feature/co_occurance_sim_train_feature.pk", "wb") as file:
    pk.dump(co_occurance_sim_train_feature, file)
with open("../dataset/feature/bge_sim_train_feature.pk", "wb") as file:
    pk.dump(bge_sim_train_feature, file)
with open("../dataset/feature/scibert_sim_train_feature.pk", "wb") as file:
    pk.dump(scibert_sim_train_feature, file)
with open("../dataset/feature/oagbert_sim_train_feature.pk", "wb") as file:
    pk.dump(oagbert_sim_train_feature, file)

# -------------------------------构造测试数据特征----------------------------------------------------
len_test_feature = {}
co_occurance_sim_test_feature = {}
bge_sim_test_feature = {}
scibert_sim_test_feature = {}
oagbert_sim_test_feature = {}

for aid, person_info in tqdm(test_author.items(), desc="process line", total=len(test_author)):
    tmp1 = {}
    tmp2 = {}
    tmp3 = {}
    tmp4 = {}
    tmp5 = {}
    for pid in person_info["papers"]:
        w_list = add_feature(aid, pid, test_author)
        tmp1.update({pid: w_list[:10]})
        tmp2.update({pid: w_list[10:15]})
        tmp3.update({pid: w_list[15:20]})
        tmp4.update({pid: w_list[20:]})
        tmp5.update({pid: get_len_feature(pid)})
    co_occurance_sim_test_feature.update({aid: tmp1})
    bge_sim_test_feature.update({aid: tmp2})
    scibert_sim_test_feature.update({aid: tmp3})
    oagbert_sim_test_feature.update({aid: tmp4})
    len_test_feature.update({aid: tmp5})
with open("../dataset/feature/len_test_feature.pk", "wb") as file:
    pk.dump(len_test_feature, file)
with open("../dataset/feature/co_occurance_sim_test_feature.pk", "wb") as file:
    pk.dump(co_occurance_sim_test_feature, file)
with open("../dataset/feature/bge_sim_test_feature.pk", "wb") as file:
    pk.dump(bge_sim_test_feature, file)
with open("../dataset/feature/scibert_sim_test_feature.pk", "wb") as file:
    pk.dump(scibert_sim_test_feature, file)
with open("../dataset/feature/oagbert_sim_test_feature.pk", "wb") as file:
    pk.dump(oagbert_sim_test_feature, file)
