import pandas as pd
import numpy as np
import json 
from tqdm import tqdm
import pickle as pk
import random

class Config():
    seed = 2024
    num_folds = 10
    TARGET_NAME = 'label'

def create_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
create_seed(Config.seed)

with open("../dataset/IND-WhoIsWho/train_author.json", "r", encoding="utf-8") as file:
    train_author = json.load(file)
with open("../dataset/IND-WhoIsWho/norm_pid_to_info_all.json", "r", encoding="utf-8") as file:
    pid_to_info = json.load(file)
with open("../dataset/IND-test-public/ind_test_author_filter_public.json", "r", encoding="utf-8") as file:
    test_author = json.load(file)
with open("../dataset/IND-test-public/ind_test_author_submit.json", "r", encoding="utf-8") as file:
    submission = json.load(file)
with open("../dataset/embedding/bge_small_paper_embedding_all_info.pk", "rb") as file:
    bge_paper_embedding = pk.load(file)
with open("../dataset/embedding/oagbert_paper_embedding_all_info.pk", "rb") as file:
    oagbert_paper_embedding = pk.load(file)


#---------------------------------------test data-------------------------------------------------------------------
with open("../dataset/feature/len_test_feature.pk", "rb") as file:
    len_test_feature = pk.load(file)
with open("../dataset/feature/co_occurance_sim_test_feature.pk", "rb") as file:
    co_occurance_sim_test_feature = pk.load( file)
with open("../dataset/feature/bge_sim_test_feature.pk", "rb") as file:
    bge_sim_test_feature = pk.load(file)
with open("../dataset/feature/oagbert_sim_test_feature.pk", "rb") as file:
    oagbert_sim_test_feature = pk.load(file)
with open("../dataset/feature/scibert_sim_test_feature.pk", "rb") as file:
    scibert_sim_test_feature = pk.load(file)
test_feats=[]
index = 0

for aid,person_info in tqdm(test_author.items(), desc="process line", total=len(test_author)):
    for pid in person_info['papers']:
        test_feats.append(
        len_test_feature[aid][pid] + 
        co_occurance_sim_test_feature[aid][pid] + 
        bge_sim_test_feature[aid][pid] + 
        oagbert_sim_test_feature[aid][pid] +
        scibert_sim_test_feature[aid][pid] + 
        #add bge embedding vector for feature
        bge_paper_embedding[pid]['title_vec'] + bge_paper_embedding[pid]['abstract_vec'] + bge_paper_embedding[pid]['title_and_abstract_vec'] + bge_paper_embedding[pid]['keywords_vec'] + bge_paper_embedding[pid]['venue_vec'] +
        #add oagbert embedding vector for feature
        oagbert_paper_embedding[pid]['paper_oagbert_vec'][0]
        )

test_feats=np.array(test_feats)
test_feats=pd.DataFrame(test_feats)
print(test_feats.head())

choose_cols=[col for col in test_feats.columns]

#------------------------------------------predict--------------------------------------------------
test_pred_pro = np.zeros((Config.num_folds, len(test_feats), 2))
for fold in range(0, Config.num_folds):
    print("-"*10 + f"fold{fold}" + "-"*10)
    with open(f"../model/xgb_model/{fold}_xgb_model.pk", "rb") as file:
        model = pk.load(file)
    test_pred_pro[fold] = model.predict_proba(test_feats)
test_preds = test_pred_pro.mean(axis=0)[:,1]

cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = test_preds[cnt]
        cnt += 1
with open('../dataset/result/xgboost_result.json', 'w', encoding='utf-8') as file:
    json.dump(submission, file, ensure_ascii=False, indent=4)
