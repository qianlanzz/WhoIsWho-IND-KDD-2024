import pandas as pd
import numpy as np
import json 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pickle as pk
import random
import time

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


#----------------------------------------train data---------------------------------------------------------
with open("../dataset/feature/len_train_feature.pk", "rb") as file:
    len_train_feature = pk.load(file)
with open("../dataset/feature/co_occurance_sim_train_feature.pk", "rb") as file:
    co_occurance_sim_train_feature = pk.load( file)
with open("../dataset/feature/bge_sim_train_feature.pk", "rb") as file:
    bge_sim_train_feature = pk.load(file)
with open("../dataset/feature/oagbert_sim_train_feature.pk", "rb") as file:
    oagbert_sim_train_feature = pk.load(file)
with open("../dataset/feature/scibert_sim_train_feature.pk", "rb") as file:
    scibert_sim_train_feature = pk.load(file)
train_feats=[]
labels=[]

for aid,person_info in tqdm(train_author.items(), desc="process line", total=len(train_author)):
    for pid in person_info['normal_data']:
        #all feature
        train_feats.append(
        len_train_feature[aid][pid] + 
        co_occurance_sim_train_feature[aid][pid] + 
        bge_sim_train_feature[aid][pid] + 
        oagbert_sim_train_feature[aid][pid] +
        scibert_sim_train_feature[aid][pid] + 
        #add bge embedding vector for feature
        bge_paper_embedding[pid]['title_vec'] + bge_paper_embedding[pid]['abstract_vec'] + bge_paper_embedding[pid]['title_and_abstract_vec'] + bge_paper_embedding[pid]['keywords_vec'] + bge_paper_embedding[pid]['venue_vec'] +
        #add oagbert embedding vector for feature
        oagbert_paper_embedding[pid]['paper_oagbert_vec'][0]
        )
        labels.append(1)
    for pid in person_info['outliers']:
        train_feats.append(
        len_train_feature[aid][pid] + 
        co_occurance_sim_train_feature[aid][pid] + 
        bge_sim_train_feature[aid][pid] + 
        oagbert_sim_train_feature[aid][pid] +
        scibert_sim_train_feature[aid][pid] + 
        # add bge embedding vector for feature
        bge_paper_embedding[pid]['title_vec'] + bge_paper_embedding[pid]['abstract_vec'] + bge_paper_embedding[pid]['title_and_abstract_vec'] + bge_paper_embedding[pid]['keywords_vec'] + bge_paper_embedding[pid]['venue_vec'] +
        # add oagbert embedding vector for feature
        oagbert_paper_embedding[pid]['paper_oagbert_vec'][0]
        )
        labels.append(0)   
train_feats=np.array(train_feats)
train_feats=pd.DataFrame(train_feats)
labels=np.array(labels)
train_feats['label']=labels
print(train_feats.head())

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
def fit_and_predict(model, train_feats = train_feats, test_feats = test_feats):
    X = train_feats[choose_cols].copy()
    y = train_feats[Config.TARGET_NAME].copy()
    test_X = test_feats[choose_cols].copy()
    oof_pred_pro = np.zeros((len(X), 2))
    test_pred_pro = np.zeros((Config.num_folds, len(test_X), 2))
    skf = StratifiedKFold(n_splits=Config.num_folds, random_state=Config.seed, shuffle=True)
    begin_time = time.time()
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y.astype(str))):
        print('-'*10 + f"fold:{fold}" + '-'*10)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100, 
            verbose=100,
        )
        oof_pred_pro[valid_index] = model.predict_proba(X_valid)
        test_pred_pro[fold] = model.predict_proba(test_X)
    print(f"final roc_auc: {roc_auc_score(y.values,oof_pred_pro[:,1])}")
    print(f"all time: {time.time() - begin_time}")
    return model, test_pred_pro

xgb_params={
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 20,
    "learning_rate": 0.05,
    "n_estimators":2800,
    "colsample_bytree": 0.9,
    "colsample_bynode": 0.9,
    "random_state": Config.seed,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "max_bin":255,
    "tree_method": "hist", 
    'device' : 'cuda:0'
}

model, xgb_test_pred_pro = fit_and_predict(model=XGBClassifier(**xgb_params))
test_preds = xgb_test_pred_pro.mean(axis=0)[:,1]

with open("../model/10fold_xgb_model.pk", "wb") as file:
    pk.dump(model, file)

cnt = 0
for id, names in submission.items():
    for name in names:
        submission[id][name] = test_preds[cnt]
        cnt += 1
with open('../dataset/result/xgboost_result.json', 'w', encoding='utf-8') as file:
    json.dump(submission, file, ensure_ascii=False, indent=4)
