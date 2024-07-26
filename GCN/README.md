## GCN

### Introduction：

Build graph relational data, train and predict results using gcn model

### Feature Description

- Point feature

  ```
  paper_emb_oag: 整篇论文oag-bert嵌入向量
  co_occurance_sim: 同名作者的一篇论文与其他论文共同title,abstract,keywords,author,org,venue权重之和，具体可参考Xgboost特征
  bge_sim: 同名作者的一篇论文与其他论文bge嵌入相似度之和
  oagbert_sim: 同名作者的一篇论文与其他论文oag-bert嵌入相似度之和
  ```

- Edge feature

  ```
  coauthor_weight: 两篇论文共同author权重
  cotitle_weight: 两篇论文共同title词权重
  coorg_weight: 两篇论文共同org词权重
  coabstract_weight: 两篇论文共同abstract词权重
  cokeywords_weight: 两篇论文共同keywords词权重
  covenue_weight: 两篇论文共同venue词权重
  oag_bert_sim: 根据两篇论文oag_bert计算出的相似度
  ```

### Model Improvement

- Activation Function

The Leaky ReLU activation function in MyGCNModel is more robust compared to ReLU. It handles negative feature values better, which might improve the model's performance.

- Edge Weights

MyGCNModel incorporates edge weights, making the model more flexible and accurate when dealing with weighted graphs. Edge weights can represent the strength or importance of edges, providing additional information to the model.

