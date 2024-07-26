from FlagEmbedding import FlagModel
from transformers import AutoTokenizer, AutoModel
import json as js
import pickle as pk
from tqdm import tqdm
import torch
from cogdl.oag import oagbert
import torch.nn.functional as F


with open("../dataset/IND-WhoIsWho/norm_pid_to_info_all.json", "r", encoding="utf-8") as file:
    papers = js.load(file)

batch_size = 500
papers = [[key, value] for key, value in papers.items()]
count = len(papers) // batch_size + (1 if len(papers) % batch_size != 0 else 0)

# -----------------------为每篇论文生成bge-embedding---------------------------
model = FlagModel("./model/bge-small-en-v1.5", use_fp16=True)
bge_paper_embedding = {}

for i in tqdm(range(0, len(papers), batch_size), desc="process line", total=count):
    batch_papers = papers[i : i + batch_size]
    title_list = [paper[1]["title"] for paper in batch_papers]
    abstract_list = [paper[1]["abstract"] for paper in batch_papers]
    title_and_abstract_list = [f"title: {paper[1]['title']}\n abstract: {paper[1]['abstract']}" for paper in batch_papers]
    keywords_list = [" ".join(paper[1]["keywords"]) for paper in batch_papers]
    venue_list = [paper[1]["venue"] for paper in batch_papers]
    title_embeddings = model.encode(title_list, batch_size=batch_size)
    abstract_embeddings = model.encode(abstract_list, batch_size=batch_size)
    title_and_abstract_embeddings = model.encode(title_and_abstract_list, batch_size=batch_size)
    keywords_embeddings = model.encode(keywords_list, batch_size=batch_size)
    venue_embeddings = model.encode(venue_list, batch_size=batch_size)
    t = 0

    for j in range(i, i + len(batch_papers)):
        pid = papers[j][0]
        title_vec = title_embeddings[t]
        abstract_vec = abstract_embeddings[t]
        title_and_abstract_vec = title_and_abstract_embeddings[t]
        keywords_vec = keywords_embeddings[t]
        venue_vec = venue_embeddings[t]
        t += 1
        bge_paper_embedding[pid] = {
            "title_vec": title_vec.tolist(),
            "abstract_vec": abstract_vec.tolist(),
            "title_and_abstract_vec": title_and_abstract_vec.tolist(),
            "keywords_vec": keywords_vec.tolist(),
            "venue_vec": venue_vec.tolist(),
        }

with open("../dataset/embedding/bge_small_paper_embedding_all_info.pk", "wb") as f:
    pk.dump(bge_paper_embedding, f)

# -----------------------为每篇论文生成scibert-embedding---------------------------
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("./model/sci-bert")
model = AutoModel.from_pretrained("./model/sci-bert").to(device)

scibert_paper_embedding = {}
for i in tqdm(range(0, len(papers), batch_size), desc="process line", total=count):
    batch_papers = papers[i : i + batch_size]
    texts = [paper[1]["title"] for paper in batch_papers]
    abstract_list = [paper[1]["abstract"] for paper in batch_papers]
    title_and_abstract_list = [f"title: {paper[1]['title']}\n abstract: {paper[1]['abstract']}" for paper in batch_papers]
    keywords_list = [" ".join(paper[1]["keywords"]) for paper in batch_papers]
    venue_list = [paper[1]["venue"] for paper in batch_papers]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    title_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
    inputs = tokenizer(abstract_list, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    abstract_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
    inputs = tokenizer(title_and_abstract_list, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    title_and_abstract_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()

    inputs = tokenizer(keywords_list, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    keywords_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()

    inputs = tokenizer(venue_list, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    venue_embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
    t = 0

    for j in range(i, i + len(batch_papers)):
        pid = papers[j][0]
        title_vec = title_embeddings[t]
        abstract_vec = abstract_embeddings[t]
        title_and_abstract_vec = title_and_abstract_embeddings[t]
        keywords_vec = keywords_embeddings[t]
        venue_vec = venue_embeddings[t]
        t += 1
        scibert_paper_embedding[pid] = {
            "title_vec": title_vec.tolist(),
            "abstract_vec": abstract_vec.tolist(),
            "title_and_abstract_vec": title_and_abstract_vec.tolist(),
            "keywords_vec": keywords_vec.tolist(),
            "venue_vec": venue_vec.tolist(),
        }
with open("../dataset/embedding/scibert_paper_embedding_all_info.pk", "wb") as file:
    pk.dump(scibert_paper_embedding, file)

# -----------------------为每篇论文生成oagbert-embedding---------------------------
tokenizer, model = oagbert("oagbert-v2-sim")
model.eval()
device = torch.device("cuda:0")
oagbert_paper_embedding = {}

for paper in tqdm(papers, desc="process line", total=len(papers)):
    pid = paper[1]["id"]
    title = paper[1]["title"]
    abstract = paper[1]["abstract"]
    keywords = paper[1]["keywords"]
    venue = paper[1]["venue"]
    auther_list = [x["name"] for x in paper[1]["authors"] if x["name"] != ""]
    affiliations_list = [x["org"] for x in paper[1]["authors"] if x["org"] != ""]
    input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
        title=title, abstract=abstract, venue=venue, authors=auther_list, concepts=keywords, affiliations=affiliations_list
    )
    try:
        _, paper_embed = model.bert.forward(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0),
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=torch.LongTensor(position_ids).unsqueeze(0),
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0),
        )
        paper_embed = F.normalize(paper_embed, p=2, dim=1)

    except Exception as e:
        paper_embed = torch.zeros((1, 768))
    oagbert_paper_embedding[pid] = {
        "paper_oagbert_vec": paper_embed.tolist(),
    }

with open("../dataset/embedding/oagbert_paper_embedding_all_info.pk", "wb") as file:
    pk.dump(oagbert_paper_embedding, file)
