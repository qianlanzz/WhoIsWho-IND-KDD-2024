import re
import json
from tqdm import tqdm

with open('../dataset/IND-WhoIsWho/pid_to_info_all.json', "r", encoding="utf-8") as file:
    info_all = json.load(file)

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']

def norm(data):
    """
    normalize venue, name and org, for build cleaned graph
    {
        id: str
        title: str
        authors:[{
            name
            org
        }]
        "abstract"
        "keywords"
        "venue"
        "year"
    }
    """
    title = ''
    if data['title']:
        title = data["title"].strip()
        title = title.lower()
        title = re.sub(puncs, ' ', title)
        title = re.sub(r'\s{2,}', ' ', title).strip()
        title = title.split(' ')
        title = [word for word in title if len(word) > 1]
        title = [word for word in title if word not in stopwords]
        title = [word for word in title if word not in stopwords_extend]
        title = [word for word in title if word not in stopwords_check]
        title = ' '.join(title)
    venue = ''
    if data['venue']:
        venue = data["venue"].strip()
        venue = venue.lower()
        venue = re.sub(puncs, ' ', venue)
        venue = re.sub(r'\s{2,}', ' ', venue).strip()
        venue = venue.split(' ')
        venue = [word for word in venue if len(word) > 1]
        venue = [word for word in venue if word not in stopwords]
        venue = [word for word in venue if word not in stopwords_extend]
        venue = [word for word in venue if word not in stopwords_check]
        venue = ' '.join(venue)
    authors = []
    if data['authors']:
        for i in data['authors']:
            org = i['org']
            if org:
                org = org.strip()
                org = org.lower()
                org = re.sub(puncs, ' ', org)
                org = re.sub(r'\s{2,}', ' ', org).strip()
                org = org.split(' ')
                org = [word for word in org if len(word) > 1]
                org = [word for word in org if word not in stopwords]
                org = [word for word in org if word not in stopwords_extend]
                org = " ".join(org)
            authors.append({
                "name": i['name'],
                "org": org
            })
    keywords = []
    if data['keywords']:
        for keyword in data['keywords']:
            keyword = keyword.strip()
            keyword = keyword.lower()
            keyword = re.sub(puncs, ' ', keyword)
            keyword = re.sub(r'\s{2,}', ' ', keyword).strip()
            keyword = keyword.split(' ')
            keyword = [word for word in keyword if len(word) > 1]
            keyword = [word for word in keyword if word not in stopwords]
            keyword = [word for word in keyword if word not in stopwords_extend]
            keyword = " ".join(keyword)
            keywords.append(keyword)
    abstract = ''
    if data['abstract']:
        abstract = data["abstract"].strip()
        abstract = abstract.lower()
        abstract = re.sub(puncs, ' ', abstract)
        abstract = re.sub(r'\s{2,}', ' ', abstract).strip()
        abstract = abstract.split(' ')
        abstract = [word for word in abstract if len(word) > 1]
        abstract = [word for word in abstract if word not in stopwords]
        abstract = [word for word in abstract if word not in stopwords_extend]
        abstract = [word for word in abstract if word not in stopwords_check]
        abstract = ' '.join(abstract)
    return {
        'id': data['id'],
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'keywords': keywords,
        'venue': venue,
        'year': data['year'],
    }

normalize_data = {}
for id, info in tqdm(info_all.items(), desc="process line", total=len(info_all)):
    normalize_data.update({id : norm(info)})  

with open("../dataset/IND-WhoIsWho/norm_pid_to_info_all.json", "w", encoding="utf-8") as file:
    json.dump(normalize_data, file)