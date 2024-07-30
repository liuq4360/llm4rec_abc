"""
利用sentence_transformers框架来实现一个最简单的个性化推荐：
1. 用户嵌入：用户浏览过的新闻的嵌入的平均值
2. 预测：利用用户嵌入与新闻嵌入的cosine余弦
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

col_spliter = "\t"
DIMS = 384  # all-MiniLM-L6-v2 模型的维数
TOP_N = 10  # 为每个用户生成10个新闻推荐

df_news = pd.read_csv("./data/mind/MINDsmall_train/news.tsv", sep=col_spliter)
df_news.columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url',
                   'title_entity', 'abstract_entity']

df_behavior = pd.read_csv("./data/mind/MINDsmall_train/behaviors.tsv", sep=col_spliter)
df_behavior.columns = ['impression_id', 'user_id', 'time', 'click_history', 'news']

model = SentenceTransformer('all-MiniLM-L6-v2')

# 获取每个新闻及对应的嵌入向量
news_embeddings = {}
for _, row in df_news.iterrows():
    news_id = row['news_id']
    title = row['title']
    embedding = model.encode(title)
    news_embeddings[news_id] = embedding


def rec_4_one_user(click_history):
    """
    为单个用户生成 TOP_N 个推荐
    """
    emb = np.zeros(DIMS, dtype=float)
    for news in click_history:
        emb = np.add(emb, news_embeddings[news])
    emb = emb / len(click_history)
    emb = emb.astype(np.float32)
    res = []
    for news_id, emb_ in news_embeddings.items():
        cos_sim = float(util.cos_sim(emb, emb_)[0][0])
        res.append((news_id, cos_sim))
    rec = sorted(res, key=lambda x: x[1], reverse=True)[:TOP_N]
    return rec


"""
为所有用户生成推荐
"""
user_rec = {}
for _, row in df_behavior.iterrows():
    user_id = row['user_id']
    click_history = row['click_history'].split(' ')
    rec = rec_4_one_user(click_history)
    user_rec[user_id] = rec
