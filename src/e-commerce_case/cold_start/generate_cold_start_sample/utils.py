import json
import pandas as pd


# 读取相关数据
def parse(path):
    g = open(path, 'r')
    for row in g:
        yield json.loads(row)


# 将数据存为 DataFrame 格式，方便后续处理
def get_df(path):
    i = 0
    df_ = {}
    for d in parse(path):
        df_[i] = d
        i += 1
    return pd.DataFrame.from_dict(df_, orient='index')