import json
import pandas as pd

path_review = "../data/amazon_review/beauty/All_Beauty.json"
path_meta = "../data/amazon_review/beauty/meta_All_Beauty.json"


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


"""
获取用户评论数据，具体字段意思如下：
reviewerID：进行评论的用户的id
reviewText：用户对商品评论的内容
asin：商品唯一id
"""
df_view = get_df(path_review)
# 选择评论分数大于3的，代表用户是正向评论
df_view = df_view[df_view["overall"] > 3][["reviewerID", "reviewText", "asin"]]
"""
获取产品metadata数据，具体字段意思如下：
asin：商品唯一id
title：商品标题
description：商品描述信息
brand：商品品牌
"""
df_meta = get_df(path_meta)
df_meta = df_meta[["asin", "title", "description", "brand"]]
# 评论数据和商品数据join
df_joined = pd.merge(df_view, df_meta, how="left", on=['asin'])
# 剔除掉字段为空值的行，每一行只要有一个空值，整行都去掉
df_joined.dropna(how='any', subset=["asin", "title", "description", "brand"], inplace=True)
df_joined = df_joined.reset_index(drop=True)

"""
生成用户的品牌兴趣画像。我最终的目的是基于用户喜欢的产品信息来预测用户最喜欢的1-2个品牌。
"""
df_joined.drop(['reviewText', 'asin', 'description'], axis=1, inplace=True)
# [reviewerID, title, brand]
# 对每个用户的喜欢的行为进行聚合，目的是将用户喜欢的产品聚合起来，将用户最喜欢的top2品牌算出来。
TOP_TWO = 2
grouped = df_joined.groupby('reviewerID')
prompt_list = []
brand_list = []
for name, group in grouped:
    if group.shape[0] > 2 * TOP_TWO:  # 至少需要5个样本，方便大模型更好地学习用户喜欢的品牌。
        title = group["title"][0:3 * TOP_TWO]  # 'pandas.core.series.Series', 最多6个样本
        brand = group["brand"][0:3 * TOP_TWO]  # 'pandas.core.series.Series'
        purchase_list = []
        brand_dic = {}
        for e in zip(title, brand):
            purchase_list.append({
                "title": e[0],
                "brand": e[1]
            })
        for b in brand:
            if b != "":
                if b in brand_dic:
                    brand_dic[b] = brand_dic[b] + 1
                else:
                    brand_dic[b] = 1
        # 算出用户最喜欢的1-2个品牌，这就是用户的品牌兴趣画像
        filtered_dict = {}
        for key, value in brand_dic.items():
            if value > 1:
                filtered_dict[key] = value
        brand_top_2 = [x for (x, t) in sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True)[0:TOP_TWO]]
        brand_top_2 = ",".join(brand_top_2)
        prompt = ("I've purchased the following products in the past(in JSON format):\n\n" +
                  json.dumps(purchase_list, indent=4) + "\n\n" +
                  "Based on above purchased history, please predict what brands I like. " +
                  "You just need to list one or two most like brand names, do not explain the reason." +
                  "If I like two brands, please separate these two brand names with comma."
                  )
        prompt_list.append(prompt)
        brand_list.append(brand_top_2)
# 创建用户兴趣画像DataFrame，这里我忽略了reviewerID，在实际使用过程中，我只要获得用户
# 喜欢过的品牌名，然后利用大模型生成用户最喜欢的2个品牌画像
df = pd.DataFrame({'prompt': prompt_list, 'label': brand_list})
print(df.shape)
df.to_csv("./data/portrait_data.csv", index=False)
