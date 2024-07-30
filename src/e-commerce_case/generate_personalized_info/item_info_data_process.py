import json
import pandas as pd

path_review = "../data/amazon_review/toys/Toys_and_Games.json"
path_meta = "../data/amazon_review/toys/meta_Toys_and_Games.json"


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
df_review = get_df(path_review)
# df_review.drop_duplicates().groupby('style').count()
# 选择评论分数大于3的，代表用户是正向评论
df_review = df_review[df_review["overall"] > 3][["reviewerID", "asin"]]
"""
获取产品metadata数据，具体字段意思如下：
asin：商品唯一id
brand：商品品牌
category：商品分类
"""
df_meta = get_df(path_meta)
df_feature = df_meta[["asin", "brand", 'category']]
# 评论数据和商品数据join
df_joined = pd.merge(df_review, df_feature, how="left", on=['asin'])
# 剔除掉字段为空值的行，每一行只要有一个空值，整行都去掉
df_joined.dropna(how='any', subset=["asin", "brand", "category"], inplace=True)
# df_joined.drop_duplicates()
df_joined = df_joined.reset_index(drop=True)

"""
计算每个用户最喜欢的top3的品牌和分类
"""
# 对每个用户的喜欢的行为进行聚合，目的是将用户喜欢的产品聚合起来，将用户最喜欢的top2品牌算出来。
grouped = df_joined.groupby('reviewerID')
# grouped.get_group('A0001528BGUBOEVR6T5U')
#                    reviewerID        asin             brand                                           category
# 895406   A0001528BGUBOEVR6T5U  B0016H5BD2  Rock Ridge Magic  [Toys & Games, Novelty & Gag Toys, Magic Kits ...
# 1042542  A0001528BGUBOEVR6T5U  B0016H5BD2  Rock Ridge Magic  [Toys & Games, Novelty & Gag Toys, Magic Kits ...
# 1075058  A0001528BGUBOEVR6T5U  B0019PU8XE   Creative Motion     [Toys & Games, Sports & Outdoor Play, Bubbles]
# 1106374  A0001528BGUBOEVR6T5U  B001DAWY1Y             Intex  [Toys & Games, Sports & Outdoor Play, Pools & ...
# 1776903  A0001528BGUBOEVR6T5U  B005HTH78W            Aketek                                                 []
TOP_FOUR = 4  # 用户至少要评论过4个商品，这样可以更好地挖掘出用户的兴趣
TOP_THREE = 3  # 用户最喜欢的品牌或者类目数量
reviewerID_list = []
category_list = []
brand_list = []
for name, group in grouped:
    if group.shape[0] >= TOP_FOUR:
        reviewerID = name
        category = group["category"]  # 'pandas.core.series.Series'
        brand = group["brand"]  # 'pandas.core.series.Series'
        category_dic = {}
        brand_dic = {}
        for b in brand:
            if b in brand_dic:
                brand_dic[b] = brand_dic[b] + 1
            else:
                brand_dic[b] = 1
        for c in category:
            for c_index in c:
                if c_index in category_dic:
                    category_dic[c_index] = category_dic[c_index] + 1
                else:
                    category_dic[c_index] = 1
        # 算出用户最喜欢的3个品牌、类目，这就是用户的兴趣画像
        reviewerID_list.append(reviewerID)
        brand_top_3 = [x for (x, _) in sorted(brand_dic.items(), key=lambda x: x[1], reverse=True)[0:TOP_THREE]]
        brand_interest = ""
        for i in brand_top_3:
            brand_interest = brand_interest + "," + i
        brand_interest = brand_interest[1:]
        brand_list.append(brand_interest)
        category_top_3 = [x for (x, _) in sorted(category_dic.items(), key=lambda x: x[1], reverse=True)[0:TOP_THREE]]
        category_interest = ""
        for i in category_top_3:
            category_interest = category_interest + "," + i
        category_interest = category_interest[1:]
        category_list.append(category_interest)
# 创建用户兴趣画像DataFrame。
df_interest = pd.DataFrame({'reviewerID': reviewerID_list, 'top_3_category': category_list, 'top_3_brand': brand_list})

"""
join相关数据，将数据放到同一个DataFrame中
"""
df = pd.merge(df_review, df_feature, how="left", on=['asin'])
df = pd.merge(df, df_interest, how="left", on=['reviewerID'])
df.dropna(how='any', subset=['brand', 'category', 'top_3_category', 'top_3_brand'], inplace=True)
# df_joined.drop_duplicates()
df = df.reset_index(drop=True)


def common_brand(br, top_3_brand):
    br = {br}
    top_3_brand_set = set(top_3_brand.split(','))
    common_brand = br.intersection(top_3_brand_set)
    if len(common_brand) == 0:
        return ''
    else:
        return common_brand.pop()


def common_category(ca, top_3_category):
    ca = set(ca)  # <class 'pandas.core.series.Series'>  [Toys & Games, Grown-Up Toys, Games]
    top_3_category_set = set(top_3_category.split(','))
    common_category = ca.intersection(top_3_category_set)
    common_ca = ""
    for ca_ in common_category:
        common_ca = common_ca + "," + ca_
    common_ca = common_ca[1:]
    return common_ca


df['common_brand'] = df.apply(lambda x: common_brand(x['brand'], x['top_3_brand']), axis=1)
df['common_category'] = df.apply(lambda x: common_category(x['category'], x['top_3_category']), axis=1)


def goods_description(common_b, common_ca):
    if common_b and common_ca:
        return common_b + "," + common_ca
    elif common_b:
        return common_b
    elif common_ca:
        return common_ca
    else:
        return ""


df['label'] = df.apply(lambda x: goods_description(x['common_brand'], x['common_category']), axis=1)
df.dropna(how='any', subset=['reviewerID', 'asin', 'brand', 'category', 'top_3_category',
                             'top_3_brand', 'label'], inplace=True)
df = df.reset_index(drop=True)

prompt = """The following are the the product brand and category data: \n
 product brand: {} \n
 product category: {}  \n
 The following are the user's interests and preferences for product brands and categories:\n 
 the top three brands user likes: {} \n
 the top three categories user likes: {} \n
 Based on the above information, predict the description label of the product, which is obtained from the
 brand and category of the product (i.e., the description label is a subset of the brand and category of
 the product). The description label you provide should meet the user's preferences for the brand and
 category of the product to the greatest extent possible. You just need to list one to four description labels, 
 do not explain the reason.If you given more than one description labels, please separate them with comma."""

df['prompt'] = df.apply(lambda row: prompt.format(str(row['brand']), str(row['category']),
                                                  str(row['top_3_brand']), str(row['top_3_category'])), axis=1)

df = df[['prompt', 'label']]
print(df.shape)
df.to_csv("./data/item_info_data.csv", index=False)
