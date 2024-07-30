import csv
import json

import pandas as pd

TRAIN_RATIO = 0.7


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


def get_cold_start_items(path_review: str = "../data/amazon_review/beauty/All_Beauty_5.json") -> set[str]:
    """
    将用户行为数据按照时间升序排列，取后面的30%的数据，该数据中的item在前面的70%中不存在，就认为是冷启动数据
    :param path_review: 用户行为数据目录
    :return: 冷启动物品
    """

    df_view = get_df(path_review)

    # 对unixReviewTime升序排序
    df_view.sort_values('unixReviewTime', ascending=True, inplace=True)
    df_view = df_view.reset_index(drop=True)

    rows_num = df_view.shape[0]
    train_num = int(rows_num * 0.7)

    train_df = df_view.head(train_num)
    test_df = df_view.iloc[train_num:]

    train_items = set(train_df['asin'].unique())  # 71个
    test_items = set(test_df['asin'].unique())  # 44个

    cold_start_items = test_items.difference(train_items)  # 14个

    return cold_start_items


def get_user_history(path_review: str = "../data/amazon_review/beauty/All_Beauty_5.json",
                     data_type: str = "train") -> dict:
    """
    将用户行为数据按照时间升序排列，取用户行为字典
    :param data_type: 是取前面70%的训练数据，还是后面30%的测试数据
    :param path_review: 用户行为数据目录
    :return: 用户行为历史
    """
    df_view = get_df(path_review)

    # 对unixReviewTime升序排序
    df_view.sort_values('unixReviewTime', ascending=True, inplace=True)
    df_view = df_view.reset_index(drop=True)

    rows_num = df_view.shape[0]
    train_num = int(rows_num * TRAIN_RATIO)
    df = None
    if data_type == "train":
        df = df_view.head(train_num)
    if data_type == "test":
        df = df_view.iloc[train_num:]

    grouped = df.groupby('reviewerID')
    """
        >>> grouped.get_group('A105A034ZG9EHO')
      overall  verified  reviewTime      reviewerID        asin              style reviewerName reviewText     summary  unixReviewTime vote image
1246      5.0      True  07 6, 2014  A105A034ZG9EHO  B0009RF9DW  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1247      5.0      True  07 6, 2014  A105A034ZG9EHO  B000FI4S1E                NaN      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1250      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1252      5.0      True  07 6, 2014  A105A034ZG9EHO  B000URXP6E  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1253      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
    """
    user_history_dict = {}
    for name, group in grouped:
        reviewerID = name
        asin = group['asin']
        user_history_dict[reviewerID] = set(asin)

    return user_history_dict


def get_metadata_dict(path: str = '../data/amazon_review/beauty/meta_All_Beauty.json') -> dict:
    """
    读取商品metadata数据，将商品的核心进行取出来，方便后面大模型使用
    :param path: 商品metadata数据目录
    :return: 商品信息字典
    """
    item_dict = {}  # meta_All_Beauty.json中获取每个item对应的信息
    """
    {"category": [], "tech1": "", "description": ["Start Up combines citrus essential oils with gentle Alpha Hydroxy Acids to cleanse and refresh your face. The 5% AHA level is gentle enough for all skin types.", "", ""], 
    "fit": "", "title": "Kiss My Face Exfoliating Face Wash Start Up, 4 Fluid Ounce", 
    "also_buy": ["B000Z96JDI", "B00006IGL8", "B007C5X34G", "B00006IGLF", "B00213WCNC", "B00D1W1QXE", "B001FB5HZG", "B000FQ86RI", "B0012BSKBM", "B0085EVLRO", "B00A2EXVQE"], 
    "tech2": "", "brand": "Kiss My Face", "feature": [], "rank": [], 
    "also_view": ["B000Z96JDI", "B001FB5HZG", "B00213WCNC", "B00BBFOVO4", "B0085EVLRO"], 
    "details": {"\n    Product Dimensions: \n    ": "2.5 x 1.6 x 7 inches ; 4 ounces", "Shipping Weight:": "4 ounces", "ASIN: ": "B00006IGL2", "UPC:": "890795851488 701320351987 601669038184 793379218755 028367831938 787734768894 756769626417", "Item model number:": "1200040"},
     "main_cat": "All Beauty", "similar_item": "", "date": "", "price": "", 
     "asin": "B00006IGL2", "imageURL": ["https://images-na.ssl-images-amazon.com/images/I/41i07fBAznL._SS40_.jpg", 
     "https://images-na.ssl-images-amazon.com/images/I/31W8DZRVD1L._SS40_.jpg"], "imageURLHighRes": ["https://images-na.ssl-images-amazon.com/images/I/41i07fBAznL.jpg", "https://images-na.ssl-images-amazon.com/images/I/31W8DZRVD1L.jpg"]}

    """
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\n')
        for row in reader:
            j = json.loads(row[0])
            item_id = j['asin']
            title = j['title']
            brand = j['brand']
            description = j['description']
            price = j['price']
            if price != "" and '$' not in price and len(price) > 10:  # 处理一些异常数据情况
                price = ""
            item_info = {
                "title": title,
                "brand": brand,
                "description": description,
                "price": price
            }
            item_dict[item_id] = item_info
    return item_dict
