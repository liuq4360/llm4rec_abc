import csv
import importlib
import json
import sys

sys.path.append('../')
utils = importlib.import_module('utils')


def get_recommendation_explain_data(path_review: str = "../../data/amazon_review/beauty/All_Beauty_5.json") -> dict:
    """
    将用户行为数据按照时间升序排列，取用户行为字典
    :param path_review: 用户行为数据目录
    :return: 生成预测推荐解释的数据
    """
    df_view = utils.get_df(path_review)

    # 对unixReviewTime升序排序
    df_view.sort_values('unixReviewTime', ascending=True, inplace=True)
    df_view = df_view.reset_index(drop=True)

    grouped = df_view.groupby('reviewerID')
    """
        >>> grouped.get_group('A105A034ZG9EHO')
      overall  verified  reviewTime      reviewerID        asin              style reviewerName reviewText     summary  unixReviewTime vote image
1246      5.0      True  07 6, 2014  A105A034ZG9EHO  B0009RF9DW  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1247      5.0      True  07 6, 2014  A105A034ZG9EHO  B000FI4S1E                NaN      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1250      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1252      5.0      True  07 6, 2014  A105A034ZG9EHO  B000URXP6E  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1253      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
    """
    user_action_dict = {}
    for name, group in grouped:
        reviewerID = name
        asin = list(group['asin'])
        dic = {
            "action": asin[0:-1],
            "recommendation": asin[-1]
        }
        user_action_dict[reviewerID] = dic

    return user_action_dict


def get_metadata_dict(path: str = '../../data/amazon_review/beauty/meta_All_Beauty.json') -> dict:
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


if __name__ == "__main__":
    rec_dict = get_recommendation_explain_data()
    item_dict = get_metadata_dict()
    dic = {
        "rec_dict": rec_dict,
        "item_dict": item_dict
    }
    utils.save_json("../data/icl_dict.json", dic)
