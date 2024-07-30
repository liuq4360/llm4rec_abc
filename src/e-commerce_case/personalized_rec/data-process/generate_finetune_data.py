import csv
import json

import pandas as pd
from sklearn.model_selection import train_test_split

"""
按照如下格式构建训练数据集：

{

"instruction": "You are a recommendation system expert who provides personalized ranking for items based on the background information provided.", 

"input": "I've ranked the following products in the past(in JSON format):

[
    {
        "title": "SF221-Shaving Factory Straight Razor (Black), Shaving Factory Hand Made Shaving Brush, 100...",
        "brand": "Shaving Factory",
        "price": "$21.95",
        "rating": 2
    },
    ...
]

Based on above rating history, please predict user's rating for the following product(in JSON format):
                 
   {
        "title": "Norelco 4821XL Micro Action Corded/Cordless Rechargeable Men's Shaver",
        "brand": "Norelco",
        "price": "$11.2"
   }           
                 
The ranking is between 1 and 5, 1 being lowest and 5 being highest. 
You just need to ranking the above product, do not explain the reason.
                              
"output": 3

}

"""

instruction = ("You are a recommendation system expert who provides personalized ranking for items "
               "based on the background information provided.")

item_dict = {}  # 从All_beauty.json中获取每个item对应的标题
with open('../../data/amazon_review/beauty/meta_All_Beauty.json', 'r') as file:
    reader = csv.reader(file, delimiter='\n')
    for row in reader:
        j = json.loads(row[0])
        item_id = j['asin']
        title = j['title']
        brand = j['brand']
        price = j['price']
        if price != "" and '$' not in price and len(price) > 10:  # 处理一些异常数据情况
            price = ""
        item_info = {
            "title": title,
            "brand": brand,
            "price": price
        }
        item_dict[item_id] = item_info


def generate_data(data_df, data_type, path):
    grouped_df = data_df.groupby('user')
    groups = grouped_df.groups
    data_list = []
    for user in groups.keys():
        user_df = grouped_df.get_group(user)
        """
        >>> grouped_df.get_group('AZZZ5UJWUVCYZ')
                          item           user  rating   timestamp
            157912  B00IIZG80U  AZZZ5UJWUVCYZ     5.0  1479859200
            250877  B01FNJ9MOW  AZZZ5UJWUVCYZ     5.0  1505865600
            358191  B01CZC20DU  AZZZ5UJWUVCYZ     5.0  1505865600
        """
        if data_type == "train" and user_df.shape[0] < 4:  # 训练数据，每个用户至少需要4条以上数据
            continue
        if data_type == "test" and user_df.shape[0] < 2:  # 测试数据，每个用户至少需要2条以上数据
            continue
        if user_df.shape[0] > 8:  # 数据量太大的不考虑，否则超出了大模型的token范围
            continue
        last_row = user_df.iloc[-1]  # 最后一行用作label
        selected_df = user_df.head(user_df.shape[0] - 1)  # 前面的用于做为特征
        user_ranking_list = []
        for _, row_ in selected_df.iterrows():
            item = row_['item']
            if item in item_dict:
                rating = row_['rating']
                item_info_ = item_dict[item]
                item_info_['rating'] = rating
                user_ranking_list.append(item_info_)
        formatted_user_ranking = json.dumps(user_ranking_list, indent=4)

        item = last_row['item']
        rating = last_row['rating']
        label_item_info_ = item_dict[item]
        if 'rating' in label_item_info_:
            del label_item_info_['rating']  # 去掉rating，这个是需要待预测的

        formatted_item_info = json.dumps(label_item_info_, indent=4)

        input = ("I've ranked the following products in the past(in JSON format):\n\n" +

                 formatted_user_ranking + "\n\n" +

                 "Based on above rating history, please predict user's rating " +
                 "for the following product(in JSON format):" + "\n\n" +

                 formatted_item_info + "\n\n" +

                 "The ranking is between 1 and 5, 1 being lowest and 5 being highest. " +
                 "You just need to ranking the above product, do not explain the reason."
                 )

        output = str(int(rating))

        res_dic = {
            "instruction": instruction,
            "input": input,
            "output": output
        }
        data_list.append(res_dic)

    res = json.dumps(data_list, indent=4, ensure_ascii=False)
    with open(path, 'a') as file_:  # 将生成的训练数据保存起来
        file_.write(res)


train_path = '../data/train.json'
test_path = '../data/test.json'
df = pd.read_csv("../../data/amazon_review/beauty/All_Beauty.csv")
df_shuffled = df.sample(frac=1).reset_index(drop=True)
train_df, test_df = train_test_split(df_shuffled, test_size=0.33, random_state=10)
generate_data(train_df, "train", train_path)
generate_data(test_df, "test", test_path)

"""
    目前train.json 989个样本。
    目前test.json 5517个样本。
    因为train要求每个用户至少要有4个以上的记录。
"""
