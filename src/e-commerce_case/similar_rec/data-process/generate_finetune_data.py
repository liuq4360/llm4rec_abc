import json
import random

"""
按照如下格式构建训练、测试数据集：

{

"instruction": "You are a product expert who judges whether 
two products are similar based on your professional knowledge.", 

"input": "I will provide you with two product related introduction information, as follows(in JSON format):

[
    {
        "title": "SF221-Shaving Factory Straight Razor (Black), Shaving Factory Hand Made Shaving Brush, 100...",
        "brand": "Shaving Factory",
        "price": "$21.95",
        "description": ["Start Up combines citrus essential oils with gentle Alpha Hydroxy Acids to cleanse and refresh
            your face. The 5% AHA level is gentle enough for all skin types.", "", ""],
    },
    {
        "title": "Loud 'N Clear&trade; Personal Sound Amplifier",
        "brand": "idea village",
        "price": "",
        "description": ["Loud 'N Clear Personal Sound Amplifier allows you to turn up the volume on what people around 
            you are saying, listen at the level you want without disturbing others, hear a pin drop from across the room."],
    }
]

Based on above information, please predict if these two products are similar. The similarity  is between 0 and 2, 
0 being lowest and 2 being highest. You just need to ranking the above product, do not explain the reason.

"output": "0"

}

"""

"""
构建训练集、测试集的思路：
基于metadata数据集中also_buy、also_view 字段，某个商品与also_buy、also_view中的商品认为是相似的，这些可以做为正样本。
但他们的相似度应该不一样，also_buy是更强烈的偏好，我们设置相似度为2，also_view设置为1
为了让训练样本更加平衡，可以随机选择两个商品对做为负样本，选择负样本的数量跟正样本差不多。负样本设置相似度为0。
下面就基于这个思路来进行处理。

"""

instruction = ("You are a product expert who judges whether "
               "two products are similar based on your professional knowledge.")


def generate_data(out_path: str, item_dict: dict, test_ratio: float = 0.3):
    data_list = []
    # 构建正样本
    for item in item_dict.keys():

        info = item_dict[item]
        title = info['title']
        brand = info['brand']
        price = info['price']
        description = info['description']
        also_view = info['also_view']
        also_buy = info['also_buy']
        _dict = {
            "title": title,
            "brand": brand,
            "price": price,
            "description": description
        }
        s = set(also_view).union(set(also_buy))
        for i in s:
            if i in item_dict:
                i_dict = {
                    "title": item_dict[i]['title'],
                    "brand": item_dict[i]['brand'],
                    "price": item_dict[i]['price'],
                    "description": item_dict[i]['description']
                }
                positive_sample_pair = [_dict, i_dict]
                formatted_input = json.dumps(positive_sample_pair, indent=4, ensure_ascii=False)
                input = ("I will provide you with two product related introduction information, as follows(in JSON " +
                         "format):\n\n" +
                         formatted_input + "\n\n" +
                         "Based on above information, please predict if these two products are similar. The similarity " +
                         "is between 0 and 2, 0 being lowest and 2 being highest. You just need to ranking the above " +
                         "product, do not explain the reason.")
                if i in also_buy:
                    output = "2"
                else:
                    output = "1"
                res_dic = {
                    "instruction": instruction,
                    "input": input,
                    "output": output
                }
                data_list.append(res_dic)

    # 构建负样本
    positive_sample_num = len(data_list)
    item_set = item_dict.keys()
    for i in range(positive_sample_num):
        negative_sample_pair = random.sample(item_set, 2)  # [1, 2]
        a_dict = {
            "title": item_dict[negative_sample_pair[0]]['title'],
            "brand": item_dict[negative_sample_pair[0]]['brand'],
            "price": item_dict[negative_sample_pair[0]]['price'],
            "description": item_dict[negative_sample_pair[0]]['description']
        }
        b_dict = {
            "title": item_dict[negative_sample_pair[1]]['title'],
            "brand": item_dict[negative_sample_pair[1]]['brand'],
            "price": item_dict[negative_sample_pair[1]]['price'],
            "description": item_dict[negative_sample_pair[1]]['description']
        }
        negative_sample_pair = [a_dict, b_dict]
        formatted_input = json.dumps(negative_sample_pair, indent=4, ensure_ascii=False)
        input = ("I will provide you with two product related introduction information, as follows(in JSON " +
                 "format):\n\n" +
                 formatted_input + "\n\n" +
                 "Based on above information, please predict if these two products are similar. The similarity " +
                 "is between 0 and 2, 0 being lowest and 2 being highest. You just need to ranking the above " +
                 "product, do not explain the reason.")
        res_dic = {
            "instruction": instruction,
            "input": input,
            "output": "0"
        }
        data_list.append(res_dic)

    # 将数据拆分为训练集和测试集
    random.shuffle(data_list)
    split_loc = int(len(data_list) * test_ratio)
    test_data_list = data_list[0: split_loc]
    train_data_list = data_list[split_loc:]
    test_res = json.dumps(test_data_list, indent=4, ensure_ascii=False)
    train_res = json.dumps(train_data_list, indent=4, ensure_ascii=False)
    with open(out_path + "/test.json", 'a') as file_:  # 将生成的训练数据保存起来
        file_.write(test_res)
    with open(out_path + "/train.json", 'a') as file_:  # 将生成的训练数据保存起来
        file_.write(train_res)


from generate_item_dict import get_metadata_dict

item_dict = get_metadata_dict()
generate_data("../data", item_dict, 0.3)

"""
    目前train.json 4616个样本。
    目前test.json 3706个样本。
"""
