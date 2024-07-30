import importlib
import json
import sys

sys.path.append('../utils')
utils = importlib.import_module('utils')

TRAIN_RATIO = 0.7

instruction = ("You are a product expert who predicts which products "
               "users prefer based on your professional knowledge.")


def formatting_input(history, candidate):
    input = f"""The user purchased the following beauty products(in JSON format): 

{history}

Predict if the user will prefer to purchase the following beauty candidate list(in JSON format):

{candidate} 

You can choice none, one or more, your output must be JSON format, you just need output item_id, the following is an
output example, A and B is product item_id.

["A", "B"]

Your output must in the candidate list, don't explain.
"""

    return input


"""
按照如下格式构建训练数据集：
[
    {
        "instruction": "You are a product expert who predicts which products users prefer based on your professional knowledge.",
        "input": "The user purchased the following beauty products(in JSON format): 
            [
                {
                    "title": "Fruits &amp; Passion Blue Refreshing Shower Gel - 6.7 fl. oz.",
                    "brand": "Fruits & Passion",
                    "price": "",
                    "item_id": "B000FI4S1E"
                },
                {
                    "title": "Yardley By Yardley Of London Unisexs Lay It On Thick Hand &amp; Foot Cream 5.3 Oz",
                    "brand": "Yardley",
                    "price": "",
                    "item_id": "B0009RF9DW"
                }
            ]
            
        Predict if the user will prefer to purchase the following beauty candidate list(in JSON format):
        
           [
                {
                    "title": "Helen of Troy 1579 Tangle Free Hot Air Brush, White, 3/4 Inch Barrel",
                    "brand": "Helen Of Troy",
                    "price": "$28.70",
                    "item_id": "B000WYJTZG"
                },
                {
                    "title": "Dolce &amp; Gabbana Compact Parfum, 0.05 Ounce",
                    "brand": "Dolce & Gabbana",
                    "price": "",
                    "item_id": "B019V2KYZS"
                },
                ...
            ]
        "output": '["B0012Y0ZG2","B000URXP6E"]'
    },
    ...
]
"""


def generate_data(output_path: str = '../data/train.json'):
    item_dict = utils.get_metadata_dict()
    train_user_dict = utils.get_user_history(data_type="train")
    cold_start_items = utils.get_cold_start_items()
    action_items = set()
    for _, items in train_user_dict.items():
        action_items = action_items.union(items)
    unique_items = action_items.difference(cold_start_items)  # 这里是测试集中不在冷启动中的item集合

    C = []
    for item in unique_items:
        info = item_dict[item]
        info['item_id'] = item
        if 'description' in info:
            del info['description']  # description 字段太长了，消耗的token太多，剔除掉
        C.append(info)
    candidate = json.dumps(C, indent=4, ensure_ascii=False)  # 这是所有训练集中不在冷启动id的item

    data_list = []
    for user, history in train_user_dict.items():
        H = []
        history = [item for item in history if item not in cold_start_items]
        if len(history) > 1:  # 该用户至少还剩余2个action items
            train_num = int(len(history) * TRAIN_RATIO)
            train_history = history[:train_num]
            test_history = history[train_num:]
            for h in train_history:
                info = item_dict[h]
                if 'description' in info:
                    del info['description']  # description 字段太长了，消耗的token太多，剔除掉
                H.append(info)
            HH = json.dumps(H, indent=4, ensure_ascii=False)
            output = json.dumps(test_history, indent=4, ensure_ascii=False)
            input = formatting_input(HH, candidate)
            d = {
                "instruction": instruction,
                "input": input,
                "output": output
            }
            data_list.append(d)

    train_res = json.dumps(data_list, indent=4, ensure_ascii=False)
    with open(output_path, 'a') as file_:  # 将生成的训练数据保存起来
        file_.write(train_res)


if __name__ == "__main__":
    generate_data()
