import importlib
import random
import sys

sys.path.append('../')
utils = importlib.import_module('utils')
from sklearn.model_selection import train_test_split

reviews_beauty = utils.load_json("../data/amazon/beauty/reviews_beauty.json")

combined_review_data = []
no_sentence = 0
for i in range(len(reviews_beauty)):
    rev_ = reviews_beauty[i]
    out = {}
    if 'sentence' in rev_:
        out['user'] = rev_['user']
        out['item'] = rev_['item']
        list_len = len(rev_['sentence'])
        selected_idx = random.randint(0, list_len - 1)  # add a random, or list all possible sentences
        out['explanation'] = rev_['sentence'][selected_idx][2]
        out['feature'] = rev_['sentence'][selected_idx][0]
        combined_review_data.append(out)
    else:
        no_sentence += 1

random.shuffle(combined_review_data)
train, test = train_test_split(combined_review_data, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

outputs = {'train': train,
           'val': val,
           'test': test,
           }

utils.save_json("../data/explain.json", outputs)

user_list = list(set([d['user'] for d in train] + [d['user'] for d in val] + [d['user'] for d in test]))
item_list = list(set([d['item'] for d in train] + [d['item'] for d in val] + [d['item'] for d in test]))

user_dict = {}
for i in range(len(user_list)):
    user_dict[user_list[i]] = i + 1

item_dict = {}
for i in range(len(item_list)):
    item_dict[item_list[i]] = i + 1

dic = {
    "user_dict": user_dict,
    "item_dict": item_dict
}

utils.save_json("../data/finetune_dict.json", dic)
