import csv
import json
from enum import Enum


class Action(Enum):
    YES = "Yes."
    NO = "No."


"""
按照如下格式构建训练数据集
{"instruction": "Given the user's preference and unpreference, identify whether the user will like the target movie by 
answering \"Yes.\" or \"No.\".", 
"input": "User Preference: \"Opinion: Colin Kaepernick is about to get what he deserves: a chance\"\nUser Unpreference: 
\"Browns apologize to Mason Rudolph, call Myles Garrett's actions 'unacceptable'\",
\"I've been writing about tiny homes for a year and finally spent 2 nights in a 300-foot home to see what it's all about 
  here's how it went\",\"The Kardashians Face Backlash Over 'Insensitive' Family Food Fight in KUWTK Clip\",
  \"THEN AND NOW: What all your favorite '90s stars are doing today\",\"Report: Police investigating woman's death after 
  Redskins' player Montae Nicholson took her to hospital\",\"U.S. Troops Will Die If They Remain in Syria, 
  Bashar Al-Assad Warns\",\"3 Indiana judges suspended after a night of drinking turned into a White Castle brawl\",
  \"Cows swept away by Hurricane Dorian found alive   but how?\",\"Surviving Santa Clarita school shooting victims on 
  road to recovery: Latest\",\"The Unlikely Star of My Family's Thanksgiving Table\",\"Meghan Markle and Hillary Clinton
   Secretly Spent the Afternoon Together at Frogmore Cottage\",\"Former North Carolina State, NBA player Anthony Grundy 
   dies in stabbing, police say\",\"85 Thanksgiving Recipes You Can Make Ahead\",\"Survivor Contestants Missy Byrd and 
   Elizabeth Beisel Apologize For Their Actions\",\"Pete Davidson, Kaia Gerber Are Dating, Trying to Stay 'Low Profile'\"
   ,\"There's a place in the US where its been over 80 degrees since March\",\"Taylor Swift Rep Hits Back at Big Machine,
    Claims She's Actually Owed $7.9 Million in Unpaid Royalties\",\"The most talked about movie moments of the 2010s\",
    \"Belichick mocks social media in comments on Garrett incident\",\"13 Reasons Why's Christian Navarro Slams Disney 
    for Casting 'the White Guy' in The Little Mermaid\"\nWhether the user will like the targe news \"66 Cool Tech Gifts 
    Anyone Would Be Thrilled to Receive\"?", 
    "output": "No."},
...
]
"""

instruction = ("Given the user's preference and unpreference, identify whether the user will like the target movie by "
               "answering \"Yes.\" or \"No.\".")

news_dict = {}  # 从news.tsv获取每个新闻id到标题的映射字典。
with open('../data/mind/news.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        news_id = row[0]
        news_title = row[3]
        news_dict[news_id] = news_title

data_list = []  # 利用behaviors.tsv数据获取用户喜欢和不喜欢的新闻
with open('../data/mind/behaviors.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        impression = row[4]
        impre_list = impression.split(" ")
        if len(impre_list) >= 5:  # 用户至少要有5个曝光历史
            preference = []
            unpreference = []
            for impre in impre_list[:-1]:  # 利用前面的新闻做为训练数据，最后一个新闻做为预测。
                [impre_id, action] = impre.split("-")
                title = news_dict[impre_id]
                if int(action) == 1:
                    preference.append("\"" + title + "\"")
                else:
                    unpreference.append("\"" + title + "\"")
            input = "User Preference: " + ','.join(preference) + "\n" + "User Unpreference: " + ','.join(unpreference)
            [impre_id, action] = impre_list[-1].split("-")
            output = Action.YES.value if int(action) == 1 else Action.NO.value
            input = input + "\n" + "Whether the user will like the targe news " + "\"" + news_dict[impre_id] + "\"?"
            res_dic = {
                "instruction": instruction,
                "input": input,
                "output": output
            }
            data_list.append(res_dic)

res = json.dumps(data_list, indent=4, ensure_ascii=False)
user_sequence_path = "../data/mind/train.json"  # 将生成的训练数据保存起来
with open(user_sequence_path, 'a') as file:
    file.write(res)
