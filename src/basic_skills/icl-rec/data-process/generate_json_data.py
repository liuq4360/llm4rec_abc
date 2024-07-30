import csv
import json

"""
按照如下格式构建验证数据集：

{

"instruction": "You are a recommendation system expert who provides personalized recommendations to users based on the background information provided.", 

"input": "I've browsed the following news in the past in order:


[''Wheel Of Fortune' Guest Delivers Hilarious, Off The Rails Introduction','Hard Rock Hotel New Orleans collapse: Former site engineer weighs in','Felicity Huffman begins prison sentence for college admissions scam','Outer Banks storms unearth old shipwreck from 'Graveyard of the Atlantic'','Tiffany's is selling a holiday advent calendar for $112,000','This restored 1968 Winnebago is beyond adorable','Lori Loughlin Is 'Absolutely Terrified' After Being Hit With New Charge','Bruce Willis brought Demi Moore to tears after reading her book','Celebrity kids then and now: See how they've grown','Felicity Huffman Smiles as She Begins Community Service Following Prison Release','Queen Elizabeth Finally Had Her Dream Photoshoot, Thanks to Royal Dresser Angela Kelly','Hundreds of thousands of people in California are downriver of a dam that 'could fail'','Alexandria Ocasio-Cortez 'sincerely' apologizes for blocking ex-Brooklyn politician on Twitter, settles federal lawsuit','The Rock's Gnarly Palm Is a Testament to Life Without Lifting Gloves']


Then if I ask you to recommend a new news to me according to my browsing history, you should recommend 'Donald Trump Jr. reflects on explosive 'View' chat: 'I don't think they like me much anymore'' and now that I've just browsed 'Donald Trump Jr. reflects on explosive 'View' chat: 'I don't think they like me much anymore'', there are 22 candidate news that I can browse next:
1. 'Browns apologize to Mason Rudolph, call Myles Garrett's actions 'unacceptable'',
2. 'I've been writing about tiny homes for a year and finally spent 2 nights in a 300-foot home to see what it's all about   here's how it went',
3. 'Opinion: Colin Kaepernick is about to get what he deserves: a chance',
4. 'The Kardashians Face Backlash Over 'Insensitive' Family Food Fight in KUWTK Clip',
5. 'THEN AND NOW: What all your favorite '90s stars are doing today',6. 'Report: Police investigating woman's death after Redskins' player Montae Nicholson took her to hospital',
7. 'U.S. Troops Will Die If They Remain in Syria, Bashar Al-Assad Warns',
8. '3 Indiana judges suspended after a night of drinking turned into a White Castle brawl',
9. 'Cows swept away by Hurricane Dorian found alive   but how?',
10. 'Surviving Santa Clarita school shooting victims on road to recovery: Latest',
11. 'The Unlikely Star of My Family's Thanksgiving Table',
12. 'Meghan Markle and Hillary Clinton Secretly Spent the Afternoon Together at Frogmore Cottage',
13. 'Former North Carolina State, NBA player Anthony Grundy dies in stabbing, police say',
14. '85 Thanksgiving Recipes You Can Make Ahead',
15. 'Survivor Contestants Missy Byrd and Elizabeth Beisel Apologize For Their Actions',
16. 'Pete Davidson, Kaia Gerber Are Dating, Trying to Stay 'Low Profile'',
17. 'There's a place in the US where its been over 80 degrees since March',
18. 'Taylor Swift Rep Hits Back at Big Machine, Claims She's Actually Owed $7.9 Million in Unpaid Royalties',
19. 'The most talked about movie moments of the 2010s',
20. 'Belichick mocks social media in comments on Garrett incident',
21. '13 Reasons Why's Christian Navarro Slams Disney for Casting 'the White Guy' in The Little Mermaid',
22. '66 Cool Tech Gifts Anyone Would Be Thrilled to Receive'


Please select some news that I would like to browse next according to my browsing history. Please think step by step.


Please show me your results. Split your output with line break. You MUST select from the given candidate news. You can not generate news that are not in the given candidate list."

"output": "['Opinion: Colin Kaepernick is about to get what he deserves: a chance'\n]"

}

"""

instruction = ("You are a recommendation system expert who provides personalized recommendations to users based on "
               "the background information provided.")

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
        history = row[3]
        impression = row[4]
        history_list = history.split(" ")
        impre_list = impression.split(" ")
        if len(history_list) >= 6 and len(impre_list) >= 10:  # 用户至少要有6个点击历史和10个曝光历史
            his = []
            for news_id in history_list[:-1]:
                title = news_dict[news_id]
                his.append(title)
            last_view_id = history_list[-1]
            last_view = news_dict[last_view_id]
            preference = []
            candidate = []
            index = 1
            for impre in impre_list:
                [impre_id, action] = impre.split("-")
                title = news_dict[impre_id]
                candidate.append(str(index) + ". " + title + "\n")
                index = index + 1
                if int(action) == 1:
                    preference.append(title + "\n")
            input = ("I’ve browsed the following news in the past in order:\n\n" +
                     "[" + ','.join(his) + "]" + "\n\n" +
                     "Then if I ask you to recommend a new news to me according to my browsing history, "
                     "you should recommend " + last_view + " and now that I've just browsed " + last_view + ", " +
                     "there are " + str(len(impre_list)) + " candidate news that I can browse next:" +
                     ','.join(candidate) + "\n\n" +
                     "Please select some news that I would like to browse next according to my browsing history. " +
                     "Please think step by step.\n\n" +
                     "Please show me your results. Split your output with line break. You MUST select from the given " +
                     "candidate news. You can not generate news that are not in the given candidate list."
                     )
            output = "[\n" + ','.join(preference) + "\n]"
            res_dic = {
                "instruction": instruction,
                "input": input,
                "output": output
            }
            data_list.append(res_dic)

res = json.dumps(data_list, indent=4, ensure_ascii=False)
user_sequence_path = "../data/mind/test.json"  # 将生成的训练数据保存起来
with open(user_sequence_path, 'a') as file:
    file.write(res)
