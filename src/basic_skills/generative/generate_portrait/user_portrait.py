import json
import os
import time
import pandas as pd
from UniTok import UniDep
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from prompter import MindPrompter, MindUser

MIN_INTERVAL = 0

current_path = os.getcwd()

mind_prompter = MindPrompter(current_path + '/data/news.tsv')
user_list = MindUser(current_path + '/data/user', mind_prompter).stringify()

# 将新闻读到dataframe中
news_df = pd.read_csv(
    filepath_or_buffer=os.path.join(current_path + '/data/news.tsv'),
    sep='\t',
    header=0,
)

# 构建新闻字典，key为新闻id，value为新闻的title
news_dict = {}
for news in tqdm(news_df.iterrows()):
    news_dict[news[1]['nid']] = news[1]['title']

depot = UniDep(current_path + '/data/user', silent=True)
nid = depot.vocabs('nid')

# 生成每个用户及他看过的新闻数据，下面是一条案例
# (0, ["'Wheel Of Fortune' Guest Delivers Hilarious, Off The Rails Introduction",
# "Three takeaways from Yankees' ALCS Game 5 victory over the Astros",
# "Rosie O'Donnell: Barbara Walters Isn't 'Up to Speaking to People' Right Now",
# "Four flight attendants were arrested in Miami's airport after bringing in thousands in cash, police say",
# 'Michigan sends breakup tweet to Notre Dame as series goes on hold',
# "This Wedding Photo of a Canine Best Man Captures Just How Deep a Dog's Love Truly Is",
# "Robert Evans, 'Chinatown' Producer and Paramount Chief, Dies at 89",
# 'Former US Senator Kay Hagan dead at 66',
# 'Joe Biden reportedly denied Communion at a South Carolina church because of his stance on abortion'])
# user_list = []
# for user in tqdm(depot):
#     list = []
#     if not user['history']:
#         user_list.append((user['uid'], None))
#     for i, n in enumerate(user['history']):
#         list.append(news_dict[nid.i2o[n]])
#     user_list.append((user['uid'], list))

# 提示词模板
system = """You are asked to describe user interest based on his/her browsed news title list, the format of which is as below:

{input}

You can only response the user interests with the following format to describe the [topics] and [regions] of the user's interest

[topics]
- topic1
- topic2
...
[region] (optional)
- region1
- region2
...

where topic is limited to the following options: 

(1) health
(2) education
(3) travel
(4) religion
(5) culture
(6) food
(7) fashion
(8) technology
(9) social media
(10) gender and sexuality
(11) race and ethnicity
(12) history
(13) economy
(14) finance
(15) real estate
(16) transportation
(17) weather
(18) disasters
(19) international news

and the region should be limited to each state of the US.

Only [topics] and [region] can be appeared in your response. If you think region are hard to predict, leave it blank. Your response topic/region list should be ordered, that the first several options should be most related to the user's interest. You are not allowed to response any other words for any explanation or note. Now, the task formally begins. Any other information should not disturb you."""

# 生成的用户兴趣画像的存储路径
save_path = current_path + '/output/user_profiler.log'

# 下面是调用LLAMA大模型的语法
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/liuqiang/Desktop/code/llm/models/gguf/qwen1.5-72b-chat-q5_k_m.gguf",
    temperature=0.8,
    top_p=0.8,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True,
)

# 先统计出哪些已经计算了，避免后面重复计算
exist_set = set()
with open(save_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data['uid'])

empty_count = 0

# 调用大模型迭代计算用户兴趣画像
for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
        empty_count += 1
        continue

    try:
        prompt = PromptTemplate(
            input_variables=["input"],
            template=system,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        enhanced = chain.run(input=content)
        enhanced = enhanced.rstrip('\n')
        with open(save_path, 'a') as f:
            f.write(json.dumps({'uid': uid, 'interest': enhanced}) + '\n')
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)

print('empty count: ', empty_count)
