import json
import os
import time

import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from tqdm import tqdm

from prompter import MindPrompter, MindColdUser

MIN_INTERVAL = 0
current_path = os.getcwd()

mind_prompter = MindPrompter(current_path + '/data/news.tsv')
user_list = MindColdUser(current_path + '/data/user', mind_prompter).stringify()

# 将新闻读到dataframe中
news_df = pd.read_csv(
    filepath_or_buffer=os.path.join(current_path + '/data/news.tsv'),
    sep='\t',
    header=0,
)

system = """You are asked to capture user's interest based on his/her browsing history, and generate a piece of news that he/she may be interested. The format of history is as below:

{input}

You can only generate a piece of news (only one) in the following json format, the json include 3 keys as follows:

<title>, <abstract>, <category>

where <category> is limited to the following options:

- lifestyle
- health
- news
- sports
- weather
- entertainment
- autos
- travel
- foodanddrink
- tv
- finance
- movies
- video
- music
- kids
- middleeast
- northamerica
- games


<title>, <abstract>, and <category> should be the only keys in the json dict. The news should be diverse, that is not too similar with the original provided news list. You are not allowed to response any other words for any explanation or note. JUST GIVE ME JSON-FORMAT NEWS. Now, the task formally begins. Any other information should not disturb you."""

save_path = current_path + '/output/personalized_news_summary.log'

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

# 调用大模型迭代计算用户可能会喜欢的新闻
for uid, content in tqdm(user_list):
    start_time = time.time()
    if uid in exist_set:
        continue

    if not content:
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
            f.write(json.dumps({'uid': uid, 'news': enhanced}) + '\n')
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)
