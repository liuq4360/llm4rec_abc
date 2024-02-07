import json
import os
import time
from tqdm import tqdm
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate

MIN_INTERVAL = 0
current_path = os.getcwd()

save_path = current_path + '/output/personalized_news.log'

prompt_template = """
you are a news writing expert, The titles of news a user browsed are as follows:

"interest": {interest}

The news in the curly braces above are a list of all the news that the user has browsed, which may contain multiple news articles.
the information in the titles stand for the category of news the user likes.
Now please write a new for the user, the news must relevant to the interest of the user, the news you write must less than 300 words.

"""

# 下面是调用LLAMA大模型的语法
# 下面是调用LLAMA大模型的语法
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="../../models/nous-hermes-2-yi-34b.Q5_K_M.gguf",
    # model_path="../../models/llama-2-7b-chat.Q5_K_M.gguf",
    temperature=0.8,
    top_p=0.8,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True,
)

prompt = PromptTemplate(
    input_variables=["interest"],
    template=prompt_template,
)
chain = LLMChain(llm=llm, prompt=prompt)

# 先统计出哪些已经计算了，避免后面重复计算
exist_set = set()
with open(save_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        exist_set.add(data['uid'])

# 打开文件并创建文件对象
file = open(current_path + '/output/personalized_news_summary.log', "r")

# 使用 readlines() 方法将文件内容按行存入列表 lines
lines = file.readlines()

# 关闭文件
file.close()

# 输出文件内容
for line in tqdm(lines):
    info = eval(line)
    uid = info['uid']
    titles = info['news']

    start_time = time.time()
    if uid in exist_set:
        continue

    if not titles:
        continue

    try:
        enhanced = chain.run(titles)
        enhanced = enhanced.rstrip('\n')
        news = {"uid": uid, "news": enhanced}
        with open(save_path, 'a') as f:
            f.write(f'{str(news)}\n')
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)