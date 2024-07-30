import os
import time
import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from tqdm import tqdm

MIN_INTERVAL = 1.5

# 新闻数据中包含的字段说明
keys = dict(
    title='title',
    abstract='abs',
    category='cat',
    subcategory='subcat',
)

current_path = os.getcwd()

# 将新闻读到dataframe中
news_df = pd.read_csv(
    filepath_or_buffer=os.path.join(current_path + '/data/news.tsv'),
    sep='\t',
    header=0,
)

# 构建新闻列表，每一个元素是元组，元组前面是新闻id，后面是dict，dict是新闻相关信息, 下面是一条数据样本
# ('N55528', {'title': 'The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By',
#             'abstract': "Shop the notebooks, jackets, and more that the royals can't live without.",
#             'category': 'lifestyle', 'subcategory': 'lifestyleroyals', 'newtitle': ''})
news_list = []
for news in tqdm(news_df.iterrows()):
    dic = {}
    for key in keys:
        dic[key] = news[1][keys[key]]
    news_list.append((news[1]['nid'], dic))

# 提示词模板
prompt_template = """You are asked to act as a news title enhancer. I will provide you a piece of news, with its original title, category, subcategory, and abstract (if exists). The news format is as below:

[title] {title}
[abstract] {abstract}
[category] {category}
[subcategory] {subcategory}

where title, abstract, category, and subcategory in the brace will be filled with content. You can only response a rephrased news title which should be clear, complete, objective and neutral. You can expand the title according to the above requirements. You are not allowed to response any other words for any explanation. Your response format should be:

[newtitle] 

where [newtitle] should be filled with the enhanced title. Now, your role of news title enhancer formally begins. Any other information should not disturb your role."""

# 生成的新的新闻标题的存储路径
save_path = current_path + '/output/news_summarizer.log'

# 下面是调用LLAMA大模型的语法
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/Users/liuqiang/Desktop/code/llm/models/gguf/qwen1.5-72b-chat-q5_k_m.gguf",
    temperature=0.8,
    top_p=0.8,
    n_ctx=6000,
    callback_manager=callback_manager,
    verbose=True,
    # stop=["<|im_end|>"]  # 生成的答案中遇到这些词就停止生成
)

"""
PromptTemplate 与 chain的使用案例：
（1）文本的使用方式
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))

（2）字典的使用方式
prompt = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({
    'company': "ABC Startup",
    'product': "colorful socks"
    }))
"""

prompt = PromptTemplate(
    input_variables=["title", "abstract", "category", "subcategory"],
    template=prompt_template,
)
chain = LLMChain(llm=llm, prompt=prompt)

# 先统计出哪些已经计算了，避免后面重复计算
exist_set = set()
with open(save_path, 'r') as f:
    for line in f:
        if line and line.startswith('N'):
            exist_set.add(line.split('\t')[0])

# 调用大模型迭代计算新闻标题
for nid, content in tqdm(news_list):
    start_time = time.time()
    if nid in exist_set:
        continue
    try:
        title = content['title']
        abstract = content['abstract']
        category = content['category']
        subcategory = content['subcategory']
        enhanced = chain.run(title=title, abstract=abstract, category=category, subcategory=subcategory)
        enhanced = enhanced.rstrip('\n')
        with open(save_path, 'a') as f:
            f.write(f'{nid}\t{enhanced}\n')
    except Exception as e:
        print(e)

    interval = time.time() - start_time
    if interval <= MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - interval)
