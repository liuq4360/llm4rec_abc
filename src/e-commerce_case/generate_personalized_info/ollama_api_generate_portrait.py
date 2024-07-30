import json
import requests

temperature = 0.1
top_p = 0.95
ctx = 13000

prompt = """
你的任务是基于用户购买的历史商品，用2个关键字来总结用户的兴趣偏好，按照用户兴趣偏好的大小排序，最喜欢的排在最前面。

下面英文是用户喜欢的品牌，同一行的多个品牌用逗号分割，可以有多行，每个品牌放在''之间。


'Pre de Provence', 'Paul Brown Hawaii', 'Michel Design Works',
'Pre de Provence', 'Pre de Provence', 'Pre de Provence',
'Pre de Provence', 'Pre de Provence', 'Pre de Provence',
'Pre de Provence', 'Pre de Provence', 'Greenwich Bay Trading Company',
'Michel Design Works', 'Pre de Provence', 'Calgon', 'Vinolia',
'AsaVea', 'Bali Soap'


上面的品牌可以看成是用户的兴趣偏好，品牌出现的次数越多，说明用户兴趣偏好越大。所以，你需要先统计每个品牌出现的次数，然后按照次数降序排列，最终选择前2个输出。

输出模板为：x、y。现在请给出你的输出。

"""

url = "http://localhost:11434/api/chat"  # Ollama的api地址
data = {
    "model": "yi:34b-chat",  # Ollama安装的模型名
    "options": {
        "temperature": temperature,
        "top_p": top_p,
        "num_ctx": ctx,
        "num_gpu": 128,
    },
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ]
}
response = requests.post(url=url, json=data, stream=True)
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    j = json.loads(chunk.decode('utf-8'))
    print(j['message']['content'], end="")
