import json
import os
import time

import torch
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()  # https://vault.dotenv.org/ui/ui1

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

OUTPUT_NUM = 50  # 只针对前面的OUTPUT_NUM个用户给出推荐原因，避免计算时间太长

instruction = ("You are a product recommendation expert, and when recommending "
               "products to users, you will provide reasons for the recommendation.")


def formatting_prompt(history, recommendation):
    prompt = f"""The user purchased the following beauty products(in JSON format): 

{history}

Based on the user's historical purchases, our recommendation system recommends the following 
product to the user(in JSON format):

{recommendation} 

Please provide a recommendation explanation in one sentence, which is why the recommendation system 
recommends this product to users. Your reasons must be clear, easy to understand, and able to be recognized by users.
The reason you provide should be between 5 and 20 words.
"""

    return prompt


def llm_api_rec_reason(dict_path: str = 'data/icl_dict.json', store_path: str = 'data/llm_api_reasons.json'):
    f = open(dict_path, "rb")
    dic = json.load(f)

    item_dict = dic['item_dict']
    train_user_dict = dic['rec_dict']

    generated_reasons = []
    print("total user number = " + str(len(train_user_dict.keys())))
    i = 0
    for user in list(train_user_dict.keys())[0:OUTPUT_NUM]:
        H = []
        dic = train_user_dict[user]
        action_list = dic['action']
        for h in action_list:
            info = item_dict[h]
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            H.append(info)
        history = json.dumps(H, indent=4, ensure_ascii=False)
        item = dic['recommendation']
        info = item_dict[item]
        info['item_id'] = item
        if 'description' in info:
            del info['description']  # description 字段太长了，消耗的token太多，剔除掉
        recommendation = json.dumps(info, indent=4, ensure_ascii=False)
        prom = formatting_prompt(history, recommendation)
        client = OpenAI(
            api_key=MOONSHOT_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
        llm_response = client.chat.completions.create(
            model="moonshot-v1-32k",  # moonshot-v1-8k 、moonshot-v1-32k、moonshot-v1-128k
            messages=[
                {
                    "role": "system",
                    "content": instruction,
                },
                {"role": "user", "content": prom},
            ],
            temperature=0.1,
            stream=False,
        )
        reason = llm_response.choices[0].message.content.strip()
        explain = {
            "user": user,
            "prompt": prom,
            "reason": reason
        }
        i += 1
        print("-------------- " + str(i) + " -----------------")
        print(json.dumps(explain, indent=4, ensure_ascii=False))
        generated_reasons.append(explain)
        if i % 7 == 0:
            time.sleep(1)  # 避免moonshot认为调用太频繁不合法

    res = json.dumps(generated_reasons, indent=4, ensure_ascii=False)

    with open(store_path, 'a') as file:  # 将生成的训练数据保存起来
        file.write(res)


def openllm_rec_reason(model_path: str = '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-14B',
                       dict_path: str = 'data/icl_dict.json',
                       store_path: str = 'data/openllm_reasons.json'):
    f = open(dict_path, "rb")
    dic = json.load(f)

    item_dict = dic['item_dict']
    train_user_dict = dic['rec_dict']

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    generated_reasons = []
    print("total user number = " + str(len(train_user_dict.keys())))
    i = 0
    for user in list(train_user_dict.keys())[0:OUTPUT_NUM]:
        H = []
        dic = train_user_dict[user]
        action_list = dic['action']
        for h in action_list:
            info = item_dict[h]
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            H.append(info)
        history = json.dumps(H, indent=4, ensure_ascii=False)
        item = dic['recommendation']
        info = item_dict[item]
        info['item_id'] = item
        if 'description' in info:
            del info['description']  # description 字段太长了，消耗的token太多，剔除掉
        recommendation = json.dumps(info, indent=4, ensure_ascii=False)

        input = formatting_prompt(history, recommendation)

        prompt = f"""### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    """

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
        outputs = model.generate(input_ids=input_ids.to('mps'),
                                 max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        reason = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        explain = {
            "user": user,
            "prompt": prompt,
            "reason": reason
        }
        i += 1
        print("-------------- " + str(i) + " -----------------")
        print(json.dumps(explain, indent=4, ensure_ascii=False))
        generated_reasons.append(explain)
        if i % 7 == 0:
            time.sleep(1)  # 避免moonshot认为调用太频繁不合法

    res = json.dumps(generated_reasons, indent=4, ensure_ascii=False)

    with open(store_path, 'a') as file:  # 将生成的训练数据保存起来
        file.write(res)


if __name__ == "__main__":
    llm_api_rec_reason()
    openllm_rec_reason()
