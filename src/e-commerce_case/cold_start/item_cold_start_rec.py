import json
import os
import time

import torch
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.utils import get_metadata_dict, get_user_history, get_cold_start_items

load_dotenv()  # https://vault.dotenv.org/ui/ui1

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

instruction = ("You are a product expert who predicts which products "
               "users prefer based on your professional knowledge.")


def formatting_prompt(history, candidate):
    prompt = f"""The user purchased the following beauty products(in JSON format): 

{history}

Predict if the user will prefer to purchase the following beauty candidate list(in JSON format):

{candidate} 

You can choice none, one or more, your output must be JSON format, you just need output item_id, the following is an
output example, A and B is product item_id.

["A", "B"]

Your output must in the candidate list, don't explain.
"""

    return prompt


def llm_api_cold_start_rec(store_path: str = 'data/llm_api_rec.json'):
    item_dict = get_metadata_dict()
    train_user_dict = get_user_history(data_type="train")
    test_user_dict = get_user_history(data_type="test")
    common_users = set(train_user_dict.keys()).intersection(set(test_user_dict.keys()))
    cold_start_items = get_cold_start_items()

    generated_rec = []
    print("total user number = " + str(len(common_users)))
    i = 0
    for user in common_users:
        H = []
        for h in train_user_dict[user]:
            info = item_dict[h]
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            H.append(info)
        history = json.dumps(H, indent=4, ensure_ascii=False)
        C = []
        for item in cold_start_items:
            info = item_dict[item]
            info['item_id'] = item
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            C.append(info)
        candidate = json.dumps(C, indent=4, ensure_ascii=False)

        prom = formatting_prompt(history, candidate)
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
        content = llm_response.choices[0].message.content.strip()
        rec = {
            "user": user,
            "rec": content
        }
        i += 1
        print("-------------- " + str(i) + " -----------------")
        print(json.dumps(rec, indent=4, ensure_ascii=False))
        generated_rec.append(rec)
        if i % 7 == 0:
            time.sleep(1)  # 避免moonshot认为调用太频繁不合法

    res = json.dumps(generated_rec, indent=4, ensure_ascii=False)

    with open(store_path, 'a') as file:  # 将生成的训练数据保存起来
        file.write(res)


def openllm_cold_start_rec(model_path: str = '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-4B',
                           store_path: str = 'data/openllm_rec.json'):
    item_dict = get_metadata_dict()
    train_user_dict = get_user_history(data_type="train")
    test_user_dict = get_user_history(data_type="test")
    common_users = set(train_user_dict.keys()).intersection(set(test_user_dict.keys()))
    cold_start_items = get_cold_start_items()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    generated_rec = []
    print("total user number = " + str(len(common_users)))
    i = 0
    for user in common_users:
        H = []
        for h in train_user_dict[user]:
            info = item_dict[h]
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            H.append(info)
        history = json.dumps(H, indent=4, ensure_ascii=False)
        C = []
        for item in cold_start_items:
            info = item_dict[item]
            info['item_id'] = item
            if 'description' in info:
                del info['description']  # description 字段太长了，消耗的token太多，剔除掉
            C.append(info)
        candidate = json.dumps(C, indent=4, ensure_ascii=False)

        input = formatting_prompt(history, candidate)

        prompt = f"""### Instruction:
                    {instruction}

                    ### Input:
                    {input}

                    ### Response:
                    """

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
        outputs = model.generate(input_ids=input_ids.to('mps'),
                                 max_new_tokens=1500, pad_token_id=tokenizer.eos_token_id)
        predict_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

        rec = {
            "user": user,
            "rec": predict_output
        }
        i += 1
        print("-------------- " + str(i) + " -----------------")
        print(json.dumps(rec, indent=4, ensure_ascii=False))
        generated_rec.append(rec)
        if i % 7 == 0:
            time.sleep(1)  # 避免moonshot认为调用太频繁不合法

    res = json.dumps(generated_rec, indent=4, ensure_ascii=False)

    with open(store_path, 'a') as file:  # 将生成的训练数据保存起来
        file.write(res)


if __name__ == "__main__":
    llm_api_cold_start_rec()
    openllm_cold_start_rec(model_path='./models', store_path='data/openllm_finetune_rec.json')
    openllm_cold_start_rec()
