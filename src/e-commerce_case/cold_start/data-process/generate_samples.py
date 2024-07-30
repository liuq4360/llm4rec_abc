import importlib
import json
import os
import random
import sys
import time

from openai import OpenAI

sys.path.append('../utils')
utils = importlib.import_module('utils')

from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault

load_dotenv()  # https://vault.dotenv.org/ui/ui1

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

instruction = ("You are a product expert who predicts which of the two products "
               "users prefer based on your professional knowledge.")


def formatting_prompt(History, item_a, item_b):
    prompt = f"""The user purchased the following beauty products in JSON format: 

{History}

Predict if the user will prefer to purchase product A or B in the next.

A is:

{item_a} 

B is:

{item_b}

Your answer must be A or B, don't explain.
"""

    return prompt


def generate_cold_start_samples(store_path: str = '../data/cold_start_action_sample.json'):
    item_dict = utils.get_metadata_dict()
    user_history = utils.get_user_history(data_type="train")
    cold_start_items = utils.get_cold_start_items()

    generated_samples = []
    i = 0
    for user, history in user_history.items():
        rd = random.random()
        if rd < 0.2:  # 随机选择20%的用户
            random_2_elements = random.sample(list(cold_start_items), 2)
            H = []
            for h in history:
                info = item_dict[h]
                H.append(info)
            HH = json.dumps(H, indent=4, ensure_ascii=False)
            A = item_dict[random_2_elements[0]]
            B = item_dict[random_2_elements[1]]
            AA = json.dumps(A, indent=4, ensure_ascii=False)
            BB = json.dumps(B, indent=4, ensure_ascii=False)
            prom = formatting_prompt(HH, AA, BB)

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
            choice = llm_response.choices[0].message.content.strip()
            sample = {}
            if choice == "A":
                sample = {
                    "user": user,
                    "item": random_2_elements[0]
                }
                generated_samples.append(sample)
            if choice == "B":
                sample = {
                    "user": user,
                    "item": random_2_elements[1]
                }
            i += 1
            print("-------------- " + str(i) + " -----------------")
            print(json.dumps(sample, indent=4, ensure_ascii=False))
            generated_samples.append(sample)
            if i % 7 == 0:
                time.sleep(1)  # 避免moonshot认为调用太频繁不合法

    res = json.dumps(generated_samples, indent=4, ensure_ascii=False)

    with open(store_path, 'a') as file:  # 将生成的训练数据保存起来
        file.write(res)


if __name__ == "__main__":
    generate_cold_start_samples()
