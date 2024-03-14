import json
import random
import time

from openai import OpenAI
from generate_item_dict import get_metadata_dict
from generate_user_action_history import get_user_history
from generate_cold_start_items import get_cold_start_items

item_dict = get_metadata_dict()
user_history = get_user_history()
cold_start_items = get_cold_start_items()

MOONSHOT_API_KEY = 'sk-8cjxTSVHcugQgumlsqPayHMFG8BLeOWjuaFST2YuW0W3tHv2'

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


generated_samples = []
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
        print(prom)
        print(llm_response.choices[0].message.content)
        if llm_response.choices[0].message.content == "A":
            sample = {
                "user": user,
                "item": random_2_elements[0]
            }
            generated_samples.append(sample)
        elif llm_response.choices[0].message.content == "B":
            sample = {
                "user": user,
                "item": random_2_elements[1]
            }
            generated_samples.append(sample)
            time.sleep(1)  # 避免moonshot认为调用太频繁不合法

print(json.dumps(generated_samples, indent=4, ensure_ascii=False))
