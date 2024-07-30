import json
import math

import fire
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def rmse(_true, _predict):
    return math.sqrt(math.fabs(_true - _predict))


def output_format(output) -> float:
    """
    :param output: 大模型的输出
    :return: int
    没有微调过的大模型输出的可能不是按照规范的，需要获得大模型对应的输出，下面是4个大模型的输出案例。
    1.  The similarity between these two products is 0.5.
    2. The similarity between the two products is 0.5. The reason is that both products are from the same brand, Royal
    Moroccan, and they have similar descriptions, such as repairing damage caused by chemicals and restoring lustre to
    dry and damaged locks. However, the price and capacity of the two products are different, which may affect the similarity score.
    3. 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    4.  [
            {
                "product\_1": "Shiseido Pureness Moisturizing Gel (Oil Free) 50ml/1.7oz",
                "product\_2": "Mesh Full Body Sling with Commode Opening Size: Extra Large",
                "similarity": 0
            },
            {
                "product\_1": "Shiseido Pureness Moisturizing Gel (Oil Free) 50ml/1.7oz",
                "product\_2": "Mesh Full Body Sling with Commode Opening Size: Large",
                "similarity": 0
            }
        ]
    """
    output = output[:1000]  # 只取前面的1000个字符，后面如果有生成额外的不考虑
    if is_float(output[0:2]):
        return float(output[0:2])
    string_1 = 'The similarity between the two products is '
    string_2 = 'The similarity between these two products is '
    string_3 = '"similarity": '
    index_1 = output.find(string_1)  # 如果找不到，返回-1
    index_2 = output.find(string_2)  # 如果找不到，返回-1
    index_3 = output.find(string_3)  # 如果找不到，返回-1
    if index_1 > -1:
        if is_float(output[index_1 + len(string_1):index_1 + len(string_1) + 2]):
            score = float(output[index_1 + len(string_1):index_1 + len(string_1) + 2])
            return score
        else:
            return -1
    elif index_2 > -1:
        if is_float(output[index_2 + len(string_2):index_2 + len(string_2) + 2]):
            score = float(output[index_2 + len(string_2):index_2 + len(string_2) + 2])
            return score
        else:
            return -1
    elif index_3 > -1:
        if is_float(output[index_3 + len(string_3):index_3 + len(string_3) + 2]):
            score = float(output[index_3 + len(string_3):index_3 + len(string_3) + 2])
            return score
        else:
            return -1
    else:
        return -1


def load_model_token(model_path: str) -> (AutoModelForCausalLM, AutoTokenizer):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'  # to prevent warnings
    return model, tokenizer


def evaluate(model_path: str,
             test_data_path: str = './data',
             keep_sample_num: int = 10) -> float:
    model, tokenizer = load_model_token(model_path)

    dataset_dict = load_dataset(test_data_path)
    test_dataset = dataset_dict['test'][0:keep_sample_num]

    acc_rmse = 0.0  # 累积误差
    acc_num = 0  # 累积的参与统计的样本数量

    for i in range(keep_sample_num):
        prompt = f"""### Instruction:
        {test_dataset['instruction'][i]}
        
        ### Input:
        {test_dataset['input'][i]}
        
        ### Response:
        """
        gold_output = float(test_dataset['output'][i])
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
        outputs = model.generate(input_ids=input_ids.to('mps'),
                                 max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        predict_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
        print("--------sample: " + str(i) + "--------")
        print("预测的输出: " + predict_output + "\n")
        score = output_format(predict_output)
        print("预测的score: " + str(score) + "\n")
        if score == -1:  # 生成的比较复杂，没有解析到对应的预测评分
            continue
        else:
            acc_num += 1
            predict_output = score
            rmse_ = rmse(gold_output, predict_output)
            acc_rmse += rmse_
            dic = {  # 将每一个样本的评估结果打印出来
                "sample": i,
                "input": test_dataset['input'][i],
                "gold_output": gold_output,
                "predict_output": predict_output,
                "rmse": rmse_
            }
            print(json.dumps(dic, indent=4, ensure_ascii=False))
            print("----------------")

    return acc_rmse / acc_num


def effect_comparison(base_model_path: str = '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-4B',
                      finetune_model_path: str = './models',
                      test_data_path: str = './data',
                      keep_sample_num: int = 10):
    avg_base_rmse = evaluate(base_model_path, test_data_path, keep_sample_num)
    avg_finetune_rmse = evaluate(finetune_model_path, test_data_path, keep_sample_num)

    print("基底模型的平均rmse：" + str(avg_base_rmse))
    print("微调模型的平均rmse：" + str(avg_finetune_rmse))


if __name__ == "__main__":
    fire.Fire(effect_comparison)
