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
    没有微调过的大模型输出的可能不是按照规范的，需要获得大模型对应的输出，下面是3个大模型的输出案例。
    1. 'Based on the user\'s rating history, the predicted rating for the product
    "Korean Hair Booster Complete Protein Keratin Treatment Replenisher Therapy For All Types
    Of Damaged Hair - 25ml" is 4.5.'
    2. {
            "title": "Williams Lectric Shave, 7 Ounce",
            "brand": "Williams",
            "price": "",
            "rating": 5.0
        }
        ### Explanation:
        The user's rating for the above product is 5.0. This is because the user has rated the product 5.0 in the past
         and the product is from the same brand.
    3.  5
        The ranking is between 1 and 5, 1 being lowest and 5 being highest. You just need to ranking the above product,
        do not explain the reason.
    """
    if is_float(output[0:2]):
        return float(output[0:2])
    index = output.find('}')  # 如果找不到，返回-1
    if index > 0:
        if is_valid_json(output[0:index + 1]):
            j = eval(output[0:index + 1])
            score = float(j['rating'])
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
        score = output_format(predict_output)
        if score == -1:  # 生成的比较复杂，没有解析到对应的预测评分
            print("--------sample: " + str(i) + "--------")
            print("预测的输出不对，而是: " + predict_output + "\n")
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
