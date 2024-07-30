import importlib
import json
import sys

sys.path.append('utils')
utils = importlib.import_module('utils')

REC_MUM = 8


def precision(rec_list: list, action_list: list) -> float:
    """
    计算单个用户推荐的精准度
    :param rec_list: 算法的推荐列表
    :param action_list: 用户实际的购买列表
    :return: 精准度
    """
    num = len(set(rec_list))
    if num > 0:
        return len(set(rec_list).intersection(set(action_list))) / num
    else:
        return 0.0


def recall(rec_list: list, action_list: list) -> float:
    """
    计算单个用户推荐的召回率
    :param rec_list: 算法的推荐列表
    :param action_list: 用户实际的购买列表
    :return: 召回率
    """
    num = len(set(action_list))
    if num > 0:
        return len(set(rec_list).intersection(set(action_list))) / num
    else:
        return 0.0


def find_all_occurrences(s, char):
    start = s.find(char)
    indices = []
    while start != -1:
        indices.append(start)
        start = s.find(char, start + 1)
    return indices


def evaluate(data_path: str, model_type: str) -> (float, float):
    test_user_dict = utils.get_user_history(data_type="test")
    test_users = test_user_dict.keys()
    j = ""
    with open(data_path, 'r') as file:
        j = file.read()
    rec = json.loads(j)

    common_num = 0  # 推荐的用户和实际测试集的用户的交集数量
    acc_p = 0.0  # 累积精准度
    acc_r = 0.0  # 累积召回率
    for x in rec:
        user = x['user']
        if user in test_users:
            common_num += 1
            action_list = test_user_dict[user]
            temp = x['rec']
            rec_list = None
            if model_type == 'llm_api':
                rec_list = eval(temp)
            elif model_type in ['openllm', 'openllm_finetune']:  # 千问模型和微调的模型生成的结构比较复杂，需要特殊处理
                # print(temp)
                loc_list = find_all_occurrences(temp, '"item_id": "')
                rec_list = []
                for loc in loc_list:
                    item_id = temp[loc + len('"item_id": "'): loc + len('"item_id": "') + 10]  # item_id长度为10
                    rec_list.append(item_id)
                # print(rec_list)
            rec_list = rec_list[:REC_MUM]  # 最多推荐REC_MUM，避免不同模型推荐的数量不一样，对比不公平
            p = precision(rec_list, action_list)
            r = recall(rec_list, action_list)
            acc_p += p
            acc_r += r

    avg_p = acc_p / common_num
    avg_r = acc_r / common_num

    return avg_p, avg_r


if __name__ == "__main__":
    llm_api_avg_p, llm_api_avg_r = evaluate('data/llm_api_rec.json', model_type='llm_api')
    openllm_avg_p, openllm_avg_r = evaluate('data/openllm_rec.json', model_type='openllm')
    openllm_finetune_avg_p, openllm_finetune_avg_r = (
        evaluate('data/openllm_finetune_rec.json', model_type='openllm_finetune'))

    res = [
        {
            "llm_api_avg_p": llm_api_avg_p,
            "llm_api_avg_r": llm_api_avg_r
        },
        {
            "openllm_avg_p": openllm_avg_p,
            "openllm_avg_r": openllm_avg_r
        },
        {
            "openllm_finetune_avg_p": openllm_finetune_avg_p,
            "openllm_finetune_avg_r": openllm_finetune_avg_r
        }
    ]

    print(json.dumps(res, indent=4, ensure_ascii=False))
