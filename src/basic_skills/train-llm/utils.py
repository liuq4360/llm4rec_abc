import math
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


def get_dict_from_lines(lines):
    """
    Used to get user or item map from lines loaded from txt file.
    """
    index_map = dict()
    for line in lines:
        info = line.split(" ")
        index_map[info[0]] = info[1]
    return index_map


def read_line(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def write_dict_2_file(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')


class EvaluationDataset(Dataset):
    def __init__(self, dataset, tokenizer, cutoff):
        super().__init__()
        self.input = tokenizer(
            dataset['input'], padding="longest", truncation=True, max_length=cutoff
        )
        self.output = tokenizer(
            dataset['output'], padding="longest", truncation=True, max_length=cutoff
        )

    def __len__(self):
        return len(self.input["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input["input_ids"][index]),
            "attention_mask": torch.tensor(self.input["attention_mask"][index]),
            'label': torch.tensor(self.output["input_ids"][index])
        }


def load_prompt_template(path, task_list):
    """
    Load prompt template from the file. Keep training tasks only.
    Input:
    - path: The path for prompt template txt file.
    - task_list: A list of required tasks.
    Return:
    - prompt_templates: a dictionary of prompt templates. e.g., {task: {'seen': {'0': {'Input': template_input, 'Output': template_output}}}}

    """

    if not os.path.exists(path):
        raise FileNotFoundError
    prompt_info = read_line(path)
    prompt_templates = dict()
    for prompt in prompt_info:
        t = [sens.strip() for sens in prompt.split(';')]
        if t[0] not in task_list:
            continue
        if t[0] not in prompt_templates:
            prompt_templates[t[0]] = dict()
        if t[1] not in prompt_templates[t[0]]:
            prompt_templates[t[0]][t[1]] = dict()
        num = len(prompt_templates[t[0]][t[1]])
        prompt_templates[t[0]][t[1]][str(num)] = dict()
        prompt_templates[t[0]][t[1]][str(num)]['Input'] = t[2]
        prompt_templates[t[0]][t[1]][str(num)]['Output'] = t[3]
    return prompt_templates


def generate_user_map(user_sequence_dict):
    """
    generate user map based on user sequence dict.
    """
    user_map = dict()
    for user in user_sequence_dict.keys():
        user_map[user] = str(len(user_map) + 1)
    return user_map


def reindex(user_sequence_dict, user_map, item_map):
    """
    reindex the given user sequence dict by given user map and item map
    """
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        uid = user_map[user]
        items = user_sequence_dict[user]
        reindex_user_sequence_dict[uid] = [item_map[i] for i in items]

    return reindex_user_sequence_dict


def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """

    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict


def sequential_indexing(data_path, dataset, user_sequence_dict, order):
    """
    Use sequential indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'item_sequential_indexing_{order}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_sequential_indexing_{order}.txt')

    if os.path.exists(reindex_sequence_file):
        user_sequence = read_line(reindex_sequence_file)

        item_info = read_line(item_index_file)
        item_map = get_dict_from_lines(item_info)

        return construct_user_sequence_dict(user_sequence), item_map

    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = read_line(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        write_dict_2_file(user_index_file, user_map)

    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = read_line(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = dict()
        if order == 'original':
            user_list = user_sequence_dict.keys()
        elif order == 'short2long':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=False)
        elif order == 'long2short':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=True)

        for user in user_list:
            items = user_sequence_dict[user][:-2]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        for user in user_list:
            items = user_sequence_dict[user][-2:]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        write_dict_2_file(item_index_file, item_map)

    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    write_dict_2_file(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map


def get_info_from_prompt(prompt_templates):
    """
    Extract the require information from the prompt templates.
    Input:
    - prompt_templates: a dictionary of prompt templates.
    Output:
    - info: a list of required information.
    """

    info = []
    for task in prompt_templates:
        for see in prompt_templates[task]:
            for i in prompt_templates[task][see]:
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Input'])
                info += re.findall(r'\{.*?\}', prompt_templates[task][see][i]['Output'])
    info = [i[1:-1] for i in set(info)]
    return info


def check_task_prompt(prompt_templates, task_list):
    """
    Check if all tasks have prompt templates. Raise Error if training tasks have no prompt.
    Input:
    - prompt_templates: A dictionary of prompt templates.
    - task_list: A list of training tasks.
    """
    for task in task_list:
        assert task in prompt_templates, f"No prompt for {task} task"


def evaluation_results(predictions, targets, scores, k):
    results = []
    batch_length = len(targets)
    for b in range(batch_length):
        one_batch_sequence = predictions[
                             b * k: (b + 1) * k
                             ]
        one_batch_score = scores[
                          b * k: (b + 1) * k
                          ]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        gt = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == gt:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)
    return results


def ndcg_at_k(relevance, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    for row in relevance:
        rel = row[:k]
        one_ndcg = 0.0
        for i in range(len(rel)):
            one_ndcg += rel[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_at_k(relevance, k):
    correct = 0.0
    for row in relevance:
        rel = row[:k]
        if sum(rel) > 0:
            correct += 1
    return correct


def get_metrics_results(rel_results, metrics):
    res = []
    for m in metrics:
        if m.lower().startswith('hit'):
            k = int(m.split('@')[1])
            res.append(hit_at_k(rel_results, k))
        elif m.lower().startswith('ndcg'):
            k = int(m.split('@')[1])
            res.append(ndcg_at_k(rel_results, k))

    return np.array(res)


def load_test(reindex_user_seq_dict, info, dataset, his_prefix, max_his, his_sep):
    data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user]
        one_sample = dict()
        one_sample['dataset'] = dataset
        one_sample['user_id'] = user
        if his_prefix > 0:
            one_sample['target'] = 'item_' + items[-1]
        else:
            one_sample['target'] = items[-1]
        if 'history' in info:
            history = items[:-1]
            if max_his > 0:
                history = history[-max_his:]
            if his_prefix > 0:
                one_sample['history'] = his_sep.join(["item_" + item_idx for item_idx in history])
            else:
                one_sample['history'] = his_sep.join(history)
        data_samples.append(one_sample)
    return data_samples


def load_validation(reindex_user_seq_dict, info, dataset, his_prefix, max_his, his_sep):
    data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user]
        one_sample = dict()
        one_sample['dataset'] = dataset
        one_sample['user_id'] = user
        if his_prefix > 0:
            one_sample['target'] = 'item_' + items[-2]
        else:
            one_sample['target'] = items[-2]
        if 'history' in info:
            history = items[:-2]
            if max_his > 0:
                history = history[-max_his:]
            if his_prefix > 0:
                one_sample['history'] = his_sep.join(["item_" + item_idx for item_idx in history])
            else:
                one_sample['history'] = his_sep.join(history)
        data_samples.append(one_sample)
    return data_samples
