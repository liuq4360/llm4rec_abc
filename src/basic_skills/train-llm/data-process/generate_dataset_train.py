import json
import os
import sys

import fire

sys.path.append('../')
from utils import sequential_indexing, load_prompt_template, check_task_prompt, get_info_from_prompt
from utils import construct_user_sequence_dict, read_line


def main(data_path: str, item_indexing: str, task: str, dataset: str, prompt_file: str, sequential_order: str,
         max_his: int, his_sep: str, his_prefix: int, skip_empty_his: int):
    file_data = dict()
    file_data['arguments'] = {
        "data_path": data_path, "item_indexing": item_indexing, "task": task,
        "dataset": dataset, "prompt_file": prompt_file, "sequential_order": sequential_order,
        "max_his": max_his, "his_sep": his_sep, "his_prefix": his_prefix, "skip_empty_his": skip_empty_his
    }
    file_data['data'] = []
    tasks = list(task)
    user_sequence = read_line(os.path.join(data_path, dataset, 'user_sequence.txt'))
    user_sequence_dict = construct_user_sequence_dict(user_sequence)

    reindex_user_seq_dict, item_map = sequential_indexing(data_path, dataset,
                                                          user_sequence_dict, sequential_order)

    # get prompt
    prompt = load_prompt_template(prompt_file, tasks)
    info = get_info_from_prompt(prompt)
    check_task_prompt(prompt, tasks)

    # Load training data samples
    training_data_samples = []
    for user in reindex_user_seq_dict:
        items = reindex_user_seq_dict[user][:-2]
        for i in range(len(items)):
            if i == 0:
                if skip_empty_his > 0:
                    continue
            one_sample = dict()
            one_sample['dataset'] = dataset
            one_sample['user_id'] = user
            if his_prefix > 0:
                one_sample['target'] = 'item_' + items[i]
            else:
                one_sample['target'] = items[i]
            if 'history' in info:
                history = items[:i]
                if max_his > 0:
                    history = history[-max_his:]
                if his_prefix > 0:
                    one_sample['history'] = his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = his_sep.join(history)
            training_data_samples.append(one_sample)
    print("load training data")
    print(f'there are {len(training_data_samples)} samples in training data.')

    # construct sentences
    for i in range(len(training_data_samples)):
        one_sample = training_data_samples[i]
        for t in tasks:
            datapoint = {'task': dataset + t, 'data_id': i}
            for pid in prompt[t]['seen']:
                datapoint['instruction'] = prompt[t]['seen'][pid]['Input']
                datapoint['input'] = prompt[t]['seen'][pid]['Input'].format(**one_sample)
                datapoint['output'] = prompt[t]['seen'][pid]['Output'].format(**one_sample)
                file_data['data'].append(datapoint.copy())

    print("data constructed")
    print(f"there are {len(file_data['data'])} prompts in training data.")

    # save the data to json file
    output_path = f'{dataset}_{task}_{item_indexing}_train.json'

    with open(os.path.join(data_path, dataset, output_path), 'w') as openfile:
        json.dump(file_data, openfile)


if __name__ == "__main__":
    fire.Fire(main)
