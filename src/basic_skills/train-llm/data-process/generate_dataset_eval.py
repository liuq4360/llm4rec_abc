import json
import os
import sys

import fire

sys.path.append('../')
from utils import sequential_indexing, load_prompt_template, check_task_prompt, get_info_from_prompt
from utils import construct_user_sequence_dict, read_line, load_test, load_validation


def main(data_path: str, item_indexing: str, task: str, dataset: str, prompt_file: str, sequential_order: str,
         max_his: int, his_sep: str, his_prefix: int, skip_empty_his: int,
         mode: str, prompt: str):
    file_data = dict()
    file_data['arguments'] = {
        "data_path": data_path, "item_indexing": item_indexing, "task": task,
        "dataset": dataset, "prompt_file": prompt_file, "sequential_order": sequential_order,
        "max_his": max_his, "his_sep": his_sep, "his_prefix": his_prefix, "skip_empty_his": skip_empty_his,
        "mode": mode, "prompt": prompt
    }
    file_data['data'] = []
    tasks = list(task)

    user_sequence = read_line(os.path.join(data_path, dataset, 'user_sequence.txt'))
    user_sequence_dict = construct_user_sequence_dict(user_sequence)
    reindex_user_seq_dict, item_map = sequential_indexing(data_path, dataset,
                                                          user_sequence_dict, sequential_order)

    # get prompt
    prompt_ = load_prompt_template(prompt_file, tasks)
    info = get_info_from_prompt(prompt_)
    check_task_prompt(prompt_, tasks)

    # Load data samples
    if mode == 'validation':
        data_samples = load_validation(reindex_user_seq_dict, info, dataset, his_prefix, max_his, his_sep)
        prompt_info = prompt.split(':')
        output_path = f'{dataset}_{task}_{item_indexing}_validation_{prompt}.json'
    elif mode == 'test':
        data_samples = load_test(reindex_user_seq_dict, info, dataset, his_prefix, max_his, his_sep)
        prompt_info = prompt.split(':')
        output_path = f'{dataset}_{task}_{item_indexing}_test_{prompt}.json'
    else:
        raise NotImplementedError

    # construct sentences
    for i in range(len(data_samples)):
        one_sample = data_samples[i]
        for t in tasks:
            datapoint = {'task': dataset + t,
                         'instruction': prompt_[t][prompt_info[0]][prompt_info[1]]['Input'],
                         'input': prompt_[t][prompt_info[0]][prompt_info[1]]['Input'].format(**one_sample),
                         'output': prompt_[t][prompt_info[0]][prompt_info[1]]['Output'].format(**one_sample)}
            file_data['data'].append(datapoint.copy())

    with open(os.path.join(data_path, dataset, output_path), 'w') as openfile:
        json.dump(file_data, openfile)


if __name__ == "__main__":
    fire.Fire(main)
