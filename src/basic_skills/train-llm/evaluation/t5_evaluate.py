import logging
import os
import sys

import fire
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
)

sys.path.append('../')
from utils import EvaluationDataset, evaluation_results, get_metrics_results


def main(log_dir: str, checkpoint_path: str, data_path: str, item_indexing: str, task: str,
         dataset: str, cutoff: int, test_prompt: str, eval_batch_size: int, metrics: str):
    # setup
    log_file = os.path.join(log_dir, dataset,
                            checkpoint_path.replace('.', '').replace('/', '_') + '.log')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    # load test data
    test_data_file = os.path.join(data_path, dataset,
                                  f'{dataset}_{task}_{item_indexing}_test_{test_prompt}.json')
    logging.info("test_data_file=" + test_data_file)
    test_data = load_dataset("json", data_files=test_data_file, field='data')
    model.eval()
    metrics = list(metrics)
    generate_num = max([int(m.split('@')[1]) for m in metrics])
    task_list = np.unique(test_data['train']['task'])
    for t in task_list:
        logging.info(f'testing on {t}')
        subset_data = test_data.filter(lambda example: example['task'] == t)
        dataset = EvaluationDataset(subset_data['train'], tokenizer, cutoff)
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        test_total = 0
        metrics_res = np.array([0.0] * len(metrics))
        for batch in tqdm(dataloader):
            """
            下面是一个batch的案例：
                {'input_ids': tensor([[    3, 21419, 12587,  ...,     0,     0,     0],
                ...,
                [    3, 21419, 12587,  ...,     0,     0,     0]]), 
                
                'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
                ...,
                [1, 1, 1,  ..., 0, 0, 0]]), 
                
                'label': tensor([[12587,  2118,   834, 22504,  2577,     1,     0],
                [12587,  2118,   834, 19993,  4867,     1,     0],
                ...,
                [12587,  2118,   834, 19993,  5062,     1,     0]])}
            """

            prediction = model.generate(  # 大模型模型生成函数
                input_ids=batch["input_ids"],  # torch.LongTensor of shape (batch_size, sequence_length)
                attention_mask=batch["attention_mask"],  # torch.FloatTensor of shape (batch_size, sequence_length)
                max_length=30,
                num_beams=generate_num,
                num_return_sequences=generate_num,
                output_scores=True,
                return_dict_in_generate=True,
            )
            output_ids = batch['label']
            prediction_ids = prediction["sequences"]  # 利用大模型进行预测，输出的向量化的，需要解码
            prediction_scores = prediction["sequences_scores"]
            gold_sents = tokenizer.batch_decode(  # 用户真实的点击记录
                output_ids, skip_special_tokens=True
            )
            generated_sents = tokenizer.batch_decode(  # 大模型预测的点击记录
                prediction_ids, skip_special_tokens=True
            )
            rel_results = evaluation_results(generated_sents, gold_sents, prediction_scores, generate_num)
            test_total += len(rel_results)
            metrics_res += get_metrics_results(rel_results, metrics)

        metrics_res /= test_total
        for i in range(len(metrics)):
            logging.info(f'{metrics[i]}: {metrics_res[i]}')


if __name__ == "__main__":
    fire.Fire(main)
