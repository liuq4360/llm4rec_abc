import os
import fire
import transformers
from datasets import load_dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


def main(backbone: str, data_path: str, item_indexing: str, task: str, dataset: str,
         valid_prompt: str, cutoff: int, model_dir: str, batch_size: int, valid_select: int,
         epochs: int, lr: float, warmup_steps: int, gradient_accumulation_steps: int,
         logging_steps: int, optim: str, eval_steps: int, save_steps: int, save_total_limit: int):

    config = T5Config.from_pretrained(backbone)
    model = T5ForConditionalGeneration.from_pretrained(backbone, config=config)
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    train_data_file = os.path.join(data_path, dataset,
                                   f'{dataset}_{task}_{item_indexing}_train.json')
    valid_data_file = os.path.join(data_path, dataset,
                                   f'{dataset}_{task}_{item_indexing}_validation_{valid_prompt}.json')
    train_data = load_dataset("json", data_files=train_data_file, field='data')
    valid_data = load_dataset("json", data_files=valid_data_file, field='data')

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, truncation=True, max_length=cutoff, padding=False, return_tensors=None,
        )
        if (isinstance(result["input_ids"][-1], int) and result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        elif isinstance(result["input_ids"][-1], list) and add_eos_token:
            for i in range(len(result['input_ids'])):
                if result["input_ids"][i][-1] != tokenizer.eos_token_id and len(result["input_ids"][i]) < cutoff:
                    result["input_ids"][i].append(tokenizer.eos_token_id)
                    result["attention_mask"][i].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def process_func(datapoint):
        encoding = tokenize(datapoint['input'], add_eos_token=True)
        labels = tokenize(datapoint['output'], add_eos_token=True)
        encoding['labels'] = labels['input_ids'].copy()
        # return encoding
        return {**datapoint, **encoding}

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    train_set = train_data['train'].shuffle().map(process_func, batched=True)
    valid_set = valid_data['train'].shuffle().map(process_func, batched=True)
    output_dir = os.path.join(model_dir, dataset, item_indexing, backbone)
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=valid_set if valid_select > 0 else None,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=logging_steps,
            optim=optim,
            evaluation_strategy="steps" if valid_select > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if valid_select > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if valid_select > 0 else False,
            group_by_length=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()  # 进行模型训练
    model.save_pretrained(output_dir)  # 保存预训练好的模型
    tokenizer.save_pretrained(output_dir)  # 保存token


if __name__ == "__main__":
    fire.Fire(main)
