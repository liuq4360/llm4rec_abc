import torch
import fire
import pandas as pd
from typing import List
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python finetune_generated_info_model.py


# 将训练数据转为大模型微调需要的数据格式
def formatting_func(spl):
    instruct = f"""### Instruction:
You are a product guide expert, use the input below to answer questions.

### Input:
{spl['prompt']}

### Response:
{spl['label']}
"""
    return instruct


def train(
        base_model: str = "",  # the only required argument
        data_path: str = "./data/mind/train.json",
        output_dir: str = "./models/item_info",
        test_size: float = 0.4,  # 测试集比例
        per_device_train_batch_size: int = 4,  # Batch size per GPU for training
        gradient_accumulation_steps=2,  # Number of update steps to accumulate the gradients for
        num_epochs: int = 1,
        learning_rate: float = 5e-4,  # Initial learning rate (AdamW optimizer)
        lora_r: int = 8,  # LoRA attention dimension
        lora_alpha: int = 16,  # Alpha parameter for LoRA scaling
        lora_dropout: float = 0.05,  # Dropout probability for LoRA layers
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
        max_grad_norm: int = 1,  # Maximum gradient normal (gradient clipping)
        weight_decay: float = 0.01,  # Weight decay to apply to all layers except bias/LayerNorm weights
        optim: str = "adamw_torch",  # Optimizer to use
        warmup_ratio: float = 0.05,  # Ratio of steps for a linear warmup (from 0 to learning rate)
        group_by_length: bool = True,
        # Group sequences into batches with same length, Saves memory and speeds up training considerably
        save_steps: int = 50,  # Save checkpoint every X updates steps
        save_total_limit: int = 3,  # At most save X times
        logging_steps: int = 20,  # Log every X updates steps
        fp16: bool = False,  # # Enable fp16/bf16 training (set bf16 to True with an A100)
        bf16: bool = False,
        report_to: List[str] = ["wandb"],
        max_seq_length: int = 512,  # 越大占用内存越多,  Maximum sequence length to use
        packing: bool = True  # Pack multiple short examples in the same input sequence to increase efficiency
):
    df = pd.read_csv(data_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print("训练的样本量：" + str(train_df.shape[0]))
    train_dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
    })

    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 为训练准备好模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        report_to=report_to,
    )

    # 创建trainer对象
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_dict['train'],
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    # 启动训练
    trainer.train()

    # 保存模型
    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
