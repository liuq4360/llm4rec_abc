import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

df = pd.read_csv("./data/item_info_data.csv")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
train_dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
})

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-4B"
device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device_map
)
model.config.use_cache = True
model.config.pretraining_tp = 1
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

sample_size = 2
sample_test_data = list(test_df['prompt'])
sample_test_data = sample_test_data[:sample_size]

num_beams = 2
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map=device_map,
    num_beams=num_beams
)
sequences = pipeline(
    sample_test_data,
    max_new_tokens=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result:\n {seq[0]['generated_text']} \n")


# 创建LoRA配置
# 根据QLoRA的论文，重要的是要考虑transformer块中的所有线性层，以获得最大性能。
from peft import LoraConfig

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 8
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.05

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# 加载训练参数
# 使用TRL库中的SFTTrainer，该库提供了一个围绕transformer Trainer的封装，
# 可以使用PEFT适配器在基于指令的数据集上轻松微调模型。
from transformers import TrainingArguments

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
model_dir = "./models"
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 8
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 5
# Initial learning rate (AdamW optimizer)
learning_rate = 5e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01
# Optimizer to use
optim = "adamw_torch"
# Number of training steps (overrides num_train_epochs)
max_steps = 150
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.05
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 50
# At most save X times
save_total_limit = 3
# Log every X updates steps
logging_steps = 20
training_arguments = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=num_train_epochs,
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
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    report_to=["wandb"]
)


# 将训练数据转为大模型微调需要的数据格式
def formatting_func(example):
    text = f"{example['prompt'][0]}\n {example['goods_description'][0]}"
    return [text]


# 创建trainer对象
from trl import SFTTrainer

################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
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

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)


# 微调完成后，加载模型，验证模型效果
trained_model_path = "./models/"
model = AutoModelForCausalLM.from_pretrained(
    trained_model_path,
    torch_dtype=torch.float16,
    device_map=device_map
)
model.config.use_cache = True
model.config.pretraining_tp = 1
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(trained_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map=device_map,
    num_beams=num_beams
)
sequences = pipeline(
    sample_test_data,
    max_new_tokens=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
# 微调后的模型输出的结果
for ix, seq in enumerate(sequences):
    print(ix, seq[0]['generated_text'] + "\n")

# 真实的结果
sample_test_result = list(test_df['goods_description'])
sample_test_result = sample_test_result[:sample_size]
print(sample_test_result)


"""
让大模型基于个性化的商品关键词，用一句话生成个性化的商品描述信息
"""
sample_test_data = ["""Describe the product in one sentence, which must include keywords such as Toys,Games and ARRIS. 
                     "Please answer in English, no less than 10 words and no more than 20 words."""]
sequences = pipeline(
    sample_test_data,
    max_new_tokens=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result\n: {seq[0]['generated_text']} \n")
