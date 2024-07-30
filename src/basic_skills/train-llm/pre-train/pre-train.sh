#!/bin/bash

dir_path="../logs/mind/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi

PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun t5_pre-train.py --item_indexing sequential \
--task sequential,straightforward --dataset mind --epochs 1 --batch_size 1024 \
--backbone t5-small --cutoff 1024 --data_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/data \
--valid_prompt seen:0 --model_dir /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/models \
--lr 1e-3 --valid_select 1 --warmup_steps 100 --gradient_accumulation_steps 10 --logging_steps 10 --optim 'adamw_torch' \
--eval_steps 200 --save_steps 200 --save_total_limit 3


