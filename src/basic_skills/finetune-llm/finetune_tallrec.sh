
python finetune_tallrec.py \
    --base_model '/Users/liuqiang/Desktop/code/llm/models/chinese-alpaca-2-7b' \
    --data_path '/Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/finetune-llm/data/mind/train.json' \
    --output_dir './lora-weights' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --cutoff_len 512 \
    --val_set_size 10000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length