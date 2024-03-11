#PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python model_finetune.py \
    --base_model '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-4B' \
    --data_path '/Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/e-commerce_case/similar_rec/data/train.json' \
    --output_dir './models' \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --cutoff_len 512 \
    --val_set_size 200 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]' \
    --train_on_inputs \
    --group_by_length