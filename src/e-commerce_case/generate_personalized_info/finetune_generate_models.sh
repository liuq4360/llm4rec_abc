
#PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python finetune_generated_info_model.py \
    --base_model '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-7B-Chat' \
    --data_path '/Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/e-commerce_case/generate_personalized_info/data/portrait_data.csv' \
    --output_dir './models/portrait_model' \
    --test_size 0.4 \
    --per_device_train_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]' \
    -- max_seq_length 512


python finetune_generated_info_model.py \
    --base_model '/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-7B-Chat' \
    --data_path '/Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/e-commerce_case/generate_personalized_info/data/item_info_data.csv' \
    --output_dir './models/item_description_model' \
    --test_size 0.999 \
    --per_device_train_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]' \
    -- max_seq_length 512