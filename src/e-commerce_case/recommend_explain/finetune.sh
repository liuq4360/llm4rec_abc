python prompt_learning.py \
--data_path ./data/explain.json \
--dict_path ./data/finetune_dict.json \
--epochs 5 \
--lr 1e-3 \
--batch_size 32 \
--log_interval 10 \
--output_path ./data/generated.json \
--words 20 \
--checkpoint ./models/ >> ./logs/finetune.log