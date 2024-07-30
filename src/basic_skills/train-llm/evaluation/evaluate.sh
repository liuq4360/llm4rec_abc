
python t5_evaluate.py --dataset mind --task sequential,straightforward --item_indexing sequential --backbone t5-small \
--checkpoint_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/models/mind/sequential/t5-small/ \
 --test_prompt seen:0 --log_dir '../logs' \
--data_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/data \
--cutoff 1024 --eval_batch_size 32 --metrics hit@5,hit@10,ndcg@5,ndcg@10


python t5_evaluate.py --dataset mind --task sequential,straightforward --item_indexing sequential --backbone t5-small \
--checkpoint_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/models/mind/sequential/t5-small/ \
--test_prompt unseen:0 --log_dir '../logs' \
--data_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/basic_skills/train-llm/data \
--cutoff 1024 --eval_batch_size 32 --metrics hit@5,hit@10,ndcg@5,ndcg@10
