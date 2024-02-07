
python generate_dataset_train.py --data_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/data/ \
 --item_indexing  sequential --task sequential,straightforward --dataset mind --prompt_file ../prompt.txt \
 --sequential_order original --max_his 10 --his_sep ' , ' --his_prefix 1 --skip_empty_his 1


python generate_dataset_eval.py --dataset mind --data_path /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/data/ \
 --item_indexing  sequential --task sequential,straightforward --prompt_file /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/prompt.txt \
 --mode validation --prompt seen:0 --sequential_order original --max_his 10 --his_sep ' , ' --his_prefix 1 --skip_empty_his 1

python generate_dataset_eval.py --dataset mind --data_path  /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/data/ \
 --item_indexing  sequential --task sequential,straightforward --prompt_file /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/prompt.txt \
 --mode test --prompt seen:0 --sequential_order original --max_his 10 --his_sep ' , ' --his_prefix 1 --skip_empty_his 1

python generate_dataset_eval.py --dataset mind --data_path  /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/data/ \
 --item_indexing  sequential --task sequential,straightforward --prompt_file /Users/liuqiang/Desktop/code/llm4rec/llm4rec_abc/src/train-llm/prompt.txt \
 --mode test --prompt unseen:0 --sequential_order original --max_his 10 --his_sep ' , ' --his_prefix 1 --skip_empty_his 1


