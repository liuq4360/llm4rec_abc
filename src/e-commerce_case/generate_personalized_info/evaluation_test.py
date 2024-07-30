import torch
import fire
import pandas as pd
from peft import AutoPeftModelForCausalLM
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate(
        sample_size: int = 5,
        test_size: int = 0.2,
        data_path: str = "./data/item_info_data.csv",
        test_model: str = "/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-14B-Chat",
        is_peft_model: bool = False

):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print("测试数据的样本量：" + str(test_df.shape[0]))

    sample_test_data = test_df[['prompt', 'label']].head(sample_size).to_dict('records')

    device_map = "auto"
    if is_peft_model:
        # 加载基础LLM模型与分词器
        model = AutoPeftModelForCausalLM.from_pretrained(
            test_model,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(test_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(test_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
        model = AutoModelForCausalLM.from_pretrained(
            test_model,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

    for dic in sample_test_data:
        messages = [
            {"role": "user", "content": dic['prompt']}
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                                  add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('mps'), max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("------------")
        print(f"Prompt:\n{dic['prompt']}\n")
        print(f"Generated label:\n{response}\n")
        print(f"Ground truth:\n{dic['goods_description']}")

    # for sample in sample_test_data:
    #     prompt = f"""### Instruction:
    #     You are a product guide expert, use the input below to answer questions.
    #
    #     ### Input:
    #     {sample['prompt']}
    #
    #     ### Response:
    #     """
    #     input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    #     outputs = model.generate(input_ids=input_ids.to('mps'),
    #                              max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
    #     print("------------")
    #     print(f"Prompt:\n{sample['prompt']}\n")
    #     print(
    #         f"Generated goods_description:\n"
    #         f"{tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]}")
    #     print(f"Ground truth:\n{sample['goods_description']}")


if __name__ == "__main__":
    fire.Fire(evaluate)
