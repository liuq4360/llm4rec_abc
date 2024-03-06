import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
让大模型基于个性化的商品关键词，用一句话生成个性化的商品描述信息
"""
test_model = "/Users/liuqiang/Desktop/code/llm/models/Qwen1.5-14B-Chat"
device_map = "auto"

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

prompt = f"""### Instruction:
You are a product guide expert, use the input below to answer questions.

### Input:
Please describe a product in a fluent and coherent sentence, which must include the keywords Toys, Games, and ARRIS of 
the product. Your output should be limited to 10 to 20 words. Please reply in English.

### Response:
"""
messages = [
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True,
                                          add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('mps'), max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print("------------")
print(response)
