import requests
import torch
import fire
import json
from langchain.chains import LLMChain
from mlx_lm import load, generate
from langchain_community.llms import LlamaCpp
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def icl_rec(
        model_path: str = "",
        icl_type: str = "",
        prompt: str = "",
        temperature: float = 0.1,
        top_p: float = 0.95,
        ctx: int = 13000,
):
    if not prompt:
        prompt = """
        I’ve watched the following movies in the past in order:
    
        ['0. Mirror, The (Zerkalo)', '1. The 39 Steps', '2. Sanjuro', '3. Trouble in Paradise']
    
        Then if I ask you to recommend a new movie to me according to my watching history, you should recommend Shampoo and
         now that I've just watched Shampoo, there are 20 candidate movies that I can watch next:
        ['0. Manon of the Spring (Manon des sources)', '1. Air Bud', '2. Citizen Kane', '3. Grand Hotel',
        '4. A Very Brady Sequel', '5. 42 Up', '6. Message to Love: The Isle of Wight Festival', '7. Screamers',
        '8. The Believers', '9. Hamlet', '10. Cliffhanger', '11. Three Wishes', '12. Nekromantik', '13. Dangerous Minds',
        '14. The Prophecy', '15. Howling II: Your Sister Is a Werewolf', '16. World of Apu, The (Apur Sansar)',
        '17. The Breakfast Club', '18. Hoop Dreams', '19. Eddie']
    
        Please rank these 20 movies by measuring the possibilities that I would like to watch next most, according to my
        watching history. Please think step by step.
    
        Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given 
        candidate movies. You can not generate movies that are not in the given candidate list.
    
        """

    assert (
        icl_type
    ), "Please specify a --icl_type, e.g. --icl_type='ollama'"

    if icl_type == "ollama":
        """
        利用Ollama框架将大模型封装成类似ChatGPT接口规范的API服务，直接通过调用接口来实现大模型icl推荐
        """
        url = "http://localhost:11434/api/chat"  # Ollama的api地址
        data = {
            "model": "yi:34b-chat",  # Ollama安装的模型名
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": ctx,
                "num_gpu": 128,
            },
            "messages": [
                    {
                      "role": "user",
                      "content": prompt
                    }
                  ]
        }
        response = requests.post(url=url, json=data, stream=True)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            j = json.loads(chunk.decode('utf-8'))
            print(j['message']['content'], end="")

    elif icl_type == "llamacpp":
        if not model_path:
            model_path = "/Users/liuqiang/Desktop/code/llm/models/gguf/qwen1.5-72b-chat-q5_k_m.gguf"
        callback = StreamingStdOutCallbackHandler()
        n_gpu_layers = 128  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        llm = LlamaCpp(
            streaming=True,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            temperature=temperature,
            top_p=top_p,
            n_ctx=ctx,
            callbacks=[callback],
            verbose=True
        )
        system = [("system",
                   "You are a recommendation system expert who provides personalized recommendations to users based on "
                   "the background information provided."),
                  ("user", "{input}")]
        template = ChatPromptTemplate.from_messages(system)
        chain = LLMChain(prompt=template, llm=llm)
        chain.invoke({"input": prompt})

    elif icl_type == "transformers":
        if not model_path:
            model_path = "/Users/liuqiang/Desktop/code/llm/models/Yi-34B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="mps",
            use_cache=True,
        )
        model = model.to("mps")
        streamer = TextStreamer(tokenizer)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            streamer=streamer,
            max_length=13000,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            top_p=top_p,
            repetition_penalty=1.15,
            do_sample=False,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        system = [("system",
                   "You are a recommendation system expert who provides personalized recommendations to users based on "
                   "the background information provided."),
                  ("user", "{input}")]
        template = ChatPromptTemplate.from_messages(system)
        chain = LLMChain(prompt=template, llm=llm)
        chain.invoke({"input": prompt})

    elif icl_type == "mlx":  # 苹果的MLX框架
        if not model_path:
            model_path = "/Users/liuqiang/Desktop/code/llm/models/Yi-34B-Chat"
        model, tokenizer = load(model_path)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            temp=temperature,
            max_tokens=10000,
            verbose=True
        )
        print(response)

    else:
        raise NotImplementedError("the case not implemented!")


if __name__ == "__main__":
    fire.Fire(icl_rec)
