import os
import sys

import fire
import gradio as gr
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


def main(
        load_8bit: bool = False,
        finetune_model: str = "./models",
        # lora_weights: str = "./lora-weights",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        server_name: str = "localhost",  # Allows to listen on all interfaces by providing '0.
        share_gradio: bool = False,
):
    finetune_model = finetune_model or os.environ.get("FINETUNE_MODEL", "")
    assert (
        finetune_model
    ), "Please specify a --finetune_model, e.g. --finetune_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(finetune_model)

    device_map = "mps"

    model = AutoModelForCausalLM.from_pretrained(
        finetune_model,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    # model = PeftModel.from_pretrained(
    #     model,
    #     lora_weights,
    #     device_map=device_map,
    #     torch_dtype=torch.float16,
    # )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device_map)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="LLM-based personalized recommendation",
        description="通过预测用户的评分，来评估用户对商品的偏好，进而为用户生成个性化推荐",  # noqa: E501
    ).queue().launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
