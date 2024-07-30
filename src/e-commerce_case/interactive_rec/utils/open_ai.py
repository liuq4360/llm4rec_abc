import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Tuple, Union
from openai import OpenAI
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault

load_dotenv()  # https://vault.dotenv.org/ui/ui1

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

TOKEN_USAGE_VAR = ContextVar(
    "openai_token_usage",
    default={
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "OAI": 0,
    },
)


@contextmanager
def get_openai_tokens():
    TOKEN_USAGE_VAR.set({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "OAI": 0})
    yield TOKEN_USAGE_VAR
    TOKEN_USAGE_VAR.set({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0, "OAI": 0})


class OpenAICall:
    def __init__(
            self,
            model: str = "moonshot-v1-8k",
            api_key: str = MOONSHOT_API_KEY,
            api_type: str = "open_ai",
            api_base: str = "https://api.moonshot.cn/v1",
            temperature: float = 0.0,
            model_type: str = "chat_completion",
            engine: str = "moonshot-v1-8k",
            stop_words: Union[str, List[str]] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_type = api_type if api_type else "open_ai"
        self.api_base = api_base if api_base else "https://api.moonshot.cn/v1"
        self.temperature = temperature
        self.model_type = model_type
        self.engine = engine  # deployment id
        self.stop_words = stop_words
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        if self.api_type and (self.api_type not in {"open_ai", "azure"}):
            raise ValueError(
                f"Only open_ai/azure API are supported, while got {api_type}."
            )
        model_type = "chat_completion" if "chat" in model_type else model_type
        if model_type not in {"chat_completion", "completion"}:
            raise ValueError(
                f"Only chat_completion and completion types are supported, while got {model_type}"
            )

    def call(
            self,
            user_prompt: str,
            sys_prompt: str = "You are a helpful assistant.",
            max_tokens: int = 512,
            temperature: float = None
    ) -> str:
        temperature = temperature if (temperature is not None) else self.temperature
        success = False
        try:
            prompt = [
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            result = self._chat_completion(prompt, max_tokens, temperature)
            if result[0]:  # content is not None
                success = True
        except Exception as e:
            raise e

        _prev_usage = TOKEN_USAGE_VAR.get()
        _total_usage = {
            k: _prev_usage.get(k, 0) + result[1].get(k, 0)
            for k in _prev_usage.keys()
            if "token" in k
        }
        _total_usage["OAI"] = _prev_usage.get("OAI", 0) + 1
        TOKEN_USAGE_VAR.set(_total_usage)
        if not success:
            reply = "Something went wrong, please retry."
        else:
            reply = result[0]
        return reply

    def _chat_completion(self, msgs: List, max_tokens: int, temperature: float) -> Tuple[str, Dict]:
        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_type != "open_ai":
            kwargs["engine"] = self.model
        if self.stop_words:
            kwargs["stop"] = self.stop_words
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if content:
            content = content.strip()
        else:
            content = ""

        usage = resp.usage

        return content, dict(usage)


if __name__ == "__main__":
    llm = OpenAICall(
        model="moonshot-v1-8k", api_key=MOONSHOT_API_KEY, model_type="chat_completion"
    )
    prompt_msgs = "Which city is the capital of the US?"
    print("OpenAI: ", llm.call(prompt_msgs))
