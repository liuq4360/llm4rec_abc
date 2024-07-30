import inspect
import json
import random
import re
from typing import *
import numpy as np
import tiktoken


def get_topk_index(scores, topk, return_score: bool = False):
    r"""Get topk index given scores with numpy. The returned index is sorted by scores descendingly."""
    scores = -scores

    # By performing a partition with (topk-1), the (topk-1)-th element will be in its final sorted position and all
    # smaller elements will be moved before it. It means elements with indices from 0 to (topk-1) are the topk
    # elements. And if topk=len(scores), the function would raise an error due to it could not get the k-th element
    # for an array of length k.
    topk_ind = np.argpartition(scores, topk - 1, axis=-1)[:, :topk]
    topk_scores = np.take_along_axis(scores, topk_ind, axis=-1)
    sorted_ind_index = np.argsort(topk_scores, axis=-1)
    sorted_index = np.take_along_axis(topk_ind, sorted_ind_index, axis=-1)
    if return_score:
        return sorted_index, np.take_along_axis(-topk_scores, sorted_ind_index, axis=-1)
    return sorted_index


def normalize_np(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _sum = a.sum()
    return a / (_sum + eps)


def num_tokens_from_string(string: str, encoding_name: str = "davinci") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def extract_integers_from_string(input_string: str) -> List[int]:
    integers = re.findall(r"\d+", input_string)
    return [int(num) for num in integers]


def cut_list(input_list: List, max_token_limit: int = 512) -> List:
    if num_tokens_from_string(str(input_list)) <= max_token_limit:
        return input_list
    else:
        if len(input_list) > 1:
            res = input_list
            while num_tokens_from_string(str(res)) > max_token_limit:
                if len(res) <= 1:
                    res = cut_list(res, max_token_limit)
                res = random.sample(res, k=len(res) - 1)
            return res
        else:
            ele = str(input_list[0])
            _len = int(len(ele) * max_token_limit / num_tokens_from_string(ele))
            if _len >= len(ele):
                cut_ele = ele[:-1]
            else:
                cut_ele = ele[:_len]
            input_list[0] = cut_ele
            return cut_list(input_list, max_token_limit)


class FuncToolWrapper:
    def __init__(self, func: Callable, name: str, desc: str) -> None:
        self.func = func
        self.name = name
        self.desc = desc

    def run(self, inputs: str) -> str:
        if len(inspect.signature(self.func).parameters) > 0:
            return self.func(inputs)
        else:
            return self.func()


def replace_substrings(s, replacements):
    for key, value in replacements.items():
        s = s.replace(key, value)
    return s


def replace_substrings_regex(s: str, replace_dict: Dict):
    pattern = re.compile("|".join(re.escape(key) for key in replace_dict.keys()))

    def replacer(match):
        return replace_dict[match.group(0)]

    return pattern.sub(replacer, s)


def read_jsonl(fpath: str) -> List[Dict]:
    res = []
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


def format_prompt(args_dict, prompt):
    for k, v in args_dict.items():
        prompt = prompt.replace(f"{{{k}}}", str(v))
    return prompt


__all__ = [
    "get_topk_index",
    "normalize_np",
    "num_tokens_from_string",
    "cut_list",
    "extract_integers_from_string",
    "FuncToolWrapper",
    "replace_substrings",
    "replace_substrings_regex",
    "read_jsonl",
    "format_prompt"
]
