import importlib
import json
import operator
import sys

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('./data-process')
generate_item_dict = importlib.import_module('generate_item_dict')


def cross_encoder_rerank(item_id: str,
                         recall_list: [[str]],
                         item_dict: dict,
                         top_n: int = 10) -> [dict]:
    """

    :param item_id: 待推荐的商品id，我们会给该商品关联相关的商品做为相似推荐
    :param recall_list: 召回的商品列表
    :param item_dict: 商品信息字典
    :param top_n: 最终相似的商品的数量，默认值为10
    :return: 最终排序后的相似结果
    """
    all_recall_items = set()
    for lst in recall_list:
        all_recall_items = all_recall_items.union(set(lst))
    model = CrossEncoder(model_name='/Users/liuqiang/Desktop/code/llm/models/bge-reranker-large',
                         max_length=512, device="mps")

    item_title = item_dict[item_id]['title']
    item_desc = item_dict[item_id]['description'][0]
    item_info = "title: " + item_title + "\n" + "description: " + item_desc
    sentence_list = []
    item_list = []
    for item in all_recall_items:
        if item in item_dict:
            title = item_dict[item]['title']
            desc = item_dict[item]['description'][0]
            info = "title: " + title + "\n" + "description: " + desc
            sentence_list.append(info)
            item_list.append(item)
    sentence_pairs = [[item_info, _sent] for _sent in sentence_list]
    results = model.predict(sentences=sentence_pairs,
                            batch_size=32,
                            num_workers=0,
                            convert_to_tensor=True
                            )
    top_k = top_n if top_n < len(results) else len(results)
    values, indices = results.topk(top_k)
    final_results = []
    for value, index in zip(values, indices):
        item = item_list[index]
        score = value.item()
        doc = {
            "item": item,
            "score": score
        }
        final_results.append(doc)
    return final_results


def llm_rerank(item_id: str,
               recall_list: [[str]],
               item_dict: dict,
               top_n: int = 10,
               model_path: str = './models') -> [dict]:
    """

    :param model_path: 预训练好的模型的存储路径
    :param item_id: 待推荐的商品id，我们会给该商品关联相关的商品做为相似推荐
    :param recall_list: 召回的商品列表
    :param item_dict: 商品信息字典
    :param top_n: 最终相似的商品的数量，默认值为10
    :return: 最终排序后的相似结果
    """
    print(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    instruction = ("You are a product expert who judges whether "
                   "two products are similar based on your professional knowledge.")

    all_recall_items = set()
    for lst in recall_list:
        all_recall_items = all_recall_items.union(set(lst))

    a_dict = {
        "title": item_dict[item_id]['title'],
        "brand": item_dict[item_id]['brand'],
        "price": item_dict[item_id]['price'],
        "description": item_dict[item_id]['description']
    }

    results = []
    for item in all_recall_items:
        if item in item_dict:
            b_dict = {
                "title": item_dict[item]['title'],
                "brand": item_dict[item]['brand'],
                "price": item_dict[item]['price'],
                "description": item_dict[item]['description']
            }
            sample_pair = [a_dict, b_dict]
            formatted_input = json.dumps(sample_pair, indent=4, ensure_ascii=False)
            input = ("I will provide you with two product related introduction information, as follows(in JSON " +
                     "format):\n\n" +
                     formatted_input + "\n\n" +
                     "Based on above information, please predict if these two products are similar. The similarity " +
                     "is between 0 and 1, 0 being lowest and 1 being highest. You just need to ranking the above " +
                     "product, do not explain the reason.")
            prompt = f"""### Instruction:
            {instruction}

            ### Input:
            {input}

            ### Response:
            """
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
            outputs = model.generate(input_ids=input_ids.to('mps'),
                                     max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
            predict_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
            doc = {
                "item": item,
                "score": float(predict_output)
            }
            results.append(doc)

    sorted_list = sorted(results, key=operator.itemgetter('score'), reverse=True)
    return sorted_list[:top_n]


if __name__ == "__main__":
    embedding_recall = ['B000052YPP', 'B00005308B', 'B0000530HZ', 'B000052YD8', 'B00005308M', 'B000052YMO',
                        '9790787006', '6546546450', '9744914572', '7414204790']
    also_view_recall = ['B000Z96JDI', 'B001FB5HZG', 'B00213WCNC', 'B00BBFOVO4', 'B0085EVLRO']
    also_buy_recall = ['B000Z96JDI', 'B00006IGL8', 'B007C5X34G', 'B00006IGLF', 'B00213WCNC', 'B00D1W1QXE', 'B001FB5HZG',
                       'B000FQ86RI', 'B0012BSKBM', 'B0085EVLRO', 'B00A2EXVQE']
    band_recall = ['B00028EYZW', 'B001E0T0HE', 'B00FTBJ6HI', 'B00KLDU08S', 'B00OQQWU4I']
    dic = generate_item_dict.get_metadata_dict('../data/amazon_review/beauty/meta_All_Beauty.json')

    # print("----------")
    # res = cross_encoder_rerank("B00006IGL2",
    #                            [embedding_recall, also_buy_recall, also_view_recall, band_recall],
    #                            item_dict, 10)
    # print(json.dumps(res, indent=4, ensure_ascii=False))
    # print("----------")

    print("----------")
    res = llm_rerank("B00006IGL2",
                     [embedding_recall, also_buy_recall, also_view_recall, band_recall],
                     dic, 10, './models')
    print(json.dumps(res, indent=4, ensure_ascii=False))
    print("----------")
