import json
import sys
import importlib
from sentence_transformers import CrossEncoder
sys.path.append('./data-process')
generate_item_dict = importlib.import_module('generate_item_dict')


def rerank_recall(item_id: str,
                  recall_list: [[str]],
                  item_dict: dict,
                  top_n: int = 10) -> [dict]:
    """

    :param item_id: 待推荐的商品id，我们会给该商品关联相关的商品做为相似推荐
    :param recall_list: 召回的商品列表
    :param item_dict: 商品信息字典
    :param top_n: 最终相似的商品的数量
    :return: 最终排序后的相似结果，默认值为10
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


if __name__ == "__main__":
    embedding_recall = ['B000052YPP', 'B00005308B', 'B0000530HZ', 'B000052YD8', 'B00005308M', 'B000052YMO',
                        '9790787006', '6546546450', '9744914572', '7414204790']
    also_view_recall = ['B000Z96JDI', 'B001FB5HZG', 'B00213WCNC', 'B00BBFOVO4', 'B0085EVLRO']
    also_buy_recall = ['B000Z96JDI', 'B00006IGL8', 'B007C5X34G', 'B00006IGLF', 'B00213WCNC', 'B00D1W1QXE', 'B001FB5HZG',
                       'B000FQ86RI', 'B0012BSKBM', 'B0085EVLRO', 'B00A2EXVQE']
    band_recall = ['B00028EYZW', 'B001E0T0HE', 'B00FTBJ6HI', 'B00KLDU08S', 'B00OQQWU4I']
    item_dict = generate_item_dict.get_metadata_dict()
    print("----------")
    res = rerank_recall("B00006IGL2",
                        [embedding_recall, also_buy_recall, also_view_recall,band_recall],
                        item_dict, 10)
    print(json.dumps(res, indent=4, ensure_ascii=False))
    print("----------")
