import importlib
import operator
import sys

from sentence_transformers import SentenceTransformer

sys.path.append('./data-process')
generate_item_dict = importlib.import_module('generate_item_dict')


def tags_recall(item_id: str, item_dict: dict) -> [str]:
    """
    基于商品的标签召回，本算法利用brand进行召回，召回的item是brand跟item_id一样的商品
    :param item_dict: 商品metadata的字典信息
    :param item_id: 商品id
    :return: 召回的item列表
    """
    brand = item_dict[item_id]['brand']
    recall_list = []
    for key, value in item_dict.items():
        if value['brand'] == brand and key != item_id:
            recall_list.append(key)
    return recall_list


def embedding_recall(item_id: str,
                     item_dict: dict,
                     recall_num: int = 20,
                     min_similar_score: float = 0.8) -> [str]:
    """
    利用商品的文本数据进行嵌入，利用嵌入向量召回
    :param min_similar_score: 最底的相似得分，大于这个得分就可以做为召回了
    :param recall_num: 召回的数量，默认是20个
    :param item_dict: 商品metadata的字典信息
    :param item_id: 商品id
    :return: 召回的item列表
    本函数只是一个方法式例，的实现效率不是很高，更好的实现方式是提前将所有商品的embedding计算出来并且放到faiss库（或者其它向量库）中，
    这样可以获得毫秒级的召回效率
    """
    model = SentenceTransformer('/Users/liuqiang/Desktop/code/llm/models/bge-large-en-v1.5')
    item_title = item_dict[item_id]['title']
    item_desc = item_dict[item_id]['description'][0]
    item_info = "title: " + item_title + "\n" + "description: " + item_desc
    sentences_1 = [item_info]
    embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    similar_list = []
    for key, value in item_dict.items():
        if len(similar_list) < recall_num and key != item_id and value['description']:
            title = value['title']
            desc = value['description'][0]
            info = "title: " + title + "\n" + "description: " + desc
            sentences_2 = [info]
            embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
            similarity = embeddings_1 @ embeddings_2.T
            if similarity[0][0] > min_similar_score:
                similar_list.append((key, similarity[0][0]))
    similar_list.sort(key=operator.itemgetter(1), reverse=True)
    slice_list = similar_list[0: recall_num]
    return [x[0] for x in slice_list]


def also_buy_recall(item_id: str, item_dict: dict) -> [str]:
    """
    亚马逊电商数据集中商品metadata中包含also_buy字段，这个字段就是跟该商品一起买的商品，可以做为召回
    :param item_dict: 商品metadata的字典信息
    :param item_id: 商品id
    :return: 召回的item列表
    """
    also_buy_list = item_dict[item_id]['also_buy']
    return also_buy_list


def also_view_recall(item_id: str, item_dict: dict) -> [str]:
    """
    亚马逊电商数据集中商品metadata中包含also_view字段，这个字段就是跟该商品一起被用户浏览的商品，可以做为召回来源
    :param item_dict: 商品metadata的字典信息
    :param item_id: 商品id
    :return: 召回的item列表
    """
    also_view_list = item_dict[item_id]['also_view']
    return also_view_list


if __name__ == "__main__":
    dic = generate_item_dict.get_metadata_dict()
    print("----------")
    print(embedding_recall("B00006IGL2", dic, 10, 0.75))
    print("----------")
