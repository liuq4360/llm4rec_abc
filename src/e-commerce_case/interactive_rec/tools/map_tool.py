from typing import *

from module.buffer import CandidateBuffer
from module.corpus import BaseGallery
from utils.util import extract_integers_from_string

_NAME = "Mapping Tool"
_DESC = """The tool is useful when you want to convert beauty id to beauty title before showing beauty to human. \
The tool is able to get stored beauty in the buffer and randomly select a specific number of beauty from the buffer. \
The input of the tool should be an integer indicating the number of beauty human needs. \
The default value is 5 if human doesn't give."""


class MapTool:
    def __init__(self, name: str, desc: str, item_corpus: BaseGallery, buffer: CandidateBuffer, max_rec_num: int = 20,
                 return_cols: List[str] = ['title']) -> None:
        self.name = name
        self.desc = desc
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.max_rec_num = max_rec_num
        self.return_cols = return_cols

    def run(self, inputs: str) -> str:
        # id = eval(os.environ.get('llm4crs_candidates', "[]"))
        try:
            n = extract_integers_from_string(inputs)[0]
        except Exception as e:
            print(e)
            n = 5
        id = self.buffer.get()

        if n > self.max_rec_num:
            prefix = "Sorry, I could only recommend you {} items per time. ".format(self.max_rec_num)
            n = self.max_rec_num
        else:
            prefix = ""

        if len(id) <= 0:
            # prefix += "There is no candidate in buffer, randomly sample {} as recommendation. ".format(n)
            # id = random.sample(list(range(1, self.item_corpus.corpus.shape[0]+1)), n)
            return "There is no suitable items."
        else:
            prefix += "There is {} candidates in buffer, select the first {} as recommendation. ".format(len(id), min(n,
                                                                                                                      len(id)))
            id = id[:n]

        info = self.item_corpus.convert_id_2_info(id, col_names=self.return_cols)

        reco_info = []
        for i in range(len(id)):
            s = ""
            for k in info.keys():
                s += f"{info[k][i]}"
            reco_info.append(s)
        reco_str = '; '.join(reco_info)
        self.buffer.clear()  # clear buffer when one-turn chat ends
        output = prefix + f"Here are recommendations: " + reco_str
        self.buffer.track(self.name, inputs, output)
        return output
