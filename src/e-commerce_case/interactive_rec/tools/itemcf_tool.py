from ast import literal_eval

import numpy as np
from module.buffer import CandidateBuffer
from module.corpus import BaseGallery
from loguru import logger


# retrieval tool


class SimilarItemTool:
    def __init__(self, name: str, desc: str, item_sim_path: str, item_corpus: BaseGallery, buffer: CandidateBuffer,
                 top_ratio: float = 0.05, **kwargs) -> None:
        self.item_sim = np.load(item_sim_path, allow_pickle=False)
        self.item_corpus = item_corpus
        self.buffer = buffer
        assert 0 < top_ratio < 1, f"`top_ratio` should be a float between 0~1, but got {top_ratio}"
        self.top_ratio = top_ratio
        self.rec_num = int(self.top_ratio * len(self.item_corpus))
        self.name = name
        self.desc = desc

    def run(self, inputs):
        logger.info(f"\n{self.name} input: {inputs}")
        try:
            games = literal_eval(inputs)
        except Exception as e:
            logger.info(e)
            games = []
            info = f"{self.name}: Input format error."
            self.buffer.track(self.name, inputs, info)
            return info

        info = ""

        candidates = self.buffer.get()

        if len(candidates) > 0:
            info += f"Before {self.name}: There are {len(candidates)} candidate in buffer. \n"
        else:
            # candidates = list(range(1, self.item_corpus.corpus.shape[0]+1))
            info += f"Before {self.name}: There is no candidate in buffer now. \n"
            return info

        try:
            candidates = np.array(candidates)
            titles = self.item_corpus.fuzzy_match(games, 'title')
            logger.info(f"Seed items: {titles}")
            games_not_found = {}
            for i, g in enumerate(games):
                if g.lower() not in titles[i].lower():
                    games_not_found[g] = titles[i]
            if len(games_not_found) > 0:
                info += (f"{self.name}: {list(games_not_found.keys())} not found in the database. "
                         f"Instead, games similar to {list(games_not_found.values())} would be selected. Tell the "
                         f"truth to human.\n")
            game_ids = self.item_corpus.convert_title_2_info(titles, col_names='id')['id']
            sim_scores = self.item_sim[game_ids][:, candidates].mean(axis=0, keepdims=False)

            all_score = self.item_sim[game_ids][:, 1:].mean(axis=0, keepdims=False)
            thres = np.partition(all_score, -self.rec_num)[-self.rec_num]

            flag = sim_scores > thres
            candidates = candidates[flag]
            cand_scores = sim_scores[flag]

            res = candidates.tolist()
            info += (f"After {self.name}: {len(res)} items similar to {titles} are selected and stored because their "
                     f"similarity scores are above the threshold. Those items are not sorted by similarity now.")

            self.buffer.push(self.name, res)
            sim_score_saved = cand_scores.tolist()
            self.buffer.save_similarity(sim_score_saved)
        except Exception as e:
            info = f"{self.name}: some thing went wrong in execution, the tool is broken for current input."

        self.buffer.track(self.name, inputs, info)
        return info

    def clear(self):
        pass
