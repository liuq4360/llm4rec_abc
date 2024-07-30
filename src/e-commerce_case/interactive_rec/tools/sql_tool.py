import random
import re

from module.buffer import CandidateBuffer
from module.corpus import BaseGallery
from utils.sql import extract_columns_from_where
from loguru import logger


# retrieval tool


class SQLSearchTool:

    def __init__(self, name: str, desc: str, item_corpus: BaseGallery, buffer: CandidateBuffer,
                 max_candidates_num: int = None) -> None:
        self.item_corpus = item_corpus
        self.buffer = buffer
        self.name = name
        self.desc = desc
        self.max_candidates_num = max_candidates_num

    def run(self, inputs: str) -> str:
        # candidates = eval(os.environ.get("llm4crs_candidates", "[]"))
        info = ""
        candidates = self.buffer.get()
        if len(candidates) > 0:
            info += f"Before {self.name}: There are {len(candidates)} candidates in buffer. \n"
            corpus = self.item_corpus.corpus.loc[candidates]
        else:
            info += f"Before {self.name}: There are {len(candidates)} candidates in buffer. Stop execution. \n"
            return info

        logger.info(f"\nSQL from AGI: {inputs}")
        try:
            inputs = self.rewrite_sql(inputs)
            logger.info(f"Rewrite SQL: {inputs}")
            info += (f"{self.name}: The input SQL is rewritten as {inputs} because "
                     f"some {list(self.item_corpus.categorical_col_values.keys())} are not existing. \n")
        except Exception as e:
            logger.exception(e)
            info += (f"{self.name}: something went wrong in execution, the tool is broken for current input. "
                     f"The candidates are not modified.\n")
            return info

        try:
            candidates = self.item_corpus(inputs, corpus=corpus)  # list of ids
            n = len(candidates)
            _info = f"After {self.name}: There are {n} eligible items. "
            if self.max_candidates_num is not None:
                if len(candidates) > self.max_candidates_num:
                    if "order" in inputs.lower():
                        candidates = candidates[: self.max_candidates_num]
                        _info += (f"Select the first {self.max_candidates_num} items from all eligible items ordered "
                                  f"by the SQL.")
                    else:
                        candidates = random.sample(candidates, k=self.max_candidates_num)
                        _info += f"Random sample {self.max_candidates_num} items from all eligible items. "
                else:
                    pass
            else:
                pass

            info += _info
            self.buffer.push(self.name, candidates)
        except Exception as e:
            logger.info(e)
            candidates = []
            info = (f"{self.name}: something went wrong in execution, the tool is broken for current input. The "
                    f"candidates are not modified.")

        self.buffer.track(self.name, inputs, info)

        # suffix = f"{len(candidates)} candidate games are selected with SQL command {inputs}. Those candidate games
        # are stored and visible to other tools. Now you need to take the next action." return  f"Here are candidates
        # id searched with the SQL command: [{','.join(map(str, candidates))}]."
        logger.info(f"{info}")
        return info

    def rewrite_sql(self, sql: str) -> str:
        """Rewrite SQL command using fuzzy search"""
        sql = re.sub(r'\bFROM\s+(\w+)\s+WHERE', f'FROM {self.item_corpus.name} WHERE', sql, flags=re.IGNORECASE)

        # grounding cols
        cols = extract_columns_from_where(sql)
        existing_cols = set(self.item_corpus.column_meaning.keys())
        col_replace_dict = {}
        for col in cols:
            if col not in existing_cols:
                mapped_col = self.item_corpus.fuzzy_match(col, 'sql_cols')
                col_replace_dict[col] = f"{mapped_col}"
        for k, v in col_replace_dict.items():
            sql = sql.replace(k, v)

        # grounding categorical values
        pattern = r"([a-zA-Z0-9_]+) (?:NOT )?LIKE '\%([^\%]+)\%'"
        res = re.findall(pattern, sql)
        replace_dict = {}
        for col, value in res:
            if col not in self.item_corpus.fuzzy_engine:
                continue
            replace_value = str(self.item_corpus.fuzzy_match(value, col))
            replace_value = replace_value.replace("'", "''")  # escaping string for sqlite
            replace_dict[f"%{value}%"] = f"%{replace_value}%"

        for k, v in replace_dict.items():
            sql = sql.replace(k, v)
        return sql
