import json
import os
import random
import sys
from typing import *

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_integer_dtype, is_bool_dtype, is_float_dtype, is_datetime64_dtype, is_object_dtype, \
    is_categorical_dtype
from pandasql import sqldf
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.exceptions import raise_error
from utils.text_sim import SentBERTEngine

_REQUIRED_COLUMNS = ['id', 'title']


def _pd_type_to_sql_type(col: pd.Series) -> str:
    if is_integer_dtype(col):
        return 'integer'
    elif is_float_dtype(col):
        return 'float'
    elif is_bool_dtype(col):
        return 'boolean'
    elif is_datetime64_dtype(col):
        return 'datetime'
    elif is_object_dtype(col) or is_categorical_dtype(col):
        return 'string'
    else:
        return 'string'


class BaseGallery:
    def __init__(self, fpath: str, column_meaning_file: str,
                 name: str = 'beauty_product_information', columns: List[str] = None,
                 sep: str = ',', parquet_engine: str = 'pyarrow',
                 fuzzy_cols=None, categorical_cols=None) -> None:
        if categorical_cols is None:
            categorical_cols = ["category", "brand"]
        if fuzzy_cols is None:
            fuzzy_cols = ['title']
        self.fpath = fpath
        self.name = name  # name of the table
        self.corpus = self._read_file(self, fpath=fpath, columns=columns, sep=sep, parquet_engine=parquet_engine)
        self.disp_cate_topk: int = 6
        self.disp_cate_total: int = 10
        self._fuzzy_bert_base = "thenlper/gte-base"
        self._required_columns_validate()
        self.column_meaning = self._load_col_desc_file(self, fpath=column_meaning_file)

        self.categorical_col_values = {}
        for col in categorical_cols:
            _explode_df = self.corpus.explode(col)[col]
            self.categorical_col_values[col] = _explode_df[~ _explode_df.isna()].unique()
            if isinstance(self.corpus[col][0], str):
                pass
            else:
                self.corpus[col] = self.corpus[col].apply(lambda x: ', '.join(x))

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'mps'
        _fuzzy_bert_engine = SentenceTransformer(self._fuzzy_bert_base, device=device)
        self.fuzzy_engine: Dict[str, SentBERTEngine] = {
            col: SentBERTEngine(
                self.corpus[col].to_numpy(),
                self.corpus["id"].to_numpy(),
                case_sensitive=False,
                model=_fuzzy_bert_engine
            )
            if col not in categorical_cols
            else SentBERTEngine(
                self.categorical_col_values[col],
                np.arange(len(self.categorical_col_values[col])),
                case_sensitive=False,
                model=_fuzzy_bert_engine
            )
            for col in fuzzy_cols
        }
        self.fuzzy_engine['sql_cols'] = SentBERTEngine(
            np.array(columns),
            np.arange(len(columns)),
            case_sensitive=False,
            model=_fuzzy_bert_engine
        )  # fuzzy engine for column names
        # title as index
        self.corpus_title = self.corpus.set_index('title', drop=True)
        # id as index
        self.corpus.set_index('id', drop=True, inplace=True)

    def __call__(self, sql: str, corpus: pd.DataFrame = None, return_id_only: bool = True) -> List:
        """Search in corpus with SQL query
        
        Args:
            sql: A sql query command.

        Returns:
            list: the result represents by id
        """
        if corpus is None:
            result = sqldf(sql, {self.name: self.corpus})  # all beauty
        else:
            result = sqldf(sql, {self.name: corpus})  # beauty in buffer

        if return_id_only:
            result = result[self.corpus.index.name].to_list()
        return result

    def __len__(self) -> int:
        return len(self.corpus)

    def info(self, remove_game_titles: bool = False, query: str = None):
        prefix = 'Table information:'
        table_name = f"Table Name: {self.name}"
        cols_info = "Column Names, Data Types and Column meaning:"
        cols_info += f"\n    - {self.corpus.index.name}({_pd_type_to_sql_type(self.corpus.index)}): {self.column_meaning[self.corpus.index.name]}"
        for col in self.corpus.columns:
            if remove_game_titles and 'title' in col:
                continue
            dtype = _pd_type_to_sql_type(self.corpus[col])
            cols_info += f"\n    - {col}({dtype}): {self.column_meaning[col]}"
            if col == 'tags':
                disp_values = self.sample_categorical_values(col, total_n=self.disp_cate_total, query=query,
                                                             topk=self.disp_cate_topk)
                _prefix = f" Related values: [{', '.join(disp_values)}]."
                cols_info += _prefix

            if dtype in {'float', 'datetime', 'integer'}:
                _min = self.corpus[col].min()
                _max = self.corpus[col].max()
                _mean = self.corpus[col].mean()
                _median = self.corpus[col].median()
                _prefix = f" Value ranges from {_min} to {_max}. The average value is {_mean}. The median is {_median}."
                cols_info += _prefix

        primary_key = f"Primary Key: {self.corpus.index.name}"
        categorical_cols = list(self.categorical_col_values.keys())
        note = (f"Note that [{','.join(categorical_cols)}] columns are categorical, must use related values to search "
                f"otherwise no result would be returned.")
        res = ''
        for i, s in enumerate([table_name, cols_info, primary_key, note]):
            res += f"\n{i}. {s}"
        res = prefix + res
        return res

    def sample_categorical_values(self, col_name: str, total_n: int, query: str = None, topk: int = None) -> List:
        # Select topk related tags according to query and sample (total_n-topk) tags
        if query is None:
            result = random.sample(self.categorical_col_values[col_name], k=total_n)
        else:
            if topk is None:
                topk = total_n
            assert total_n >= topk, f"`topk` must be smaller than `total_n`, while got {topk} > {total_n}."
            topk_values = self.fuzzy_engine[col_name](query, return_doc=True, topk=topk)
            topk_values = list(topk_values)
            result = topk_values
            if total_n > topk:
                while (len(result) < total_n) and (len(result) < len(self.categorical_col_values[col_name])):
                    random_values = random.choice(self.categorical_col_values[col_name])
                    if random_values not in result:
                        result.append(random_values)
        return result

    def convert_id_2_info(self, item_id: Union[int, List[int], np.ndarray],
                          col_names: Union[str, List[str]] = None) -> \
            Union[Dict, List[Dict]]:
        """Given game item_id, get game informations.
        
        Args:
            - item_id: game ids. 
            - col_names: column names to be returned

        Returns:
            - information of given game ids, each game is formatted as a dict, whose key is column name.
        
        """
        if col_names is None:
            col_names = self.corpus.columns
        else:
            if isinstance(col_names, str):
                col_names = [col_names]
            elif isinstance(col_names, list):
                pass
            else:
                raise_error(TypeError, "Not supported type for `col_names`.")

        if isinstance(item_id, int):
            items = self.corpus.loc[item_id][col_names].to_dict()
        elif isinstance(item_id, list) or isinstance(item_id, np.ndarray):
            items = self.corpus.loc[item_id][col_names].to_dict(orient='list')
        else:
            raise_error(TypeError, "Not supported type for `item_id`.")

        return items

    def convert_title_2_info(self, titles: Union[int, List[int], np.ndarray],
                             col_names: Union[str, List[str]] = None) -> Union[Dict, List[Dict]]:
        """Given game title, get game informations.
        
        Args:
            - titles: game titles. Note that the game title must exist in the table. 
            - col_names: column names to be returned

        Returns:
            - information of given game titles, each game is formatted as a dict, whose key is column name.
        
        """
        if col_names is None:
            col_names = self.corpus_title.columns
        else:
            if isinstance(col_names, str):
                col_names = [col_names]
            elif isinstance(col_names, list):
                pass
            else:
                raise_error(TypeError, "Not supported type for `col_names`.")

        if isinstance(titles, str) or (isinstance(titles, np.ndarray) and len(titles.shape) == 0):
            items = self.corpus_title.loc[titles][col_names].to_dict()
        elif isinstance(titles, list) or (isinstance(titles, np.ndarray) and len(titles.shape) > 0):
            items = self.corpus_title.loc[titles][col_names].to_dict(orient='list')
        else:
            raise_error(TypeError, "Not supported type for `titles`.")

        return items

    @staticmethod
    def _read_file(self, fpath: str, columns: List[str] = None, sep: str = ',',
                   parquet_engine: str = 'pyarrow') -> pd.DataFrame:
        if fpath.endswith('.csv') or fpath.endswith('.tsv'):
            df = pd.read_csv(fpath, sep=sep, names=columns)
        elif fpath.endswith('.ftr'):
            df = pd.read_feather(fpath)
        elif fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath, engine=parquet_engine)
        else:
            raise_error(TypeError, "Not support for such file type now.")
        if columns is not None:
            df = df[columns]
        return df

    @staticmethod
    def _load_col_desc_file(self, fpath: str) -> Dict:
        assert fpath.endswith('.json'), "Only support json file now."
        with open(fpath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _required_columns_validate(self) -> None:
        for col in _REQUIRED_COLUMNS:
            if col not in self.corpus.columns:
                raise_error(ValueError,
                            f"`id` and `name` are required in item corpus table but {col} not found, please check the "
                            f"table file `{self.fpath}`.")

    def fuzzy_match(self, value: Union[str, List[str]], col: str) -> Union[str, List[str]]:
        if col not in self.fuzzy_engine:
            raise_error(ValueError, f"Not support fuzzy search for column {col}")
        else:
            res = self.fuzzy_engine[col](value, return_doc=True, topk=1)
            res = np.squeeze(res, axis=-1)
            return res


if __name__ == '__main__':
    from environ_variables import (BEAUTY_INFO_FILE,
                                           TABLE_COL_DESC_FILE, USE_COLS, CATEGORICAL_COLS, CATEGORICAL_COLS)

    gallery = BaseGallery(
            BEAUTY_INFO_FILE,
            TABLE_COL_DESC_FILE,
            "beauty_product_information",
            columns=USE_COLS,
            fuzzy_cols=["title"] + CATEGORICAL_COLS,
            categorical_cols=CATEGORICAL_COLS,
        )
    # print(gallery.info())

    sql = f"SELECT * FROM {gallery.name} WHERE id<5"
    # sql = f"SELECT COUNT(*) FROM beauty_product_information;"

    print("SQL result: ", gallery(sql, return_id_only=False))

    res = gallery.fuzzy_match(['Call of Duty', 'Call Duty2'], 'title')
    print(res)
    res = gallery.fuzzy_match('Call of Duty', 'title')
    print(res)

    print('Test end.')
