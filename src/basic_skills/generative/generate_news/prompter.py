import json
import os

import pandas as pd
from UniTok import UniDep
from tqdm import tqdm


# 生成MIND数据集的prompt
class MindPrompter:
    def __init__(self, data_path):
        self.data_path = data_path

        self.news_df = pd.read_csv(
            filepath_or_buffer=os.path.join(data_path),
            sep='\t',
            header=0,
        )

        self.keys = dict(
            title='title',
            abstract='abs',
            category='cat',
            subcategory='subcat',
        )

        self._news_list = None
        self._news_dict = None

    def stringify(self):
        if self._news_list is not None:
            return self._news_list
        self._news_list = []
        for news in tqdm(self.news_df.iterrows()):
            string = ''
            for key in self.keys:
                string += f'[{key}] {news[1][self.keys[key]]}\n'
            self._news_list.append((news[1]['nid'], string))
        return self._news_list

    def get_news_dict(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for news in tqdm(self.news_df.iterrows()):
            self._news_dict[news[1]['nid']] = news[1]['title']
        return self._news_dict

    def get_news_dict_with_category(self):
        if self._news_dict is not None:
            return self._news_dict
        self._news_dict = {}
        for news in tqdm(self.news_df.iterrows()):
            self._news_dict[news[1]['nid']] = f'({news[1]["cat"]}) {news[1]["title"]}'
        return self._news_dict


class MindUser:
    def __init__(self, data_path, mind_prompter):
        self.depot = UniDep(data_path, silent=True)
        self.nid = self.depot.vocabs('nid')
        self.news_dict = mind_prompter.get_news_dict()

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            string = ''
            if not user['history']:
                self._user_list.append((user['uid'], None))
            for i, n in enumerate(user['history']):
                string += f'({i + 1}) {self.news_dict[self.nid.i2o[n]]}\n'
            self._user_list.append((user['uid'], string))
        return self._user_list


# 生成用户新闻观看历史记录（包含类型，标题），下面是一个案例：
# (4, '(1) (autos) The Difference Between Green and Orange Antifreeze\n(2) (travel) Road built by biblical villain uncovered in Jerusalem\n(3) (weather) Boat inches closer to Niagara Falls edge after being grounded for century\n(4) (health) 24 Ways to Shrink Your Belly in 24 Hours\n')
class MindColdUser:
    def __init__(self, data_path, mind_prompter):
        self.depot = UniDep(data_path, silent=True)
        self.nid = self.depot.vocabs('nid')
        self.news_dict = mind_prompter.get_news_dict_with_category()

        self._user_list = None

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            string = ''
            if not user['history'] or len(user['history']) > 5:
                continue
            for i, n in enumerate(user['history']):
                string += f'({i + 1}) {self.news_dict[self.nid.i2o[n]]}\n'
            self._user_list.append((user['uid'], string))
        return self._user_list


class MindCoT:
    def __init__(self, data_path, plugin_path, mind_prompter, allowed_user_path):
        self.depot = UniDep(data_path, silent=True)
        self.plugin = UniDep(plugin_path, silent=True)
        self.tv = self.plugin.vocabs['topic']
        self.rv = self.plugin.vocabs['region']

        self.nid = self.depot.vocabs('nid')
        self.news_dict = mind_prompter.get_news_dict_with_category()

        self._user_list = None
        self.allowed_user = json.load(open(allowed_user_path))

    def stringify(self):
        if self._user_list is not None:
            return self._user_list
        self._user_list = []
        for user in tqdm(self.depot):
            if user['uid'] not in self.allowed_user:
                continue
            string = ''
            pg = self.plugin[user['uid']]
            string += 'Interest Topics:\n'
            for t in pg['topic']:
                string += f'- {self.tv[t]}\n'
            string += '\n'

            string += 'History:\n'
            for i, n in enumerate(user['history']):
                string += f'({i + 1}) {self.news_dict[self.nid.i2o[n]]}\n'
            self._user_list.append((user['uid'], string))
        return self._user_list
