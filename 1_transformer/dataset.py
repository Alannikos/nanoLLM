import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, n_padding: int, mode: str="train"):
        """构造函数

        Parameters
        ----------
        n_padding : int
            指定padding的长度
        mode : str, optional
            指定当前模式是训练还是测试, by default "train"
        """

        assert mode in ['train', 'valid', 'test'], "mode must in ['train', 'valid', 'test']"

        print(f"load {mode} data")
        self.mode = mode
        self.n_padding = n_padding

        # 读取词表文件
        self.vocab_zh = np.load("../data/vocab_zh.npy").item()
        self.vocab_en = np.load("../data/vocab_en.npy").item()

        # 读取数据文件
        self.data_zh = self._getData(f"../data/{mode}.zh.tok", language="zh")
        self.data_en = self._getData(f"../data/{mode}.en.tok", language="en")

    def _getData(self, file_path: str, language: str):
        """读取数据文件，并将其转化为词表对应的index

        Parameters
        ----------
        file_path : str
            文本数据的路径
        language : str
            指定什么类型的语言
        """

        with open(file_path, "r") as f:
            data = [x.strip() for x in f.readlines()]  # readlines返回的是一个列表
        
        # 将数据通过词表转换为对应的index
        dataset = []
        for sentence in data:
            # 根据指定的语言将文本转化为index
            if language == "en":
                dataset.append([self.vocab_en.get(x, 2) for x in sentence.split(" ")])
            else:
                dataset.append([self.vocab_en.get(x, 2) for x in sentence.split(" ")])

        return dataset

    def __len__(self):
        return len(self.data_en)
    
    def getVocabZhLen(self):
        return len(self.vocab_zh)
    
    def getVocabEnLan(self):
        return len(self.vocab_en)
    
    # 记得修一下这里的type
    def __getitem__(self, index):
        """按照index返回对应的en和zh数据

        Parameters
        ----------
        index : _type_
            _description_
        """

        zh = self.data_zh[index]
        en = self.data_en[index]

        # {'<START>': 0, '<END>': 1, 'UNK': 2, 'PAD': 3}
        # 为句子加上 start 和 end 符号
        zh = [0] + zh + [1]
        en = [0] + en + [1]

        len_zh = len(zh) - 1

        # 对句子进行padding
        if len(zh) < self.padding:
            zh.extend([3] * (self.padding - len(zh)))
        else:
            zh = zh[:self.padding]

        if len(en) < self.n_padding:
            en.extend([3] * (self.n_padding - len(en)))
        else:
            en = en[:self.padding]
        
        zh = torch.tensot(zh)
        en = torch.tensor(en)

        # 训练用的zh[:-1], 作为目标的zh[1:]
        if self.mode == "train":
            return en, zh[:-1], zh[1:], len_zh
        else:
            return en, zh, zh, len_zh

