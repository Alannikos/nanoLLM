import os
import sys
import copy
import math
from typing import List

import torch
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor, pad_idx: int):
    """因为可能在数据中使用了padding, 不希望pad被加入到注意力中进行计算

    Parameters
    ----------
    seq_q : torch.Tensor
        (batch_size, len_q)
    seq_k : torch.Tensor
        (batch_size, len_k)
    pad_idx : int
        pad所使用的idx
    """

    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()

    # (batch_size, 1, len_k)
    pad_attn_mask = seq_k.eq(pad_idx).unsqueeze(1)

    pad_attn_mask.expand(batch_size, len_q, len_k)

    return pad_attn_mask

def get_subsequent_mask(seq: torch.Tensor):
    """生成因果mask

    Parameters
    ----------
    seq : torch.Tensor
        输入的序列, (batch_size, target_len)
    """

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # torch.triu是留下了上三角矩阵，然后diagonal为1的话，则对角线的是0
    # [[[0, 1, 1, 1],
    # [0, 0, 1, 1],
    # [0, 0, 0, 1],
    # [0, 0, 0, 0]]]
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.unit8), diagonal=1)

    return subsequent_mask

def aggreVocab(filePaths: List[str], outputPath: str):
    """提取原始数据的词表，并且保存为字典文件

    Parameters
    ----------
    filePaths : List[str]
        输入文件位置
    outputPath : str
        保存文件的位置
    """

    # 存储所有的词汇
    vocabList = []
    for file in filePaths:
        print(f"read {file}")
        with open(file, "r") as f:
            for line in f.readlines():
                # 遍历每个词汇
                words = line.strip().split(" ")
                for word in words:
                    vocabList.append(word)
    
    # vocabSet就是词表
    vocabSet = set(vocabList)

    # vocabDict存储{word:index}二元组
    vocabDict = {'<START>': 0, '<END>': 1, 'UNK': 2, 'PAD': 3}
    for index, word in enumerate(vocabSet):
        vocabDict[word] = index + 4
    
    np.save(outputPath, vocabDict)

def compute_bleu(translate, reference, references_lens):
    translate = translate.tolist()
    reference = reference.tolist()

    smooth = SmoothingFunction()
    references_lens = references_lens.tolist()
    bleu_score = []

    for translate_sentence, reference_sentence, references_len in zip(translate, reference, references_lens):
        if 1 in translate_sentence:
            index = translate_sentence.index(1)
        else:
            index = len(translate_sentence)
        
        bleu_score.append(sentence_bleu([reference_sentence[:references_len]], translate_sentence[:index], weights=(0.3, 0.4, 0.3, 0.0), smoothing_function=smooth.method1))

    return bleu_score

if __name__ == "__main__":

    # 把训练集，验证集和测试集的所有词汇都提取出来，防止OOD
    aggreVocab(['./data/train.en.tok', './data/valid.en.tok', './data/test.en.tok'], './data/vocab_en_freq_2.npy')
    aggreVocab(['./data/train.zh.tok', './data/valid.zh.tok', './data/test.zh.tok'], './data/vocab_zh_freq_2.npy')
