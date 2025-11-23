"""
author: Alannikos
date:   2025/11/23
"""
import math
import numpy as np
from typing import List
from collections import Counter


def bleu_stats(hypothesis: List[str], reference: List[str]):
    stats = []
    stats.append(len(hypothesis))  # 候选翻译长度
    stats.append(len(reference))   # 参考翻译长度

    for n in range(1, 5):  # 计算1-gram到4-gram
        h_ngrams = Counter(
            [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) + 1 - n)]
        )

        r_ngrams = Counter(
            tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)
        )

        # 计算匹配的n-gram数量
        stats.append(max([sum((h_ngrams & r_ngrams).values()), 0]))
        # 候选翻译中n-gram的总数
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    
    return stats

def bleu(stats: List):
    # 检查是否有零匹配的情况
    if len(list(filter(lambda x : x == 0, stats))) > 0:
        return 0
    
    (c, r) = stats[0:2]

    # 计算n-gram精确度的几何平均值的对数
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.0

    # 计算长度惩罚因子并返回BLEU分数
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypothesis: List[str], reference: List[str]):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # 累加所有翻译对的统计量
    for hyp, ref in zip(hypothesis, reference):
        stats += np.array(bleu_stats(hyp, ref))

    # 计算整体BLEU分数
    return 100 * bleu(stats)
