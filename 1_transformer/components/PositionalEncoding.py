"""
author: Alannikos
date:   2025/11/11
"""

import os
import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    在注意力机制中，不仅要关注序列本身的信息，还要关注它们各自位置之间的信息，
    例如“我吃苹果”和“苹果吃我”，两句话的token相同，但是意思确是天差地别。最终
    的Transformer词向量是由词嵌入向量和位置编码向量直接相加得到的。
    """
    def __init__(
        self,
        embedding_dim: int,
        max_len: int,
        dropout: float,
        device
    ) -> None:
        """位置编码初始化

        Parameters
        ----------
        embedding_dim : int
            embedding的维度
        max_len : int, optional
            表示一个句子的最大长度, by default 128
        dropout : float
            dropout的比例
        """

        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, embedding_dim, device=device)
        pe.requires_grad = False

        # (max_len, ) -> (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        _2i = torch.arange(0, embedding_dim, step=2, device=device)
        # (embedding_dim / 2, ) -> (1, embedding_dim / 2)
        _2i = _2i.unsqueeze(0)

        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / embedding_dim)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / embedding_dim)))

        # (max_len, embedding_dim) -> (1, max_len, embedding_dim)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # x.shape = (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        # 加dropout: 随机丢弃一部分神经元的输出，防止模型过拟合。
        return self.dropout(x)