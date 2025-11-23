"""
author: Alannikos
date:   2025/11/11
"""

import os
import math

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    在这个过程中，TokenEmbedding这个模块主要是建立一个嵌入向量的矩阵，
    由token ID转化为嵌入向量这个过程是通过查找嵌入向量矩阵来的。我们在训练时，
    嵌入向量矩阵中的参数也是需要训练的。
    """
    def __init__(self,
        embedding_dim: int,
        vocab_size: int,
        padding_idx: int
    ) -> torch.Tensor:
        """ToeknEmbedding的初始化

        Parameters
        ----------
        embedding_dim : int
            embedding的维度，通常是512
        vocab_size : int
            词汇表vocab的大小
        padding_idx: int
        """

        super(TokenEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        # padding_idx参数是一个索引，如果选定，
        # 那么此处的词向量全部设为0，并且不更新它的梯度
        self.lut = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x的处理流程

        Parameters
        ----------
        x : torch.Tensor
            token的idx序列
        """

        # embedding matrix的初始化方式是xavier init，这种方式的方差是1/embedding size，
        # 因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更
        # 有利于embedding matrix的收敛。
        return self.lut(x) * math.sqrt(self.embedding_dim)
