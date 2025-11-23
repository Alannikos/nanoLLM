"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

from components.LayerNorm import LayerNorm
from components.MultiHeadAttention import MultiHeadAttention
from components.PositionWiseFeedForward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        embedding_dim: int,
        intermediate_dim: int,
        dropout: float,
        device
    ) -> None:
        """encoder的一层layer，主要包含attention和mlp

        Parameters
        ----------
        n_head : int
            multiHead的头数
        embedding_dim : int
            embedding的dim
        intermediate_size : int
            mlp的中间隐藏层的dim
        dropout : float
            随机失活（随机置为0）
        """

        super(EncoderLayer, self).__init__()

        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.device = device

        #========================================
        #             Attention                 #
        #========================================
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_head=n_head,
            device=device
        )
        self.norm1 = LayerNorm(
            device=device,
            embedding_dim=embedding_dim
        )
        self.dropout1 = nn.Dropout(p=dropout)

        #========================================
        #                  FFN                  #
        #========================================
        self.ffn = PositionWiseFeedForward(
            embedding_dim=embedding_dim,
            interdediate_dim=intermediate_dim
        )
        self.norm2 = LayerNorm(
            device=device,
            embedding_dim=embedding_dim
        )
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """一个encoder的前向传播

        Parameters
        ----------
        x : torch.Tensor
            每个encoderLayer的输入，shape=(batch_size, seq_len, embedding_dim)
        mask : torch.Tensor
            掩码
        """

        x = x.to(self.device)
        mask = mask.to(self.device)

        #========================================
        #             Attention                 #
        #========================================
        residual = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        # add & norm
        x = self.norm1(residual + x)

        #========================================
        #                  FFN                  #
        #========================================
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        # add & norm
        x = self.norm2(residual + x)

        return x
