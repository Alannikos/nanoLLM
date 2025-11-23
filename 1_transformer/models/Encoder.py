"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

from blocks.EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_head: int,
        embedding_dim: int,
        intermediate_dim: int,
        dropout: float,
        device
    ) -> None:
        """Encoder部分

        Parameters
        ----------
        n_layers : int
            encoderLayer的层数
        n_head : int
            多头注意力的头数
        embedding_dim : int
            embedding_dim
        intermediate_size : int
            mlp的中间隐藏层的dim
        dropout : float
            随机失活（随机置为0）
        """

        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(
                n_head=n_head,
                embedding_dim=embedding_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                device=device
            ) for _ in range(n_layers)]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        enc_mask: torch.Tensor
    ) -> torch.Tensor:
        # 多个相同的EncoderLayer
        for layer in self.layers:
            x = layer(x, enc_mask)
        
        return x
