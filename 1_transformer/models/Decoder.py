"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

from blocks.DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_head: int,
        embedding_dim: int,
        intermediate_dim: int,
        dropout: float,
        device
    ) -> None:
        """Decoder部分

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

        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [DecoderLayer(
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
        encoder_y: torch.Tensor,
        decoder_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor
    ) -> torch.Tensor:
        # 多个相同的DncoderLayer
        for layer in self.layers:
            x = layer(
                decoder_x=x,
                encoder_y=encoder_y,
                decoder_mask=decoder_mask,
                enc_dec_mask=enc_dec_mask
            )
        
        return x
