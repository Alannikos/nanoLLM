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

class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        embedding_dim: int,
        intermediate_dim: int,
        dropout: float,
        device
    ) -> None:
        """decoder的一个layer

        Parameters
        ----------
        n_head : int
            头的数量
        embedding_dim : int
            embedding的dim
        intermediate_dim : int
            mlp的中间隐藏层的dim
        dropout : float
            随机失活（随机置为0）
        """

        super(DecoderLayer, self).__init__()

        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.device = device

        #========================================
        #              maskAttention            #
        #========================================
        self.maskAttention = MultiHeadAttention(
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
        #             Attention                 #
        #========================================
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_head=n_head,
            device=device
        )
        self.norm2 = LayerNorm(
            device=device,
            embedding_dim=embedding_dim
        )
        self.dropout2 = nn.Dropout(p=dropout)

        #========================================
        #                  FFN                  #
        #========================================
        self.ffn = PositionWiseFeedForward(
            embedding_dim=embedding_dim,
            interdediate_dim=intermediate_dim
        )
        self.norm3 = LayerNorm(
            device=device,
            embedding_dim=embedding_dim
        )
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        decoder_x: torch.Tensor,
        encoder_y: torch.Tensor,
        decoder_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor
    ) -> torch.Tensor:
        """decoder的一个block的过程

        Parameters
        ----------
        decoder_x : torch.Tensor
            shape=(batch_size, target_len, embedding_dim)
        encoder_y : torch.Tensor
            shape=(batch_size, source_len, embedding_dim)
        decoder_mask : torch.Tensor
            shape=(batch_size, target_len, target_len)
        enc_dec_mask : torch.Tensor
            shape=(batch_size, target_len, source_len)
        """

        # step1: 计算mask注意力分数
        residual = decoder_x
        x = self.maskAttention(decoder_x, decoder_x, decoder_x, decoder_mask)
        # step2: add&norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        if encoder_y is not None:
            # step3: 计算encoder-decoder attention分数
            # 注意这里的residual是x而不是encoder_y
            residual = x
            x = self.attention(query=x, key=encoder_y, value=encoder_y, mask=enc_dec_mask)

            # step4: add&norm
            x = self.dropout2(x)
            x = self.norm2(x + residual)

        # step5: fnn
        residual = x
        x = self.ffn(x)
        # step6: add & norm
        x = self.dropout3(x)
        x = self.norm3(x + residual)

        return x