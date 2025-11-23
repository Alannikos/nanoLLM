"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    def __init__(
        self,
        d_k: int,
        device
    ) -> None:
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k
        self.device = device
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """点积注意力的实现

        Parameters
        ----------
        query : torch.Tensor
            经过Q矩阵的变换的input, shape=(batch_size, n_head, seq_len, d_k)
        key : torch.Tensor
            经过K矩阵的变换的input, shape=(batch_size, n_head, seq_len, d_k)
        value : torch.Tensor
            经过V矩阵的变换的input, shape=(batch_size, n_head, seq_len, d_k)
        mask : torch.Tensor
            掩码
        """
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        mask = mask.to(self.device)

        # shape = (batch_size, n_head, seq_len, seq_len)
        attention_weight = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask, 1e-9)

        attention_wieght = self.softmax(attention_wieght)

        # (batch_size, n_head, seq_len, d_k)
        attention_score = torch.matmul(attention_wieght, value)

        return attention_score, attention_wieght