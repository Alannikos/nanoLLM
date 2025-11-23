"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

from ScaleDotProductAttention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_head: int,
        device
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.device = device

        assert embedding_dim % n_head == 0, "head的数量必须是embedding_dim的因子"

        # d_k就是各个head里面embedding_dim的大小
        self.d_k = embedding_dim // n_head

        # 下面就是attention
        self.attention = ScaleDotProductAttention(self.d_k, device=self.device)

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False, device=self.device)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False, device=self.device)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False, device=self.device)
        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False, device=self.device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # 流程：线性变化->划分多个head->attention->concat 

        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        mask = mask.to(self.device)

        batch_size = query.size(0)
        q, k, v = self.Wq(query), self.Wk(key), self.Wv(value)

        # (batch_size, seq_len, embedding_dim) -> \
        #        (batch_size, seq_len, n_head, d_k)
        q = q.reshape(batch_size, -1, self.n_head, self.d_k)
        k = k.reshape(batch_size, -1, self.n_head, self.d_k)
        v = v.reshape(batch_size, -1, self.n_head, self.d_k)

       # 进行转置方便注意力计算
        # (batch_size, seq_len, n_head, d_k) ->
        #     (batch_size, n_head, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 暂时还不知道这里做了什么变化，后面注意一下
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # (batch_size, n_head, seq_len, d_k)
        attention_output, attention_weights = self.attention(q, k, v, mask)
        # (batch_size, seq_len, n_head, d_k)
        attention_output = attention_output.transpose(1, 2)
        # (batch_size, seq_len, embedding_dim)
        attention_output = attention_output.reshape(batch_size, -1, self.embedding_dim)

        output = self.Wo(attention_output)

        return output