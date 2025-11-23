"""
author: Alannikos
date:   2025/11/11
"""

import os
import math

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(
        self,
        device,
        embedding_dim: int,
        eps: float = 1e-12
    ) -> None:
        """LayerNorm实现

        Parameters
        ----------
        embedding_dim : int
            输入张量的特征维度（最后一个维度的大小）
        eps : float, optional
            eps, by default 1e-12
        """

        self.eps = eps
        self.device = device

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameters(torch.ones(embedding_dim))
        self.beta = nn.Parameters(torch.zeros(embedding_dim))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """进行norm的前向过程

        Parameters
        ----------
        x : torch.Tensor
            形状一般是(batch_size, seq_len, embedding_dim)
        """
        x = x.to(self.device)
        
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)

        out = (x - x_mean) / torch.sqrt(x_var + self.eps)
        # pytorch的广播机制
        out = self.gamma * out + self.beta

        return out
