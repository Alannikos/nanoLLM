"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class OutputGenerator(nn.Module):
    def __init__(
        self,
        embedding_dim,
        vocab_size: int,
        device
    ) -> None:
        """decoder的输出还需要经过一个线性变化和softmax操作

        Parameters
        ----------
        embedding_dim : int
            embedding的维度，通常是512等
        vocab_size : int
            decoder词汇表vocab的大小
        """

        super(OutputGenerator, self).__init__()
        self.device = device

        self.proj = nn.Linear(embedding_dim, vocab_size, device=device)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        output = self.proj(x)

        return output
