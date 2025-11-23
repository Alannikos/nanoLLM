"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        interdediate_dim: int
    ) -> None:
        """前馈层，两个linear层加一个Relu"""
        super(PositionWiseFeedForward, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, interdediate_dim),
            nn.ReLU(),
            nn.Linear(interdediate_dim, embedding_dim)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:            
        # shape = (batch_size, seq_len, embedding_dim)
        out = self.fc(x)
        return out
