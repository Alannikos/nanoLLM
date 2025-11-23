"""
author: Alannikos
date:   2025/11/12
"""

import os
import math

import torch
import torch.nn as nn

from components.TokenEmbedding import TokenEmbedding
from components.PositionalEncoding import PositionalEncoding
from models.Encoder import Encoder
from models.Decoder import Decoder
from models.OutputGenerator import OutputGenerator

class Transformer(nn.Module):
    def __init__(
        self,
        padding_idx: int,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        n_head: int,
        n_layers: int,
        max_len: int,
        embedding_dim: int,
        intermediate_dim: int,
        dropout: float,
        device
    ) -> None:
        super(Transformer, self).__init__()

        self.device = device

        # Embedding层
        self.src_embedding = TokenEmbedding(
            embedding_dim=embedding_dim,
            vocab_size=encoder_vocab_size,
            padding_idx=padding_idx
        )
        self.tgt_embedding = TokenEmbedding(
            embedding_dim=embedding_dim,
            vocab_size=decoder_vocab_size,
            padding_idx=padding_idx
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_len=max_len,
            dropout=dropout,
            device=device
        )

        # 核心内容
        self.encoder = Encoder(
            n_layers=n_layers,
            n_head=n_head,
            embedding_dim=embedding_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            device=device
        )
        self.decoder = Decoder(
            n_layers=n_layers,
            n_head=n_head,
            embedding_dim=embedding_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            device=device
        )
        self.generator = OutputGenerator(
            embedding_dim=embedding_dim,
            vocab_size=decoder_vocab_size,
            device=device
        )

    def forward(
        self,
        encoder_x: torch.Tensor,
        decoder_x: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor
    ) -> torch.Tensor:
        encoder_y = self.encode(encoder_x, encoder_mask)

        decoder_output = self.decode(
            decoder_x=decoder_x,
            encoder_y=encoder_y,
            decoder_mask=decoder_mask,
            enc_dec_mask=enc_dec_mask
        )

        output = self.generator(decoder_output)

        return output

    def encode(
        self,
        encoder_x: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        # (batch_size, max_len) -> (batch_size, max_len, embedding_dim)
        src_emb = self.src_embedding(encoder_x)
        # 加上位置信息
        src_emb = self.positional_encoding(src_emb)

        encoder_y = self.encoder(src_emb, encoder_mask)

        return encoder_y

    def decode(
        self,
        decoder_x: torch.Tensor,
        encoder_y: torch.Tensor,
        decoder_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor
    ) -> torch.Tensor:
        # (batch_size, max_len) -> (batch_size, max_len, embedding_dim)
        tgt_emb = self.tgt_embedding(decoder_x)
        # 加上位置信息
        tgt_emb = self.positional_encoding(tgt_emb)

        decoder_output = self.decoder(
            tgt_emb,
            encoder_y,
            decoder_mask,
            enc_dec_mask
        )

        return decoder_output
