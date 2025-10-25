import os
import math
import copy
import argparse

import torch
import torch.nn as nn
from torch.nn import funtional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .dataset import MTDataset
# from .utils import *

# 这里有的继承nn.Embedding?
class TokenEmbedding(nn.Module):
    """
    在这个过程中，TokenEmbedding这个模块主要是建立一个嵌入向量的矩阵，
    由token ID转化为嵌入向量这个过程是通过查找嵌入向量矩阵来的。我们在训练时，
    嵌入向量矩阵中的参数也是需要训练的。
    """
    def __init__(self, embedding_dim: int, vocab_size: int):
        """ToeknEmbedding的初始化

        Parameters
        ----------
        embedding_dim : int
            embedding的维度，通常是512等
        vocab_size : int
            词汇表vocab的大小
        """
        super(TokenEmbedding, self).__init__()
        self.lut = nn.Embedding(num_embeddings=vocab_size, \
                                embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor):
        """x的处理流程

        Parameters
        ----------
        x : torch.Tensor
            token的idx序列
        """

        # embedding matrix的初始化方式是xavier init，这种方式的方差是1/embedding size，
        # 因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更
        # 有利于embedding matrix的收敛。
        return self.lut(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    在注意力机制中，不仅要关注序列本身的信息，还要关注它们各自位置之间的信息，
    例如“我吃苹果”和“苹果吃我”，两句话的token相同，但是意思确是天差地别。最终
    的Transformer词向量是由词嵌入向量和位置编码向量直接相加得到的。
    """

    def __init__(self, embedding_dim: int, dropout: float, max_len: int = 5000):
        """位置编码初始化

        Parameters
        ----------
        embedding_dim : int
            embedding的维度
        dropout : float
            dropout的比例
        max_len : int, optional
            表示一个句子的最大长度, by default 5000
        """

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

        # 计算位置编码
        pe = torch.zeros(max_len, embedding_dim)
        # (max_len, ) -> (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)

        """
        数学推导：(d_model == embedding_dim)
            10000^(2i/d_model) = exp(log(10000^(2i/d_model)))
                               = exp((2i/d_model) * log(10000))
            倒数：1/10000^(2i/d_model) = exp(-(2i/d_model) * log(10000))
        """
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) \
                             * (-math.log(10000.0)) / embedding_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, embedding_dim) -> (1, max_len, embedding_dim)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        
        # x.shape = (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        # 加dropout: 随机丢弃一部分神经元的输出，防止模型过拟合。
        return self.dropout(x)



if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="choose the mode: train or test")
    parser.add_argument("--batch_size", default=64, type=int, help="set the batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="set the learning rate")
    parser.add_argument("--num_epochs", default=1, type=int, help="set the number of epochs")
    parser.add_argument("--embedding_dim", default=128, type=int, help="set number of word embedding")
    parser.add_argument("--gpu", default=0, type=int, help="refer to the GPU ID")
    parser.add_argument("--head_num", default=8, type=int, help="set the multi head number")
    parser.add_argument("--hidden_num", default=2048, type=int, help="set the hidden neural number")
    parser.add_argument("--dropout", default=0.2, type=float, help="dropout rate, default=0.2")
    parser.add_argument("--n_padding", default=50, type=int, help="set the padding length")
    parser.add_argument("--model_path", type=str, help="refer to thr trained model path")

    args = parser.parse_args()

    # 指定device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
        if args.gpu >= 0:
            print(f"Warning: GPU {args.gpu} is not available, using CPU instead")
    
    # 准备数据读取数据
    trainData = MTDataset(n_padding=args.n_padding, mode="train")
    validData = MTDataset(n_padding=args.n_padding, mode='valid')
    testData = MTDataset(n_padding=args.n_padding, mode='test')

    # 准备dataloader
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=32, drop_last=True)
    validDataLoader = DataLoader(validData, batch_size=args.batch_size, shuffle=True, num_workers=32, drop_last=True)
    testDataLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True, num_workers=32, drop_last=True)

    # 对于Vocab来说，三种模式下的vocabZh和vocabEn都一样
    zhVocabLen = trainData.getVocabZhLen()
    enVocabLen = trainData.getVocabEnLen()

