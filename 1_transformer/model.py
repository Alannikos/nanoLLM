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
from .utils import get_attn_pad_mask, get_subsequent_mask
from .utils import *


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        """进行norm的前向过程

        Parameters
        ----------
        x : torch.Tensor
            形状一般是(batch_size, seq_len, embedding_dim)
        """

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out


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


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super(ScaleDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
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
            subsequent_mask
        """

        # shape = (batch_size, n_head, seq_len, seq_len)
        attention_wieght = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            # !!!
            # 这里写1e-9好像会有问题，之后再改
            attention_wieght = attention_wieght.masked_fill(mask, -1e9)

        attention_wieght = self.softmax(attention_wieght)

        # (batch_size, n_head, seq_len, d_k)
        # 这个embedding_dim要看实际上分了多少个head
        attention_score = torch.matmul(attention_wieght, value)

        return attention_score, attention_wieght


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, embedding_dim: int):
        self.n_head = n_head
        self.embedding_dim = embedding_dim

        assert embedding_dim % n_head == 0, "head的数量必须是embedding_dim的因子"

        # d_k就是各个head里面embedding_dim的大小
        self.d_k = embedding_dim // n_head
        
        # 下面就是attention, layernorm; 残差连接会体现在forward函数里
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.attention = ScaleDotProductAttention(self.d_k)

        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        # 注意，其实现在输入的q, k, v应该都是x

        # 流程：线性变化->划分多个head->attention->concat 

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

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        
        # (batch_size, n_head, seq_len, d_k)
        attention_output, attention_weights = self.attention(q, k, v, mask)
        # (batch_size, seq_len, n_head, d_k)
        attention_output = attention_output.transpose(1, 2)
        # (batch_size, seq_len, embedding_dim)
        attention_output = attention_output.reshape(batch_size, -1, self.embedding_dim)

        output = self.Wo(attention_output)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """前馈层，两个linear层加一个Relu

        Parameters
        ----------
        embedding_dim : int
        hidden_dim : int
        dropout : float, optional
            dropout, by default 0.1
        """

        super(PositionWiseFeedForward, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forwrd(self, x):

        # shape = (batch_size, seq_len, embedding_dim)
        out = self.fc(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, n_head: int, embedding_dim: int, hidden_dim: int, dropout: float):
        self.n_head = n_head
        self.embedding_dim = embedding_dim

        self.multiHeadAttention = MultiHeadAttention(n_head=n_head, embedding_dim=embedding_dim)
        self.norm1 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout1 = nn.Dropout(p=dropout)

        self.positionWiseFeedForward = PositionWiseFeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.norm2 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """一个encoder的前向传播

        Parameters
        ----------
        x : torch.Tensor
            每个encoderLayer的输入，shape=(batch_size, seq_len, embedding_dim)
        mask : torch.Tensor
            掩码
        """

        #========================================
        #             Attention                 #
        #========================================
        residual = x
        x = self.multiHeadAttention(x, x, x, mask)
        x = self.dropout1(x)
        # add & norm
        x = self.norm1(x + residual)

        #========================================
        #                  FFN                  #
        #========================================
        residual = x
        x = self.positionWiseFeedForward(x)
        x = self.dropout2(x)
        # add & norm
        x = self.norm2(x + residual)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_head: int, embedding_dim: int, hidden_dim: int, dropout: float):
        
        self.maskMultiHeadAttention = MultiHeadAttention(n_head=n_head, embedding_dim=embedding_dim)
        self.norm1 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout1 = nn.Dropout(p=dropout)

        self.multiHeadAttention = MultiHeadAttention(n_head=n_head, embedding_dim=embedding_dim)
        self.norm2 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout2 = nn.Dropout(p=dropout)

        self.positionWiseFeedForward = PositionWiseFeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.norm3 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, decoder_x: torch.Tensor, encoder_y: torch.Tensor, decoder_mask: torch.Tensor, encoder_mask: torch.Tensor):
        """decoder的一个block的过程

        Parameters
        ----------
        decoder_x : torch.Tensor
            shape=(batch_size, target_len, embedding_dim)
        encoder_y : torch.Tensor
            shape=(batch_size, source_len, embedding_dim)
        decoder_mask : torch.Tensor
            shape=(batch_size, target_len, target_len)
        encoder_mask : torch.Tensor
            shape=(batch_size, target_len, source_len)
        """

        # step1: 计算自注意力分数
        residual = decoder_x
        x = self.maskMultiHeadAttention(decoder_x, decoder_x, decoder_x, decoder_mask)

        # step2: add&norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        if encoder_y is not None:
            # step3: 计算encoder-decoder attention分数
            residual = encoder_y
            x = self.multiHeadAttention(query=x, key=encoder_y, value=encoder_y, mask=encoder_mask)
            
            # step4: add&norm
            x = self.dropout2(x)
            x = self.norm2(x + residual)

        # 5. fnn
        residual = x

        x = self.positionWiseFeedForward(x)
        
        # 6. add & norm
        x = self.dropout3(x)
        x = self.norm3(x + residual)

        return x


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

