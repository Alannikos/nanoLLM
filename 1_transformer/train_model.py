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

    
