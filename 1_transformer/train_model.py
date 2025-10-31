import os
import math
import copy
import argparse
import time
import warnings
import numpy as np
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import funtional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import Transformer
from dataset import MTDataset
from utils import get_attn_pad_mask, get_subsequent_mask, compute_bleu

warnings.filterwarnings("ignore")

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.optimizer, criterion: nn.Module, device: torch.device, pad_idx: int):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (src_seq, tgt_seq, tgt_seq_y, tgt_lens) in enumerate(tqdm(dataloader, desc="Training")):
        batch_count += 1

        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)
        tgt_seq_y = tgt_seq_y.to(device)

        # 创建掩码
        enc_mask = get_attn_pad_mask(src_seq, src_seq, pad_idx)
        dec_self_mask = get_attn_pad_mask(tgt_seq, tgt_seq, pad_idx) & get_subsequent_mask(tgt_seq)
        # 这个和enc_mask有什么区别呢？
        dec_enc_mask = get_attn_pad_mask(tgt_seq, src_seq, pad_idx)

        # 前向传播
        output = model(src_seq, tgt_seq, enc_mask, dec_self_mask, dec_enc_mask)

        # 计算损失
        output = output.contiguous().view(-1, output.size(-1))
        tgt_seq_y = tgt_seq_y.contiguous().view(-1)
        loss = criterion(output, tgt_seq_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / batch_count

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, pad_idx: int, max_len: int, index2word_zh: Dict, index2word_en: Dict):
    model.eval()
    total_loss = 0
    batch_count = 0
    bleu_scores = []

    with torch.no_grad():
        for batch_idx, (src_seq, tgt_seq, tgt_seq_y, tgt_lens) in enumerate(tqdm(dataloader, desc="Validating")):
            batch_count += 1

            src_seq = src_seq.to(device)
            tgt_seq = tgt_seq.to(device)
            tgt_seq_y = tgt_seq_y.to(device)

            # 创建掩码
            enc_mask = get_attn_pad_mask(src_seq, src_seq, pad_idx)
            dec_self_mask = get_attn_pad_mask(tgt_seq, tgt_seq, pad_idx) & get_subsequent_mask(tgt_seq)
            dec_enc_mask = get_attn_pad_mask(tgt_seq, src_seq, pad_idx)

            # 前向传播
            output = model(src_seq, tgt_seq, enc_mask, dec_self_mask, dec_enc_mask)
            
            # 计算损失
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_seq_y_flat = tgt_seq_y.contiguous().view(-1)
            loss = criterion(output_flat, tgt_seq_y_flat)
            total_loss += loss.item()

            # 推理模式计算BLEU
            translations = generate_translations(model, src_seq, enc_mask, device, pad_idx, max_len)
            bleu_scores.extend(compute_bleu(translations, tgt_seq_y, tgt_lens))

            # 打印示例
            if batch_idx % 10 == 0:
                print_example(src_seq[0], translations[0], tgt_seq_y[0], index2word_en, index2word_zh, tgt_lens[0])
            
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    return total_loss / batch_count, avg_bleu

def generate_translations(model, enc_x, enc_mask, device, pad_idx, max_len=50):
    batch_size = enc_x.size(0)
    
    # 初始化解码器输入（开始符号）
    dec_x = torch.ones(batch_size, 1).fill_(0).long().to(device)  # 假设0是开始符号
    
    # 编码
    memory = model.encode(enc_x, enc_mask)
    
    # 自回归生成
    for i in range(max_len - 1):
        dec_self_attn_mask = get_attn_pad_mask(dec_x, dec_x, pad_idx) | get_subsequent_mask(dec_x)
        enc_dec_attn_mask = get_attn_pad_mask(dec_x, enc_x, pad_idx)
        
        output = model.decode(dec_x, memory, dec_self_attn_mask, enc_dec_attn_mask)
        prob = model.generator(output[:, -1:])
        _, next_word = torch.max(prob, dim=-1)
        
        dec_x = torch.cat([dec_x, next_word], dim=1)
        
        # 如果所有序列都生成了结束符，提前停止
        if (next_word == 1).all():  # 假设1是结束符
            break
    
    return dec_x

def print_example(enc_seq, trans_seq, ref_seq, enc_len, ref_len, index2word_en, index2word_zh):
    # 获取原文
    enc_words = []
    for i in range(min(enc_len, len(enc_seq))):
        word = index2word_en.get(enc_seq[i].item(), '<UNK>')
        if word == '<EOS>':
            break
        enc_words.append(word)
    
    # 获取机翻译文
    trans_words = []
    for i in range(len(trans_seq)):
        word = index2word_zh.get(trans_seq[i].item(), '<UNK>')
        if word == '<EOS>':
            break
        trans_words.append(word)
    
    # 获取参考译文
    ref_words = []
    for i in range(min(ref_len, len(ref_seq))):
        word = index2word_zh.get(ref_seq[i].item(), '<UNK>')
        if word == '<EOS>':
            break
        ref_words.append(word)

    print(f"原文: {' '.join(enc_words)}")
    print(f"机翻译文: {''.join(trans_words)}")
    print(f"参考译文: {''.join(ref_words)}")
    print("-" * 50)

def test_model(model, test_loader, device, pad_idx, index2word_zh, index2word_en, output_file=None):
    model.eval()
    bleu_scores = []
    
    if output_file:
        fp = open(output_file, 'w', encoding='utf-8')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            enc_x, dec_x, dec_y, enc_x_lens, dec_y_lens = batch
            
            enc_x = enc_x.to(device)
            dec_y = dec_y.to(device)
            
            enc_mask = get_attn_pad_mask(enc_x, enc_x, pad_idx)
            translations = generate_translations(model, enc_x, enc_mask, device, pad_idx)
            
            batch_bleu_scores = compute_bleu(translations, dec_y, dec_y_lens)
            bleu_scores.extend(batch_bleu_scores)
            
            # 输出所有测试结果
            for i in range(len(enc_x)):
                enc_seq = enc_x[i]
                trans_seq = translations[i]
                ref_seq = dec_y[i]
                enc_len = enc_x_lens[i]
                ref_len = dec_y_lens[i]
                
                print_example(enc_seq, trans_seq, ref_seq, enc_len, ref_len, index2word_en, index2word_zh)
                
                if output_file:
                    # 获取原文
                    enc_words = []
                    for j in range(min(enc_len, len(enc_seq))):
                        word = index2word_en.get(enc_seq[j].item(), '<UNK>')
                        if word == '<EOS>':
                            break
                        enc_words.append(word)
                    
                    # 获取机翻译文
                    trans_words = []
                    for j in range(len(trans_seq)):
                        word = index2word_zh.get(trans_seq[j].item(), '<UNK>')
                        if word == '<EOS>':
                            break
                        trans_words.append(word)
                    
                    # 获取参考译文
                    ref_words = []
                    for j in range(min(ref_len, len(ref_seq))):
                        word = index2word_zh.get(ref_seq[j].item(), '<UNK>')
                        if word == '<EOS>':
                            break
                        ref_words.append(word)
                    
                    fp.write(f"原文: {' '.join(enc_words)}\n")
                    fp.write(f"机翻译文: {''.join(trans_words)}\n")
                    fp.write(f"参考译文: {''.join(ref_words)}\n")
                    fp.write("-" * 50 + "\n")
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    print(f"测试集平均BLEU分数: {avg_bleu:.4f}")
    
    if output_file:
        fp.write(f"测试集平均BLEU分数: {avg_bleu:.4f}\n")
        fp.close()
    
    return avg_bleu

def main(args):
    # 指定device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
        if args.gpu >= 0:
            print(f"Warning: GPU {args.gpu} is not available, using CPU instead")
    
    print(f"Using device: {device}")

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

    print(f"中文词汇表大小: {zhVocabLen}")
    print(f"英文词汇表大小: {enVocabLen}")

    index2word_zh = np.load("./data/index2word_zh.npy", allow_pickle=True).item()
    index2word_en = np.load("./data/index2word_en.npy", allow_pickle=True).item()

    model = Transformer(
        n_layers=args.n_layers,
        n_head=args.head_num,
        embedding_dim=args.embedding_dim,
        src_vocab_size=enVocabLen,
        tgt_vocab_size=zhVocabLen,
        hidden_dim=args.hidden_num,
        max_len=args.n_padding,
        dropout=args.dropout
    )

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.NLLLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    # 训练日志
    log_file = os.path.join(args.log_dir, f"transformer_train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    best_bleu = 0
    train_losses = []
    valid_losses = []
    bleu_scores = []

    if args.mode == "train":
        print("开始训练...")
        with open(log_file, "w") as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters: {vars(args)}\n\n")

            for epoch in range(args.num_epochs):
                print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                f.write(f"\nEpoch {epoch+1}/{args.num_epochs}\n")

                # 训练
                train_loss = train_epoch(model, trainDataLoader, optimizer, criterion, device, pad_idx=3)
                train_losses.append(train_loss)

                # 验证
                valid_loss, bleu_score = validate_epoch(model, validDataLoader, criterion, device, 
                                                       pad_idx=3, max_len=args.n_padding, index2word_zh=index2word_zh, 
                                                       index2word_en=index2word_en)
                valid_losses.append(valid_loss)
                bleu_scores.append(bleu_score)

                # 打印和记录结果
                epoch_log = (f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                           f"Valid Loss: {valid_loss:.4f}, BLEU: {bleu_score:.4f}")
                print(epoch_log)
                f.write(epoch_log + "\n")

                # 保存最佳模型
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    model_dir = "../data/model"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, 
                                            f"transformer_best_bleu_{bleu_score:.4f}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"保存最佳模型，BLEU: {bleu_score:.4f}")
                    f.write(f"保存最佳模型，BLEU: {bleu_score:.4f}\n")

    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("测试模式需要提供模型路径 --model_path")
        
        print("开始测试...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        test_output_file = os.path.join(args.log_dir, f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        test_bleu = test_model(model, testDataLoader, device, pad_idx=3, 
                              index2word_zh=index2word_zh, index2word_en=index2word_en,
                              output_file=test_output_file)
        
        print(f"测试完成! BLEU分数: {test_bleu:.4f}")

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
    parser.add_argument("--n_layers", default=6, type=int, help="number of encoder/decoder layers")
    parser.add_argument("--n_padding", default=50, type=int, help="set the padding length")
    parser.add_argument("--model_path", type=str, help="refer to thr trained model path")
    parser.add_argument("--log_dir", default="./data/log", type=str, help="log directory")
    parser.add_argument("--model_dir", default="./data/model", type=str, help="model save directory")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    main(args)
