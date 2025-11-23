import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


class MTDataLoader():
    def __init__(
        self,
        model_name: str,
        max_len: int,
        batch_size: int,
        sos_token: str
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.sos_token = sos_token    # 起始标记

        self.tokenizer = AutoTokenizer(self.model_name)

        print('dataloader initializing start')

    def make_dataset(
        self,
        dataset_name: str = "bentrevett/multi30k"
    ):
        dataset = load_dataset(dataset_name)
        print('Dataset loaded')

        train_data = dataset['train']      # 训练集
        valid_data = dataset['validation'] # 验证集  
        test_data = dataset['test']        # 测试集

        return train_data, valid_data, test_data

    def tokenize_function(
        self,
        example
    ):
        source_text = [text for text in example['de']]
        target_text = [self.sos_token + text for text in example['en']]

        input_ids = self.tokenizer(source_text, padding="max_length", truncation=True, max_length=self.max_len)["input_ids"]
        labels = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_len)['input_ids']

        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def make_iter(
        self,
        train_data,
        valid_data,
        test_data
    ):
        train_data = train_data.map(self.tokenize_function, batched=True, num_proc=6)
        valid_data = valid_data.map(self.tokenize_function, batched=True, num_proc=6)
        test_data = test_data.map(self.tokenize_function, batched=True, num_proc=6)

        train_data.set_format(type='torch', columns=['input_ids', 'labels'])
        valid_data.set_format(type='torch', columns=['input_ids', 'labels'])
        test_data.set_format(type='torch', columns=['input_ids', 'labels'])

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)
        valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=6)

        print('Data Initializing done')
        
        return train_dataloader, valid_dataloader, test_dataloader