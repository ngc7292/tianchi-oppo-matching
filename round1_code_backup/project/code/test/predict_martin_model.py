# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/3/28"
"""
import numpy as np

from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description="input test file's path")

parser.add_argument("--testfile", type=str)

args = parser.parse_args()

test_path = args.testfile

import torch
from martin.modeling_nezha import NeZhaForSequenceClassification
from martin.configuration_nezha import NeZhaConfig
from transformers import DataCollatorWithPadding
from transformers import BertTokenizer
from torch.utils.data import DataLoader

vocab_file = './martin/tokens_ngram.txt'
config_file = '../train/martin/mynezha_ngram/config.json'
model_file = '../train/martin/mynezha_ngram'

def load_data(path):
    max_length = 0
    samples = []
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            temp = line.split('\t')
            new_line =temp[0] +' [SEP] ' + temp[1]
            samples.append(new_line)
            if len(temp)>2:
                labels.append(int(temp[2]))
    return np.array(samples),np.array(labels),max_length


class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels,tokenizer):
        self.texts = texts
        self.labels = labels
    def __getitem__(self, idx):
        examples = tokenizer(self.texts[idx], add_special_tokens=True, truncation=True)
        if isinstance(self.labels,np.ndarray):
            examples['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return examples
    def __len__(self):
        return len(self.texts)
test_texts, _, max_length = load_data(test_path)
tokenizer = BertTokenizer(vocab_file=vocab_file)
config = NeZhaConfig.from_pretrained(config_file,num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer)
test_dataset = MatchingDataset(test_texts,None,tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=data_collator)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_test(model,test_loader):
    model.eval() 
    y_pred = []
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids,attention_mask)[0]
            output = output.cpu().numpy()
            y_pred.append(output)
    y_pred = np.concatenate(y_pred)
    y_pred = np.exp(y_pred) / (np.exp(y_pred).sum(axis=1, keepdims=True)) # softmax
    #y_pred = y_pred[:, 1] / y_pred.sum(axis=1)
    return y_pred

model = NeZhaForSequenceClassification.from_pretrained(model_file, config=config)
model.to(device)

model_pred = np.zeros((len(test_texts), 2))
for fold in range(5):
    model.load_state_dict(torch.load(f'./martin/fold_{fold}.pkl'))
    model_pred += predict_test(model,test_loader)
model_pred = model_pred / 5
model_pred = model_pred[:, 1] / model_pred.sum(axis=1)

with open('../../prediction_result/result-du.txt', 'w') as f:
    for val in model_pred:
        f.write(str(val) + '\n')
