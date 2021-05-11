# -*- coding: utf-8 -*-
"""
__title__="fineturn_ngram"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import os
import torch
import numpy as np
import pandas as pd

from transformers import LineByLineTextDataset
from modeling_nezha import NeZhaForSequenceClassification, NeZhaForMaskedLM, NeZhaPreTrainedModel
from configuration_nezha import NeZhaConfig

train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

vocab_file = './tokens.txt'

raw_text = './raw_text_ngram.txt'

# 统计词频
vocab_frequence = {}

with open(train_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        rows = line.split('\t')[0:2]
        for sent in rows:
            for key in sent.split(' '):
                key = key.strip()
                vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

with open(test_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        rows = line.split('\t')[0:2]
        for sent in rows:
            for key in sent.split(' '):
                key = key.strip()
                vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

vocab_frequence = sorted(vocab_frequence.items(), key=lambda s: -s[1])

nezha_orgin_vocab = []
with open('/remote-home/zyfei/project/tianchi/models/nezha-large-www/vocab.txt', encoding="utf-8") as f:
    for line in f.read().splitlines():
        line = line.strip()
        if line != '你':
            nezha_orgin_vocab.append(line)
        else:
            break

vocab = nezha_orgin_vocab + [key[0] for key in vocab_frequence]


# 不删除低频词
def load_data_pair_sent(path, result):
    if path.split(".")[-1] == "tsv":
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                rows = line.split('\t')[0:2]
                a = []
                for key in rows[0].split(' '):
                    key = key.strip()
                    a.append(key)
                b = []
                for key in rows[1].split(' '):
                    key = key.strip()
                    b.append(key)
                result.append(' '.join(a) + ' [SEP] ' + ' '.join(b))
                result.append(' '.join(b) + ' [SEP] ' + ' '.join(a))
    elif path.split(".")[-1] == "csv":
        data_frame = pd.read_csv(path)
        for index, data in data_frame.iterrows():
            result.append(data[1] + ' [SEP] ' + data[2])
            result.append(data[2] + ' [SEP] ' + data[1])

    else:
        raise NotImplementedError


train_result = []
test_result = []
load_data_pair_sent(train_path, train_result)
load_data_pair_sent(test_path, test_result)

all_result = train_result + test_result
with open(raw_text, 'w') as f:
    for key in all_result:
        f.write(str(key) + '\n')

vocab = vocab[:21128]
with open(vocab_file, 'w') as f:
    for key in vocab:
        f.write(str(key) + '\n')
