# -*- coding: utf-8 -*-
"""
__title__="further_pretrain_data_process"
__author__="ngc7293"
__mtime__="2021/4/15"
"""
import os
from transformers.models.albert import modeling_albert
data_path = '/remote-home/zyfei/project/tianchi/data'

train_round1_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
test_round1_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

train_round2_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv'

vocab_file = 'vocab.txt'

raw_file_path = './raw_text/raw_text_roberta_base.txt'

# raw_file_path = os.path.join(data_path, raw_text)

model_name_or_path = "/remote-home/zyfei/project/tianchi/models/chinese-roberta-wwm-ext"

new_vocab_file = "./raw_text/roberta_base_vocab.txt"

# 统计词频
vocab_frequence = {}

with open(train_round1_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        rows = line.split('\t')[0:2]
        for sent in rows:
            for key in sent.split(' '):
                key = key.strip()
                vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

with open(train_round2_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        rows = line.split('\t')[0:2]
        for sent in rows:
            for key in sent.split(' '):
                key = key.strip()
                vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

with open(test_round1_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        rows = line.split('\t')[0:2]
        for sent in rows:
            for key in sent.split(' '):
                key = key.strip()
                vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

vocab_frequence = sorted(vocab_frequence.items(), key=lambda s: -s[1])

# 对齐字典 只能对齐大概
nezha_orgin_vocab = []
with open(os.path.join(model_name_or_path, vocab_file), encoding="utf-8") as f:
    for line in f.read().splitlines():
        line = line.strip()
        if line != '!':
            nezha_orgin_vocab.append(line)
        else:
            break

vocab = nezha_orgin_vocab + [key[0] for key in vocab_frequence]


# 不删除低频词
def load_data_pair_sent(path, result):
    if path.split(".")[-1] == "tsv":
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                result.append(line)
    else:
        raise NotImplementedError


train_result = []
test_result = []
# load_data_pair_sent(train_round1_path, train_result)
load_data_pair_sent(train_round2_path, train_result)
# load_data_pair_sent(test_round1_path, test_result)

all_result = train_result
# + test_result
with open(raw_file_path, 'w') as f:
    for key in all_result:
        f.write(str(key) + '\n')

with open(new_vocab_file, 'w') as f:
    for key in vocab:
        f.write(str(key) + '\n')
