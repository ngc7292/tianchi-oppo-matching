# -*- coding: utf-8 -*-
"""
__title__="parpare_data"
__author__="ngc7293"
__mtime__="2021/3/17"
"""
train_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"
test_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv"

vocab_data_path = "./vocab.txt"

raw_text = './raw_text.txt'

print("loading train data...")
train_mlm_data = []
vocab = set()
with open(train_data_path, encoding="utf-8") as f:
    for line in f.readlines():
        rows = line.split('\t')
        for sent in rows[0:2]:
            vocab.update(sent.split(' '))
        train_mlm_data.append(rows[0] + ' [SEP] ' + rows[1])
        train_mlm_data.append(rows[1] + ' [SEP] ' + rows[0])

print("loading test data...")
test_mlm_data = []
with open(test_data_path, encoding="utf-8") as f:
    for line in f.readlines():
        rows = line.replace("\n","").split('\t')
        for sent in rows[0:2]:
            vocab.update(sent.split(' '))
        test_mlm_data.append(rows[0] + ' [SEP] ' + rows[1])
        test_mlm_data.append(rows[1] + ' [SEP] ' + rows[0])


print("save pretrain data...")
all_mlm_data = train_mlm_data + test_mlm_data
with open(raw_text, 'w') as f:
    for key in all_mlm_data:
        f.write(str(key) + '\n')

vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(vocab)

print("save vocab data...")
with open(vocab_data_path, "w") as fp:
    for key in vocab:
        fp.write(str(key) + "\n")
