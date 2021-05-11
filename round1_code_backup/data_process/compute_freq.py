import re
import os

import jieba
import pandas as pd
from get_word import get_mask_word, replace_id_to_mask
import tqdm


def get_word_list(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    regEx = re.compile(r'^w+$')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r"([\u4e00-\u9fa5])")  # [\u4e00-\u9fa5]中文范围

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) is None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                c = ch.split(" ")
                if len(c) > 1:
                    str1_list.extend(c)
                else:
                    str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符

    return list_word1


def load_data(filename, ids=True):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    if ids:
        with open(filename) as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    a, b, c = l[0], l[1], int(l[2])
                else:
                    a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
                a = [int(i) for i in a.split(' ')]
                b = [int(i) for i in b.split(' ')]
                D.append((a, b, c))
        return D
    else:
        with open(filename) as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    a, b, c = l[0], l[1], int(l[2])
                else:
                    a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
                a = list(jieba.cut(a))
                b = list(jieba.cut(b))
                D.append((a, b, c))
        return D


def update_seq(tokens, file_path=None, min_count=0, ids=True):
    assert file_path is not None
    data = load_data(file_path, ids=ids)
    for d in data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    tokens = {i: j for i, j in tokens.items() if j >= min_count}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    tokens = {t[0]: t[1] for i, t in enumerate(tokens)}
    return tokens


def update_head_seq(tokens, file_path=None, min_count=0, ids=True, id_tokens=None):
    assert file_path is not None
    data = load_data(file_path, ids=ids)
    for d in data:
        try:
            tokens[d[0][0]] = tokens.get(d[0][0], 0) + 1
        except:
            print(d[0])
        try:
            tokens[d[1][0]] = tokens.get(d[1][0], 0) + 1
        except:
            print(d[1])

    tokens = {i: j for i, j in tokens.items() if j >= min_count}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    tokens = {t[0]: t[1] for i, t in enumerate(tokens)}
    return tokens


def update_tail_seq(tokens, file_path=None, min_count=0, ids=True, id_tokens=None):
    assert file_path is not None
    data = load_data(file_path, ids=ids)
    for d in data:
        try:
            tokens[d[0][-1]] = tokens.get(d[0][-1], 0) + 1
        except:
            print(d[0])
        try:
            tokens[d[1][-1]] = tokens.get(d[1][-1], 0) + 1
        except:
            print(d[1])

    tokens = {i: j for i, j in tokens.items() if j >= min_count}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    tokens = {t[0]: t[1] for i, t in enumerate(tokens)}
    return tokens


data_path = "/remote-home/zyfei/project/tianchi/data"

# 未脱敏数据
train_1_name = "round1_train.tsv"
test_1_name = "round1_testA.tsv"

train_22_name = "gaiic_track3_round1_train_20210220.tsv"
test_22v2_name = "gaiic_track3_round1_testA_20210220v2.tsv"

# 脱敏数据
train_28_name = "gaiic_track3_round1_train_20210228.tsv"
test_28_name = "gaiic_track3_round1_testA_20210228.tsv"

id_tokens = {}
word_tokens = {}

id_tokens = update_seq(id_tokens, os.path.join(data_path, train_28_name))
id_tokens = update_seq(id_tokens, os.path.join(data_path, test_28_name))

word_tokens = update_seq(word_tokens, os.path.join(data_path, train_1_name), ids=False)
word_tokens = update_seq(word_tokens, os.path.join(data_path, test_1_name), ids=False)

id_head_tokens = {}
word_head_tokens = {}

id_head_tokens = update_head_seq(id_head_tokens, os.path.join(data_path, train_28_name), id_tokens=id_tokens)
id_head_tokens = update_head_seq(id_head_tokens, os.path.join(data_path, test_28_name), id_tokens=id_tokens)

word_head_tokens = update_head_seq(word_head_tokens, os.path.join(data_path, train_1_name), ids=False,
                                   id_tokens=word_tokens)
word_head_tokens = update_head_seq(word_head_tokens, os.path.join(data_path, test_1_name), ids=False,
                                   id_tokens=word_tokens)

id_tail_tokens = {}
word_tail_tokens = {}

id_tail_tokens = update_tail_seq(id_tail_tokens, os.path.join(data_path, train_28_name), id_tokens=id_tokens)
id_tail_tokens = update_tail_seq(id_tail_tokens, os.path.join(data_path, test_28_name), id_tokens=id_tokens)

word_tail_tokens = update_tail_seq(word_tail_tokens, os.path.join(data_path, train_1_name), ids=False,
                                   id_tokens=word_tokens)
word_tail_tokens = update_tail_seq(word_tail_tokens, os.path.join(data_path, test_1_name), ids=False,
                                   id_tokens=word_tokens)

vocab = {}
word_head_list = list(word_head_tokens.items())[::-1]

print(len(word_head_list))

for i, t in enumerate(id_head_tokens):
    if t in vocab.values():
        continue
    if i > 10:
        break
    while True:
        word = word_head_list.pop()
        word_num = int(word[1])
        word = word[0]
        w_id = t
        if word in vocab.keys() or word_num < 2000:
            continue
        else:
            vocab[word] = w_id
            break

word_tail_list = list(word_tail_tokens.items())[::-1]
for i, t in enumerate(id_tail_tokens):
    if t in vocab.values():
        continue
    if i > 10:
        break
    while True:
        word = word_tail_list.pop()
        word_num = int(word[1])
        word = word[0]
        w_id = t
        if word in vocab.keys() or word_num < 1000:
            continue
        else:
            vocab[word] = w_id
            break

word_list = list(word_tokens.items())[::-1]
for i, t in enumerate(id_tokens.items()):
    if t[0] in vocab.values():
        continue
    if t[1] < 5000:
        break
    while True:
        word = word_list.pop()[0]
        w_id = t[0]
        if word in vocab.keys():
            continue
        else:
            vocab[word] = w_id
            break

word_to_id_vocab = {j: i for i, j in vocab.items()}

data_path = "/remote-home/zyfei/project/tianchi/data"

train_data = load_data(os.path.join(data_path, train_28_name))
test_data = load_data(os.path.join(data_path, test_28_name))

replace_train_data = []

replace_train_28_name = "replace_train_20210228.csv"
replace_test_28_name = "replace_testA_20210228.csv"

for text_1, text_2, _ in train_data:
    text_1 = [word_to_id_vocab.get(i, i) for i in text_1]
    text_2 = [word_to_id_vocab.get(i, i) for i in text_2]
    replace_train_data.append(text_1)
    replace_train_data.append(text_2)

for text_1, text_2, _ in test_data:
    text_1 = [word_to_id_vocab.get(i, i) for i in text_1]
    text_2 = [word_to_id_vocab.get(i, i) for i in text_2]
    replace_train_data.append(text_1)
    replace_train_data.append(text_2)

real_vocab = {}
for i in vocab:
    real_vocab[vocab[i]] = {i: 1}

import tqdm

for text in tqdm.tqdm(replace_train_data):
    count = 0
    for i in text:
        if isinstance(i, int):
            count += 1
    if count >= 4:
        continue

    a = replace_id_to_mask(text)

    for j, k in zip(a[0], a[1]):
        word_dict = get_mask_word(text_list=j, id_list=k)

        for i in word_dict:
            if i not in real_vocab:
                real_vocab[i] = {}
            if word_dict[i] not in real_vocab[i]:
                real_vocab[i][word_dict[i]] = 1
            else:
                real_vocab[i][word_dict[i]] += 1

import json

with open("./output", "w") as fp:
    fp.write(json.dumps(real_vocab))


def get_most_vocab(vocab):
    """
    取出vocab中频数最大的字。
    :param vocab:{'xxx':x,'xxx':x}
    :return:
    """
    a = sorted(vocab.items(), lambda s: -s[1])
    print(a)
