# -*- coding: utf-8 -*-
"""
__title__="test_data"
__author__="ngc7293"
__mtime__="2021/4/26"
"""
import tqdm


def load_data_fastnlp(path):
    true_samples = []
    false_samples = []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            samples.append([temp[0], temp[1], int(temp[2])])
            if temp[2] == "0":
                false_samples.append("\t".join(temp))
            else:
                true_samples.append("\t".join(temp))
    return samples, true_samples, false_samples


round1_data, _, round1_train_data = load_data_fastnlp("/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv")
round2_data, true_samples, false_samples = load_data_fastnlp(
    "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv")

train_data = []
train_set = set()
dev_set = set()
true_set = set()
false_set = set()

for i in true_samples[:50000]:
    true_set.add(i)

for i in true_samples[50000:]:
    if i not in true_set:
        train_set.add(i)

for i in false_samples:
    if i not in train_set:
        train_set.add(i)


for i in round1_train_data:
    if i not in train_set:
        false_set.add(i)

dev_list = []
count = 0
for i in list(true_set):
    dev_list.append(i)
    count += 1
    if count >= 50000:
        break

count = 0
for i in list(false_set):
    dev_list.append(i)
    count += 1
    if count >= 50000:
        break

with open("./data/dev.tsv", "w",encoding="utf-8") as f:
    for i in dev_list:
        f.write(i+"\n")

train_list = list(train_set)

samples = []
labels = []
for temp in train_list:
    temp = temp.split("\t")
    samples.append([temp[0], temp[1]])
    labels.append(int(temp[2]))

import networkx as nx

pos_G = nx.Graph()
neg_G = nx.Graph()

for index, label in enumerate(labels):
    if label == 1:
        pos_G.add_edge(samples[index][0], samples[index][1])
    else:
        neg_G.add_edge(samples[index][0], samples[index][1])

pos_dataset = []
neg_dataset = []
for c in nx.connected_components(pos_G):
    nodes = list(c)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pos_dataset.append([nodes[i], nodes[j]])
        if nodes[i] in neg_G:
            for neiber_neg in neg_G.adj[nodes[i]]:
                for z in range(len(nodes)):
                    if z != i:
                        neg_dataset.append([neiber_neg, nodes[z]])
pos_samples = []
pos_labels = []
for i in pos_dataset:
    pos_samples.append(i[0], i[1])
    pos_labels.append(1)

neg_samples = []
neg_labels = []
for i in neg_dataset:
    neg_samples.append(i[0], i[1])
    neg_labels.append(0)

from sklearn.model_selection import train_test_split

pos_samples, _, pos_labels, _ = train_test_split(pos_samples,pos_labels, train_size=150000, shuffle=True)
neg_samples, _, neg_labels, _ = train_test_split(neg_samples,neg_labels, train_size=150000, shuffle=True)

train_data = []
for index, sample in enumerate(pos_samples):
    train_data.append(sample[0]+"\t"+sample[1]+"\t1")

for index, sample in enumerate(neg_samples):
    train_data.append(sample[0]+"\t"+sample[1]+"\t0")


