# -*- coding: utf-8 -*-
"""
__title__="enhance_train_data"
__author__="ngc7293"
__mtime__="2021/4/3"
"""
import numpy
import pandas as pd

samples = []
labels = []

path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"

with open(path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        temp = line.split('\t')
        samples.append([temp[0], temp[1]])
        if len(temp) > 2:
            labels.append(int(temp[2]))

enhance_path = "/remote-home/source/cxan/tianchi/samples_for_training.txt"

with open(enhance_path, encoding="utf-8") as f:
    for line in f.read().splitlines():
        temp = line.split(',')
        samples.append([temp[0], temp[1]])
        labels.append(1)

enhance_data_path = "/remote-home/zyfei/project/tianchi/data/enhance_train_cx.csv"

df = pd.DataFrame([samples, labels])
df.to_csv(enhance_data_path)