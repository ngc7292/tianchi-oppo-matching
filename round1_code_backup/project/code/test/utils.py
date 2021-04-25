# -*- coding: utf-8 -*-
"""
__title__="utils"
__author__="ngc7293"
__mtime__="2021/4/7"
"""
import torch
import numpy as np
import pandas as pd

class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx][0] + ' [SEP] ' + self.texts[idx][1]

        sentence_list1, sentence_list2 = text.split("[SEP]")
        sentence_list1, sentence_list2 = sentence_list1.strip().split(" "), sentence_list2.strip().split(
            " ")
        sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
        # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
        # 1 is word in other sentence and 0 is not in other sentence
        co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
            1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]

        examples = self.tokenizer(text, add_special_tokens=True, truncation=True)
        if isinstance(self.labels, np.ndarray):
            examples['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        examples['token_type_ids'] = co_ocurrence_list

        return examples

    def __len__(self):
        return len(self.texts)


def load_data(path):
    max_length = 0
    samples = []
    labels = []
    if path.split(".")[-1] == "tsv":
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                temp = line.split('\t')
                samples.append([temp[0], temp[1]])
                if len(temp) > 2:
                    labels.append(int(temp[2]))
        return np.array(samples), np.array(labels), max_length
    elif path.split(".")[-1] == "csv":
        lines = pd.read_csv(path)
        for i, data in lines.iterrows():
            samples.append([data[1], data[2]])
            if len(data) > 3:
                labels.append(int(data[3]))
        return np.array(samples), np.array(labels), max_length