# -*- coding: utf-8 -*-
"""
__title__="fineturning_nezha"
__author__="ngc7293"
__mtime__="2021/4/16"
"""
import os
import time
import torch
import tqdm
import random
import transformers
import fitlog
import numpy as np
import pandas as pd
import networkx as nx
import argparse

# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
from fastNLP import DataSet as FastNLPDataSet
from fastNLP.core import Instance, DataSetIter, RandomSampler, cache_results
from fastNLP.core.metrics import AccuracyMetric
from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
from modeling_ensamble import EnsambleModel
from transformers import BertTokenizer

fitlog.set_log_dir("/remote-home/zyfei/project/tianchi/round2-code/logs")


def tokenize_data(text_1, text_2, tokenizer):
    sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
        " ")
    sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
    # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
    # 1 is word in other sentence and 0 is not in other sentence
    co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
        1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)
    return {
        "input_ids": sample.input_ids,
        "token_type_ids": sample.token_type_ids,
        "attention_mask": sample.attention_mask,
        "co_ocurrence_ids": co_ocurrence_list
    }


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-17-with-label-fineturning", _refresh=False)
def load_data_fastnlp(path, tokenizer):
    ds = FastNLPDataSet()
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            if len(temp) > 2:
                data_1 = tokenize_data(temp[0], temp[1], tokenizer)
                ds.append(Instance(input_ids=data_1["input_ids"], token_type_ids=data_1["token_type_ids"],
                                   attention_mask=data_1["attention_mask"], co_ocurrence_ids=data_1["co_ocurrence_ids"],
                                   label=int(temp[2])))
    ds.set_input("input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids", "label")
    ds.set_target("label")
    return ds


def evaluate(eval_model, val_dataset, device, batch_size=128):
    eval_model.eval()
    y_pred = []
    mac_pred = []
    nezha_pred = []
    e_pred = []
    y_true = []
    sampler = RandomSampler()
    val_loader = DataSetIter(val_dataset, batch_size=batch_size, sampler=sampler)
    with torch.no_grad():
        for batch, _ in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)

            mac_output, nezha_output, output = eval_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, co_ocurrence_ids=co_ocurrence_ids)

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            y_pred.append(output_score[:, 1] / output_score.sum(axis=1))

            mac_output = mac_output.cpu().numpy()
            # mac_output_score = np.exp(mac_output[:, 1])/ (np.exp(mac_output).sum(axis=1))
            mac_output_score = np.exp(mac_output) / (np.exp(mac_output).sum(axis=1, keepdims=True))
            mac_output = mac_output_score[:, 1] / mac_output_score.sum(axis=1)
            mac_pred.append(mac_output)

            nezha_output = nezha_output.cpu().numpy()
            # nezha_output_score = np.exp(nezha_output[:, 1])/ (np.exp(nezha_output).sum(axis=1))
            nezha_output_score = np.exp(nezha_output) / (np.exp(nezha_output).sum(axis=1, keepdims=True))
            nezha_output = nezha_output_score[:, 1] / nezha_output_score.sum(axis=1)
            nezha_pred.append(nezha_output)

            e_pred.append((nezha_output+mac_output)/2)

            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    nezha_pred = np.concatenate(nezha_pred)
    mac_pred = np.concatenate(mac_pred)
    e_pred = np.concatenate(e_pred)

    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true,mac_pred), roc_auc_score(y_true, nezha_pred), roc_auc_score(y_true, y_pred), roc_auc_score(y_true, e_pred)


def run():
    transformers.logging.set_verbosity_error()

    # tokenizer_file = '/remote-home/zyfei/project/tianchi/model_output/nezha_output_2'
    tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"

    test_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"
    fold_path = "./model_ensamble_1"

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    nezha_name_or_path = "./model_19"
    macbert_name_or_path = "./model_22"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dev_dataset = load_data_fastnlp(test_path, tokenizer,
                                    _cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-17-with-label-fineturning-dev",
                                    _refresh=False)

    model = EnsambleModel(macbert_path=macbert_name_or_path, nezha_path=nezha_name_or_path)

    model.to(device)

    # fitlog.add_hyper(attack_model.name, "attack_model")
    tokenizer.save_pretrained(fold_path)
    torch.save(model, os.path.join(fold_path, "pytorch.bin"))

    result = evaluate(model, dev_dataset, device, batch_size=128)

    print(f"ensamble model auc is :")
    print(result)


if __name__ == '__main__':
    run()
