# -*- coding: utf-8 -*-
"""
__title__="send_data"
__author__="ngc7293"
__mtime__="2021/4/19"
"""
import requests
import json
import time
import tqdm
import numpy as np

from fastNLP.core import  cache_results
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def send_eval(data_json, log):
    url = "http://127.0.0.1:8080/tccapi"
    try:
        start = time.time()
        res = requests.post(url=url, data=data_json)
        cost_time = time.time() - start
        res = json.loads(res.text)
        # res = json.dumps(res_batch)
    except Exception as e:
        index_list = data_json["index"]

        res_batch = {}
        res_batch["results"] = []
        for index_str in index_list:
            res_elem = {}
            res_elem["ok"] = False
            res_elem["index"] = index_str
            res_elem["predict"] = 0
            res_elem["msg"] = "get result from tccapi failed, set default predict to 0"
            res_batch["results"].append(res_elem)
        res = json.dumps(res_batch)
    return res, cost_time


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-22-with-label-predict", _refresh=False)
def load_data_fastnlp(path):
    samples = []
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            samples.append(temp[0] + "\t" + temp[1])
            labels.append(int(temp[2]))
    return samples, labels


if __name__ == '__main__':
    # train_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv"
    # train_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"
    train_path = "./data/dev.tsv"

    samples, labels = load_data_fastnlp(train_path)

    train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=5000, random_state=2021, shuffle=True)

    count = 1
    times = []
    preds = []
    labels = []
    for sample, label in tqdm.tqdm(zip(test_samples, test_labels)):
        data = {'input': [sample], 'index': [count]}
        res, send_time = send_eval(data, None)
        count += 1
        pred = float(res["results"][0]["predict"])
        preds.append(pred)
        labels.append(label)
        # print(res)
        times.append(send_time)

    print(roc_auc_score(labels, preds))
    print(np.mean(times))
    print(1)