# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/4/19"
"""

import os
import torch
import time
import scipy
import onnxruntime
import numpy as np


from scipy.special import log_softmax, softmax
from flask import Flask, request

from transformers import BertTokenizer, BertTokenizerFast

app = Flask(__name__)

import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# 正式测试，batch_size固定为1
@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    data = request.get_data()
    if data == b"exit":
        print("received exit command, exit now")
        os._exit(0)

    input_list = request.form.getlist("input")
    index_list = request.form.getlist("index")

    response_batch = {}
    response_batch["results"] = []
    for i in range(len(index_list)):
        index_str = index_list[i]

        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()
            predict = infer(session_1, session_2, tokenizer, query_A, query_B)
            response["predict"] = str(predict)
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return response_batch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 需要根据模型类型重写
def infer(session_1, session_2, tokenizer, text_1, text_2):
    start_time = time.time()
    sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
        " ")
    sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
    # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
    # 1 is word in other sentence and 0 is not in other sentence
    co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
        1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)

    ort_inputs = {session_1.get_inputs()[0].name: [sample.input_ids],
                  session_1.get_inputs()[1].name: [sample.token_type_ids],
                  session_1.get_inputs()[2].name: [co_ocurrence_list]}
    print("data process")
    print(time.time() - start_time)

    start_time = time.time()
    output_1 = session_1.run(None, ort_inputs)[0]
    print("session_1 time")
    print(time.time() - start_time)
    start_time = time.time()
    # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
    output_score_1 = np.exp(output_1) / (np.exp(output_1).sum(axis=1, keepdims=True))
    result_1 = output_score_1[:, 1] / output_score_1.sum(axis=1)
    print("softmax_time")
    print(time.time() - start_time)

    start_time = time.time()
    output_2 = session_2.run(None, ort_inputs)[0]
    print("session_2 time")
    print(time.time() - start_time)

    start_time = time.time()
    # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
    output_score_2 = np.exp(output_2) / (np.exp(output_2).sum(axis=1, keepdims=True))
    result_2 = output_score_2[:, 1] / output_score_2.sum(axis=1)

    print("softmax_time")
    print(time.time() - start_time)

    start_time = time.time()
    result_2_softmax = softmax(output_2[0])
    print("scipy softmax:")
    print(time.time() - start_time)

    start_time = time.time()
    result_2_logsoftmax = log_softmax(output_2[0])
    print("scipy log softmax:")
    print(time.time() - start_time)


    print(result_1)
    print(result_2)
    print(result_2_softmax)
    print(result_2_logsoftmax)
    result = (result_1[0] + result_2[0]) / 2

    return result


# 需要根据模型类型重写
def init_model(tokenizer_path):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    session_1 = onnxruntime.InferenceSession(
        "/remote-home/zyfei/project/tianchi/round2-code/nezha_5_2_3/fold_co_1/fold-0.onnx")
    session_2 = onnxruntime.InferenceSession(
        "./test.onnx")
    # session_3 = onnxruntime.InferenceSession(
    #     "/remote-home/zyfei/project/tianchi/round2-code/nezha_5_2_1/fold_co_3/fold-2.onnx")

    return tokenizer, session_1, session_2


if __name__ == "__main__":
    tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30_v2_round2data"
    tokenizer, session_1, session_2 = init_model(tokenizer_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(model)
    app.run(host="0.0.0.0", port=8080)
