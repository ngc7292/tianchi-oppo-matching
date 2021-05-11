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
            predict = infer(session_1, tokenizer, query_A, query_B)
            response["predict"] = str(predict)
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            print(e)
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return response_batch


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 需要根据模型类型重写
def infer(session_1, tokenizer, text_1, text_2):
    # start_time = time.time()
    text = text_1 + " [SEP] " + text_2
    sample = tokenizer(text=text, add_special_tokens=True, truncation=True)

    ort_inputs = {session_1.get_inputs()[0].name: [sample.input_ids],
                  session_1.get_inputs()[1].name: [sample.attention_mask]}
    # print("data process")
    # print(time.time() - start_time)

    # start_time = time.time()
    output_1 = session_1.run(None, ort_inputs)[0]
    # print("session_1 time")
    # print(time.time() - start_time)

    # start_time = time.time()
    # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
    output_score_1 = np.exp(output_1) / (np.exp(output_1).sum(axis=1, keepdims=True))
    result_1 = output_score_1[:, 1] / output_score_1.sum(axis=1)
    # print("softmax_time")
    # print(time.time() - start_time)
    #
    # print(result_1)
    return result_1[0]


# 需要根据模型类型重写
def init_model():
    tokenizer = BertTokenizerFast(vocab_file="./model/vocab.txt")
    session_1 = onnxruntime.InferenceSession("test.onnx")
    return tokenizer, session_1


if __name__ == "__main__":
    tokenizer, session_1 = init_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(model)
    app.run(host="0.0.0.0", port=8080)
