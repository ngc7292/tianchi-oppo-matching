# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/4/19"
"""

import os
import torch
import onnx
import time
import onnxruntime

from onnxruntime import backend
import numpy as np
from flask import Flask, request

from transformers import BertTokenizer

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

    response_batch = {"results": []}
    for i in range(len(index_list)):
        index_str = index_list[i]

        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()
            predict = infer(session_1, session_2, session_3, session_4, tokenizer, query_A, query_B)
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
def infer(session_1, session_2, session_3, session_4, tokenizer, text_1, text_2):
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)

    ort_inputs = {session_1.get_inputs()[0].name: [sample.input_ids],
                  session_1.get_inputs()[1].name: [sample.token_type_ids]}

    output_1 = session_1.run(None, ort_inputs)[0]

    output_2 = session_2.run(None, ort_inputs)[0]

    output_3 = session_3.run(None, ort_inputs)[0]

    output_4 = session_4.run(None, ort_inputs)[0]

    output = output_1 + output_2 + output_3 + output_4

    output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
    result = output_score[:, 1] / output_score.sum(axis=1)
    result = result[0]

    return result


# 需要根据模型类型重写
def init_model(tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    session_1 = onnxruntime.InferenceSession("../onnx_models/fold-0.onnx")
    session_2 = onnxruntime.InferenceSession("../onnx_models/fold-1.onnx")
    session_3 = onnxruntime.InferenceSession("../onnx_models/fold-2.onnx")
    session_4 = onnxruntime.InferenceSession("../onnx_models/fold-3.onnx")

    return tokenizer, session_1 ,session_2 , session_3, session_4


if __name__ == "__main__":
    tokenizer_path = "../origin_model/tokenizer"
    tokenizer, session_1, session_2, session_3, session_4 = init_model(tokenizer_path) # , session_2, session_3, session_4
    app.run(host="0.0.0.0", port=8080)
