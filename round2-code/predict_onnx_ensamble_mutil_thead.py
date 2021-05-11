# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/4/19"
"""

import os
import torch
import onnxruntime
import numpy as np
import threading
from flask import Flask, request
from multiprocessing import Process, Queue
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

def run_proc(session, inputs, q):
    print("runnning proc")
    output = session.run(None, inputs)[0]
    output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
    result = output_score[:, 1] / output_score.sum(axis=1)
    q.put(result)

class MyThread(threading.Thread):
    def __init__(self, session, inputs):
        super(MyThread, self).__init__()
        self.result = 0
        self.session = session
        self.inputs = inputs

    def run(self):
        print("runnning proc")
        output = self.session.run(None, self.inputs)[0]
        output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
        result = output_score[:, 1] / output_score.sum(axis=1)
        self.result = result

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

# 需要根据模型类型重写
def infer(session_1, session_2, tokenizer, text_1, text_2):
    sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
        " ")
    sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
    # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
    # 1 is word in other sentence and 0 is not in other sentence
    co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
        1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)
    input_ids = torch.tensor([sample.input_ids], device=device)
    token_type_ids = torch.tensor([sample.token_type_ids], device=device)
    co_ocurrence_list = torch.tensor([co_ocurrence_list], device=device)

    ort_inputs = {session_1.get_inputs()[0].name: to_numpy(input_ids),
                  session_1.get_inputs()[1].name: to_numpy(token_type_ids),
                  session_1.get_inputs()[2].name: to_numpy(co_ocurrence_list)}

    p_1 = MyThread(session_1, ort_inputs)
    p_2 = MyThread(session_2, ort_inputs)

    p_1.start()
    p_2.start()

    p_1.join()
    p_2.join()
    result = (p_1.get_result() + p_2.get_result()) / 2
    return result[0]

# 需要根据模型类型重写
def init_model(tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    session_1 = onnxruntime.InferenceSession("nezha-base-17-co.onnx")
    session_2 = onnxruntime.InferenceSession("nezha-base-22-co.onnx")
    return tokenizer, session_1, session_2


if __name__ == "__main__":
    tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"
    # tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/macbert_base_output_without_round1"
    tokenizer, session_1, session_2 = init_model(tokenizer_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(model)
    app.run(host="0.0.0.0", port=8080)
