# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/4/19"
"""

import os
import torch
import numpy as np
from flask import Flask, request

from modeling_nezha import NeZhaForSequenceClassificationWithClsCat, NeZhaForSequenceClassificationWithHeadClass, NeZhaConfig
from configuration_nezha import NeZhaConfig
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
            predict = infer(model, tokenizer, query_A, query_B)
            response["predict"] = str(predict)
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return response_batch


# 需要根据模型类型重写
def infer(model, tokenizer, text_1, text_2):
    device = model.device
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)
    input_ids = torch.tensor([sample.input_ids], device=device)
    token_type_ids = torch.tensor([sample.token_type_ids], device=device)
    model.eval()
    output = model(input_ids, token_type_ids=token_type_ids)[0]
    output = output.detach().cpu().numpy()
    # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
    output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
    result = output_score[:, 1] / output_score.sum(axis=1)
    return result[0]


# 需要根据模型类型重写
def init_model(tokenizer_path, model_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    config = NeZhaConfig.from_pretrained(model_path)
    model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_path, config=config)

    return tokenizer, model


if __name__ == "__main__":
    tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_with_label_2"
    model_path = "./model_9"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # print(device)
    tokenizer, model = init_model(tokenizer_path, model_path)
    model.to(device)
    # print(model)
    app.run(host="0.0.0.0", port=8080)
