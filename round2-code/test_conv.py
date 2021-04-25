# -*- coding: utf-8 -*-
"""
__title__="test_conv"
__author__="ngc7293"
__mtime__="2021/4/22"
"""
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained("/remote-home/zyfei/project/tianchi/models/chinese-roberta-wwm-ext")
model.save_pretrained("/remote-home/zyfei/project/tianchi/models/chinese-roberta-wwm-ext")
