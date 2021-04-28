# -*- coding: utf-8 -*-
"""
__title__="predict_onnx"
__author__="ngc7293"
__mtime__="2021/4/22"
"""
import onnx
import torch
import time
import onnxruntime
import numpy as np
from modeling_nezha import NeZhaForSequenceClassificationWithHeadClass, NeZhaForSequenceClassificationWithClsCat, \
    NeZhaForSequenceClassificationWithHeadClassMD
from configuration_nezha import NeZhaConfig

from transformers import BertTokenizer


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model_name_or_path = "./model_28"
tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"

config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)

tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
model.to(device)

text_1 = "1 2 3 4 5 6 7"
text_2 = "8 9 10 4 11"
# label = 0
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
co_ocurrence_ids = torch.tensor([co_ocurrence_list], device=device)

ort_session = onnxruntime.InferenceSession("model-28-co.onnx")

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
              ort_session.get_inputs()[1].name: to_numpy(token_type_ids),
              ort_session.get_inputs()[2].name: to_numpy(co_ocurrence_ids)}

start_time = time.time()
output = ort_session.run(None, ort_inputs)[0]
output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
ort_result = output_score[:, 1] / output_score.sum(axis=1)
print(time.time() - start_time)

start_time = time.time()
output = model(input_ids, token_type_ids=token_type_ids, co_ocurrence_ids=co_ocurrence_ids)[0]
output = output.detach().cpu().numpy()
output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
result = output_score[:, 1] / output_score.sum(axis=1)
print(time.time() - start_time)

print(ort_result)
print(result)
