# -*- coding: utf-8 -*-
"""
__title__="convert_pytorch_to_onnx"
__author__="ngc7293"
__mtime__="2021/4/21"
"""
import torch
import torch.onnx
from modeling_nezha import NeZhaForSequenceClassificationWithHeadClass, NeZhaForSequenceClassificationWithClsCat, \
    NeZhaForSequenceClassificationWithHeadClassMD
from configuration_nezha import NeZhaConfig

from modeling_bert import BertConfig, BertForSequenceClassificationWithClsCat

from transformers import BertTokenizer

device = torch.device('cuda:0')  # if torch.cuda.is_available() else torch.device('cpu')

model_name_or_path = "./nezha_base_v2_4_28_4"
tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"
# tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/macbert_base_output_without_round1"

config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)

# config = BertConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
# model = BertForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)

tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
model.to(device)

text_1 = "1959 177"
text_2 = "29 35 35 19 1696 23 43"

sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)

input_ids = torch.tensor([sample.input_ids], device=device)
token_type_ids = torch.tensor([sample.token_type_ids], device=device)

output = model(input_ids, token_type_ids)

sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
    " ")
sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
# [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
# 1 is word in other sentence and 0 is not in other sentence
co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
    1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]

input_dict = (torch.tensor([sample.input_ids], device=device),
              {
                  'token_type_ids': torch.tensor([sample.token_type_ids], device=device),
                  'co_ocurrence_ids': torch.tensor([co_ocurrence_list], device=device)
              })

input_names = ["input_ids", "token_type_ids", "co_ocurrence_ids"]
output_names = ["logtis"]

torch.onnx.export(model, input_dict, "nezha-4.onnx", verbose=True, input_names=input_names,
                  output_names=output_names, dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                                           'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                                                           'co_ocurrence_ids': {0: 'batch_size', 1: 'sequence'}},
                  opset_version=10)
