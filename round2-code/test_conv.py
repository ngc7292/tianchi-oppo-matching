# -*- coding: utf-8 -*-
"""
__title__="test_conv"
__author__="ngc7293"
__mtime__="2021/4/22"
"""
from transformers import BertTokenizer, AutoModel
import torch
from modeling_nezha import NeZhaConfig, NeZhaForMaskedLM

MODEL_NAME = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_2_v3_clean_round2data/checkpoint-30000"
tokenizer = BertTokenizer.from_pretrained("/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_2_v3_clean_round2data/checkpoint-40000")
model = NeZhaForMaskedLM.from_pretrained(MODEL_NAME)


# a = "[CLS] 134 272 91 2109 208 [SEP] 227 134 2107 2109 208 [SEP]"
# a_mask = "134 272 [MASK] 2109 208"
# b = "227 [MASK] 2107 12 208"

a = "[CLS] 12 5 239 243 29 1001 126 1405 11 [SEP] 29 485 12 251 1405 11"
a_mask = "12 5 239 243 29 1001 126 1405 11"
b = "29 485 [MASK] [MASK] 1405 11"
batch = tokenizer(text=[a_mask], text_pair=[b], return_tensors='pt')

output = model(batch["input_ids"], token_type_ids=batch["token_type_ids"])[0]

output = output.view(-1, tokenizer.vocab_size)
output_s = torch.argmax(output, dim=-1)
s = tokenizer.decode(output_s)

print(output.view)