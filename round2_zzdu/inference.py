# -*- coding: utf-8 -*-
"""
__title__="inference"
__author__="ngc7293"
__mtime__="2021/5/7"
"""
import torch

from transformers import BertTokenizer
from modeling_nezha import NeZhaConfig, NeZhaForSequenceClassification

model_dict = torch.load("./model/model.pkl")

config = NeZhaConfig.from_pretrained("config.json", num_labels=2)
model = NeZhaForSequenceClassification(config=config)

a = model.state_dict()
model.load_state_dict(model_dict)

tokenizer = BertTokenizer(vocab_file="./model/vocab.txt")

device = torch.device("cuda:0")
model.to(device)

text_1 = "1959 177"
text_2 = "29 35 35 19 1696 23 43"

text = text_1 + " [SEP] " + text_2

sample = tokenizer(text=text, add_special_tokens=True, truncation=True)

input_ids = torch.tensor([sample.input_ids], device=device)
# token_type_ids = torch.tensor([sample.token_type_ids], device=device)
attention_mask = torch.tensor([sample.attention_mask], device=device)

input_dict = (input_ids, attention_mask)
input_name = ["input_ids", "attention_mask"]
output_name = ["logits"]

torch.onnx.export(model, input_dict, "test.onnx", verbose=True, input_names=input_name,
                  output_names=output_name, dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                                          'attention_mask': {0: 'batch_size', 1: 'sequence'}},
                  opset_version=10, use_external_data_format=True)
