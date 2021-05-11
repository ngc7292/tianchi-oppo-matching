# -*- coding: utf-8 -*-
"""
__title__="test"
__author__="ngc7293"
__mtime__="2021/3/16"
"""
import torch
from transformers import BertForMaskedLM, TFBertForMaskedLM, BertTokenizer,BertModel

vocab_data_path = "./vocab.txt"

print("create tokenizer...")
tokenizer = BertTokenizer(vocab_file=vocab_data_path)

tokenizer.save_pretrained(save_directory="/remote-home/zyfei/project/tianchi/baseline/mybert/bert-pretrain-with-mlm-60000")

exit()
model_path = "./mybert/checkpoint-60000"

model = BertModel.from_pretrained(model_path)

text = "[CLS] 12 15 11 16 [SEP] 8 9 10 [MASK] 11 [SEP]"

tokenized_text = tokenizer.tokenize(text)

print(tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])

model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    # outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)
    predictions = outputs.logits

predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0, (len(tokenized_text)))]
predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in
                   range(0, (len(tokenized_text)))]

print('Predicted token is:', "".join(predicted_token))


