# -*- coding: utf-8 -*-
"""
__title__="test_data"
__author__="ngc7293"
__mtime__="2021/4/26"
"""
from transformers import BertTokenizer

from DataCollator import get_special_tokens_mask

tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

a = [102, 110, 106, 116, 103, 111, 114, 115, 103, 0, 0, 0]

a_list = get_special_tokens_mask(tokenizer, a, already_has_special_tokens=True)

print(a)
