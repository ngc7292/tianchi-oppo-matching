# -*- coding: utf-8 -*-
"""
__title__="test_data"
__author__="ngc7293"
__mtime__="2021/4/26"
"""
from transformers import BertTokenizer

from DataCollator import get_special_tokens_mask

import argparse

a = argparse.ArgumentParser()
a.add_argument("--drop", default=None, type=float)

b = a.parse_args()

print(b)