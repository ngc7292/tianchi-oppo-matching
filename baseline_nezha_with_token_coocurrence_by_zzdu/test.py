# -*- coding: utf-8 -*-
"""
__title__="test"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
from tqdm import tqdm
import time
a = range(100)

with tqdm(a) as phar:
    phar.set_description("epoch 1")
    for i in phar:
        phar.set_postfix({"loss":i})
        time.sleep(1)
