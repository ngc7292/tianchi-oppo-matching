# -*- coding: utf-8 -*-
"""
__title__="test_vote"
__author__="ngc7293"
__mtime__="2021/4/3"
"""

# maj = np.apply_along_axis(
#                 lambda x: np.argmax(
#                     np.bincount(x, weights=self._weights_not_none)),
#                 axis=1, arr=predictions)

import numpy as np
from scipy.special import softmax

# a = [[[0.1, 0.9], [0.2, 0.8], [0.19, 0.7], [0.4, 0.5], [0.5, 0.6]], [[0.1, 0.9], [0.2, 0.8], [0.19, 0.7], [0.4, 0.5], [0.5, 0.6]]]
#
# a = np.array(a)
# b = softmax(a, axis=-1)
#
# # print(b)
# #
# # c = sorted(b, key=lambda x: np.abs(x[0] - 0.5))[::-1]
# #
# # print(c)
#
# print(a[:,:1,,])
# d = np.concatenate((a[:,1], a), axis=-2)

# print(d)

from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoConfig

a = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

config = AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

print(a)