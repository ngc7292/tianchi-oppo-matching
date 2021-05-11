# -*- coding: utf-8 -*-
"""
__title__="test_2"
__author__="ngc7293"
__mtime__="2021/5/5"
"""
import torch.nn as nn
import torch

class a(nn.Module):
    def __init__(self):
        super(a, self).__init__()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x-mean).pow(2).mean(-1) + 1e-5)
        return torch.cat()