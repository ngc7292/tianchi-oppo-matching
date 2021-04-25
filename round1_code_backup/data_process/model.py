# -*- coding: utf-8 -*-
"""
__title__="model"
__author__="ngc7293"
__mtime__="2021/3/1"
"""
import torch
from torch import nn

import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import BertForPreTraining, BertModel, BertForMaskedLM


class ERINE_Matching(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained("nghuyong/ernie-1.0")

    def masked_loss(self, y_true, y_pred, y_mask):
        y_pred = y_pred.transpose(1, 2)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(y_pred, y_true)
        loss = torch.sum(loss * y_mask) / torch.sum(y_mask)
        return loss

    def forward(self, **kwargs):
        input_ids = kwargs['token_ids']
        token_type_ids = kwargs['token_type_ids']
        y_mask = kwargs['attention_mask']
        target = kwargs['target']

        model_outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

        loss = self.masked_loss(target, model_outputs.logits, y_mask)
        pred = model_outputs.logits
        result = {'pred': pred, 'loss': loss}
        return result

    def predict(self, **kwargs):
        input_ids = kwargs['token_ids']
        token_type_ids = kwargs['token_type_ids']
        attention_mask = kwargs['attention_mask']

        model_outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return {'pred': model_outputs.logits}
