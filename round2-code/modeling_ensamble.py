# -*- coding: utf-8 -*-
"""
__title__="modeling_ensamble"
__author__="ngc7293"
__mtime__="2021/4/25"
"""
import torch.nn as nn
import numpy as np
from modeling_bert import BertConfig, BertForSequenceClassificationWithClsCat
from modeling_nezha import NeZhaConfig, NeZhaForSequenceClassificationWithClsCat


class EnsambleModel(nn.Module):
    """
    this model only for infer
    """

    def __init__(self, macbert_path=None, nezha_path=None):
        super(EnsambleModel, self).__init__()
        self.macbert_config = BertConfig.from_pretrained(macbert_path)
        self.macbert = BertForSequenceClassificationWithClsCat.from_pretrained(macbert_path, config=self.macbert_config)
        self.macbert.eval()
        self.macbert.set_attack()
        self.nezha_config = NeZhaConfig.from_pretrained(nezha_path)
        self.nezha = NeZhaForSequenceClassificationWithClsCat.from_pretrained(nezha_path, config=self.nezha_config)
        self.nezha.set_attack()
        self.nezha.eval()

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                co_ocurrence_ids=None,
                attention_mask=None,
                ):
        macbert_output = self.macbert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, co_ocurrence_ids=co_ocurrence_ids)[0]
        nezha_output = self.nezha(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, co_ocurrence_ids=co_ocurrence_ids)[0]

        output = macbert_output + nezha_output

        macbert_output = macbert_output.cpu().numpy()
        # mac_output_score = np.exp(mac_output[:, 1])/ (np.exp(mac_output).sum(axis=1))
        mac_output_score = np.exp(macbert_output) / (np.exp(macbert_output).sum(axis=1, keepdims=True))
        macbert_output = mac_output_score[:, 1] / mac_output_score.sum(axis=1)

        nezha_output = nezha_output.cpu().numpy()
        # nezha_output_score = np.exp(nezha_output[:, 1])/ (np.exp(nezha_output).sum(axis=1))
        nezha_output_score = np.exp(nezha_output) / (np.exp(nezha_output).sum(axis=1, keepdims=True))
        nezha_output = nezha_output_score[:, 1] / nezha_output_score.sum(axis=1)

        output = (nezha_output + macbert_output) / 2

        return output
