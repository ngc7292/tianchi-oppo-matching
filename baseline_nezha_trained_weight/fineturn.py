# -*- coding: utf-8 -*-
"""
__title__="fineturning"
__author__="ngc7293"
__mtime__="2021/3/18"
"""
import os
import argparse
import fitlog
import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import CrossEntropyLoss, Linear, Dropout
from torch.optim import AdamW

from fastNLP import DataSet, Instance, cache_results, LossInForward, Trainer, Tester, AccuracyMetric, MetricBase
from fastNLP.io import DataBundle
from fastNLP.core.callback import WarmupCallback, FitlogCallback
from transformers import BertModel, BertTokenizer
from sklearn.metrics import roc_auc_score


class MatchingModel(nn.Module):
    def __init__(self,
                 pretrain_model_path="/remote-home/zyfei/project/tianchi/baseline/mybert/bert-pretrain-with-mlm-140000"):
        super(MatchingModel, self).__init__()
        self.pretrain_model = BertModel.from_pretrained(pretrain_model_path)

        self.dropout = Dropout(p=0.5)
        self.linear_1 = Linear(self.pretrain_model.config.hidden_size, 2048)
        self.linear_2 = Linear(2048, 2)
        # self.linear = Linear(self.pretrain_model.config.hidden_size, 2)

        self.loss_func = CrossEntropyLoss()

    def forward(self, **kwargs):
        words = kwargs['words']
        target = kwargs['target']

        model_output = self.pretrain_model(words)

        output = model_output.pooler_output
        output = self.linear_2(self.dropout(self.linear_1(output)))
        # output = self.linear(self.dropout(output))
        # output = F.softmax(output)
        output = F.sigmoid(output)

        loss = self.loss_func(output, target)

        return {'pred': output, 'loss': loss}

    def predict(self, **kwargs):
        words = kwargs['words']
        model_output = self.pretrain_model(words)
        output = model_output.pooler_output
        output = self.linear_2(self.dropout(self.linear_1(output)))
        # output = self.linear(self.dropout(output))
        # output = F.softmax(output)
        output = F.sigmoid(output)
        return {'pred': output}


class Dataloader:
    def __init__(self,
                 model_name_or_path="/remote-home/zyfei/project/tianchi/baseline/mybert/bert-pretrain-with-mlm-90000"):
        self.dir_path = "/remote-home/zyfei/project/tianchi/data"
        self.train_data_path = "gaiic_track3_round1_train_20210228.tsv"
        self.test_data_path = "gaiic_track3_round1_testA_20210228.tsv"

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

    def _load(self, data_path):
        ds = DataSet()
        with open(data_path, "r") as fp:
            for data in fp.readlines():
                data = data.replace("\n", "")
                data = data.split("\t")
                if len(data) == 3:
                    ds.append(Instance(words1=data[0], words2=data[1], target=int(data[2])))
                elif len(data) == 2:
                    ds.append(Instance(words1=data[0], words2=data[1]))
                else:
                    raise NotImplementedError
        return ds

    def load_data(self):
        return DataBundle(datasets={
            'train': self._load(data_path=os.path.join(self.dir_path, self.train_data_path)),
            'dev': self._load(data_path=os.path.join(self.dir_path, self.test_data_path)),
            'test': self._load(data_path=os.path.join(self.dir_path, self.test_data_path))
        })

    def get_databundle(self):
        data_bundle = self.load_data()

        def tokenize(ins):

            text = "[CLS]" + ins["words1"] + "[SEP]" + ins["words2"] + "[SEP]"

            tokenized_text = self.tokenizer.tokenize(text)

            indexed_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            return {'words': indexed_text}

        for name, dataset in data_bundle.datasets.items():
            dataset.apply_more(tokenize)

            input_fields = ["words"]
            target_fields = ["target"]
            dataset.set_input(*input_fields, flag=True)
            for fields in target_fields:
                if dataset.has_field(fields):
                    dataset.set_target(fields, flag=True)
                    dataset.set_input(fields)

        return data_bundle


class AUCMetric(MetricBase):
    def __init__(self, label=None, pred=None):
        super().__init__()
        self._init_param_map(label=label, pred=pred)

        self.pred_list = []
        self.target_list = []

    def evaluate(self, label, pred):
        # pred = torch.sigmoid(pred)[:, 1]
        pred = pred[:, 1]
        label = label.cpu().tolist()
        pred = pred.cpu().tolist()
        self.pred_list.extend(pred)
        self.target_list.extend(label)
        # self.auc_num += roc_auc_score(label, pred)
        # self.total += label.size(0)

    def get_metric(self, reset=True):  # 在这里定义如何计算metric
        auc = roc_auc_score(self.target_list, self.pred_list)
        if reset:  # 是否清零以便重新计算
            self.pred_list = []
            self.target_list = []
        return {'auc': auc}  # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


class ProjectConfig:
    def __init__(self):
        self.arg = argparse.ArgumentParser()

        self.arg.add_argument("--batch_size_per_gpu", default=64, type=int)
        self.arg.add_argument("--n_epochs", default=50, type=int)

        self.arg.add_argument("--lr", default=2e-5, type=float)
        self.arg.add_argument("--weight_decay", default=0.01, type=float)
        self.arg.add_argument("--seed", default=42)
        self.arg.add_argument("--output_path", default="./output")
        self.arg.add_argument("--debug", action="store_true")

        self.arg = self.arg.parse_args()

    def get_arg(self):
        return self.arg


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/tokenized_data_mlm-60000", _refresh=True)
def load_data():
    return Dataloader().get_databundle()


def run():
    fitlog.set_log_dir("/remote-home/zyfei/project/tianchi/logs")
    args = ProjectConfig().get_arg()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # read dataset
    data_bundle = load_data(_refresh=False)

    if args.debug:
        data_bundle.datasets["dev"] = data_bundle.datasets["train"][1000:2000]
        data_bundle.datasets["train"] = data_bundle.datasets["train"][:1000]
    else:
        data_bundle.datasets["train"], data_bundle.datasets["dev"] = data_bundle.datasets['train'].split(0.2)

    print(data_bundle)
    data_bundle.datasets["train"].print_field_meta()

    device_num = torch.cuda.device_count()
    device = [i for i in range(device_num)]

    print(f"training, and using {device_num} gpus which is {str(device)}")

    model = MatchingModel()

    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    callbacks = [WarmupCallback(warmup=0.1, schedule='linear'),
                 FitlogCallback(data_bundle.get_dataset("dev"))]

    # define trainer
    trainer = Trainer(train_data=data_bundle.get_dataset("train"),
                      model=model,
                      optimizer=optimizer,
                      batch_size=device_num * args.batch_size_per_gpu,
                      n_epochs=args.n_epochs,
                      dev_data=data_bundle.get_dataset("dev"),
                      metrics=AUCMetric(pred='pred', label='target'),
                      metric_key='auc',
                      loss=LossInForward(),
                      device=device,
                      check_code_level=-1,
                      save_path=args.output_path,
                      callbacks=callbacks)

    train_return = trainer.train(load_best_model=True)

    print(train_return)


if __name__ == '__main__':
    run()
