# -*- coding: utf-8 -*-
"""
__title__="fineturning_nezha"
__author__="ngc7293"
__mtime__="2021/4/16"
"""
import os
import time
import torch
import tqdm
import random
import transformers
import fitlog
import numpy as np
import pandas as pd
import networkx as nx

# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
from fastNLP import DataSet as FastNLPDataSet
from fastNLP.core import Instance, DataSetIter, RandomSampler, cache_results
from fastNLP.core.metrics import AccuracyMetric
from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
from modeling_nezha import NeZhaForSequenceClassificationWithClsCat
from configuration_nezha import NeZhaConfig
from transformers import BertTokenizer

fitlog.set_log_dir("/remote-home/zyfei/project/tianchi/round2-code/logs")


class FGM():
    def __init__(self, model):
        self.name = "fgm"
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.2, emb_name='word_embeddings.'):

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.name = "pgd"
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1, alpha=0.2, emb_name='word_embeddings.', is_first_attack=False):
        # epsilon = 1. alpha = 0.3
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FreeLB():
    def __init__(self, model):
        self.name='freelb'
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1, alpha=0.2, emb_name="word_embeddings.", is_first_attck=False):
        pass

def load_data(path):
    max_length = 0
    samples = []
    labels = []
    if path.split(".")[-1] == "tsv":
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                temp = line.split('\t')
                samples.append([temp[0], temp[1]])
                if len(temp) > 2:
                    labels.append(int(temp[2]))
        return np.array(samples), np.array(labels), max_length
    elif path.split(".")[-1] == "csv":
        lines = pd.read_csv(path)
        for i, data in lines.iterrows():
            samples.append([data[1], data[2]])
            if len(data) > 3:
                labels.append(int(data[3]))
        return np.array(samples), np.array(labels), max_length


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-17-with-label-fineturning", _refresh=False)
def load_data_fastnlp(path, tokenizer):
    ds = FastNLPDataSet()
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            if len(temp) > 2:
                sentence_list1, sentence_list2 = temp[0].strip().split(" "), temp[1].strip().split(
                    " ")
                sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
                # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
                # 1 is word in other sentence and 0 is not in other sentence
                co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
                    1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
                sample = tokenizer(text=temp[0], text_pair=temp[1], add_special_tokens=True, truncation=True)
                ds.append(Instance(input_ids=sample.input_ids, token_type_ids=sample.token_type_ids,
                                   attention_mask=sample.attention_mask, co_ocurrence_ids=co_ocurrence_list,
                                   label=int(temp[2])))
    ds.set_input("input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids", "label")
    ds.set_target("label")
    return ds


def train(model, attack_model, train_dataset, optimizer, device, epoch=0, epochs=0, batch_size=64):
    model.train()  # Turn on the train mode
    start_time = time.time()
    samplr = RandomSampler()
    train_loader = DataSetIter(dataset=train_dataset, batch_size=batch_size, sampler=samplr)
    with tqdm.tqdm(train_loader, leave=False) as phar:
        phar.set_description("epoch " + str(epoch) + "/" + str(epochs))
        for batch, _ in phar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            labels = batch['label'].to(device)

            loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
            loss.backward()

            if attack_model.name == "fgm":
                attack_model.attack()
                loss, logits = model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
                attack_model.restore()  # 恢复embedding参数
            elif attack_model.name == "pgd":
                if epoch > -1:
                    attack_model.backup_grad()
                    K = 3
                    for t in range(K):
                        attack_model.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K - 1:
                            model.zero_grad()
                        else:
                            attack_model.restore_grad()
                        loss_adv, logits = model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids,
                                                 labels=labels)[:2]
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    attack_model.restore()  # 恢复embedding参数
            else:
                raise NotImplementedError

            optimizer.step()

            phar.set_postfix({"loss": float(loss.cpu().detach())})

    spend_time = time.time() - start_time
    print("model traing finish and use time " + str(spend_time) + " s")


def evaluate(eval_model, val_dataset, device, batch_size=64):
    eval_model.eval()
    y_pred = []
    y_true = []
    sampler = RandomSampler()
    val_loader = DataSetIter(val_dataset, batch_size=batch_size, sampler=sampler)
    with torch.no_grad():
        for batch, _ in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            output = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            y_pred.append(output_score[:, 1] / output_score.sum(axis=1))
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true, y_pred)

def evaluate_acc(eval_model, val_dataset, device, batch_size=64):
    eval_model.eval()
    y_pred = []
    y_true = []
    sampler = RandomSampler()
    val_loader = DataSetIter(val_dataset, batch_size=batch_size, sampler=sampler)
    with torch.no_grad():
        for batch, _ in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            output = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            pred_label = np.argmax(output_score,axis=-1)
            y_pred.append(pred_label)
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    # return roc_auc_score(y_true, y_pred)
    return accuracy_score(y_true, y_pred)

def run():
    transformers.logging.set_verbosity_error()

    train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv'

    # tokenizer_file = '/remote-home/zyfei/project/tianchi/model_output/nezha_output_2'
    tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_output_with_label_2"

    epochs = 5
    lr = 1e-5
    batch_size = 128
    fold_path = "./model_4"
    evalution_method = "acc"

    fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

    random_seed = 2021
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_output_with_label_2"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_file,
                                              model_input_names=["input_ids", "attention_mask", "token_type_ids",
                                                                 "co_ocurrence_ids"])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset, dev_dataset = load_data_fastnlp(train_path, tokenizer).split(0.3, shuffle=True)

    config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
    config.num_hidden_layers = 12
    model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    attack_model = FGM(model)
    # attack_model = PGD(model)

    best_result = 0
    for i in range(epochs):
        train(model, attack_model, train_dataset, optimizer, device, i, epochs, batch_size)
        if evalution_method == "auc":
            result = evaluate(model, dev_dataset, device, batch_size)
        else:
            result = evaluate_acc(model, dev_dataset, device, batch_size)
        print(f'epoch:{i},{evalution_method}:{result}')
        fitlog.add_metric({"dev": {f"{evalution_method}": result}}, step=i)
        if result > best_result:
            best_result = result
            model.save_pretrained(fold_path)
            # torch.save(model.state_dict(), current_fold_path)
    print(f'best result:{best_result}')


if __name__ == '__main__':
    run()
