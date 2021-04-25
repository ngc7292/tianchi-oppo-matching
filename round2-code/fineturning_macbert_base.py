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
import argparse

# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
from fastNLP import DataSet as FastNLPDataSet
from fastNLP.core import Instance, DataSetIter, RandomSampler, cache_results
from fastNLP.core.metrics import AccuracyMetric
from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
from modeling_bert import BertForSequenceClassificationWithClsCat, BertConfig
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
    def __init__(self):
        self.name = 'freelb'
        self.adv_init_mag = 1
        self.adv_steps = 4
        self.gradient_accumulation_steps = 1
        self.adv_lr = 1e-1
        self.adv_max_norm = 1


def tokenize_data(text_1, text_2, tokenizer):
    sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
        " ")
    sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
    # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
    # 1 is word in other sentence and 0 is not in other sentence
    co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
        1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True)
    return {
        "input_ids": sample.input_ids,
        "token_type_ids": sample.token_type_ids,
        "attention_mask": sample.attention_mask,
        "co_ocurrence_ids": co_ocurrence_list
    }


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/macbert-4-17-with-label-fineturning", _refresh=False)
def load_data_fastnlp(path, tokenizer):
    ds = FastNLPDataSet()
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            if len(temp) > 2:
                data_1 = tokenize_data(temp[0], temp[1], tokenizer)
                ds.append(Instance(input_ids=data_1["input_ids"], token_type_ids=data_1["token_type_ids"],
                                   attention_mask=data_1["attention_mask"], co_ocurrence_ids=data_1["co_ocurrence_ids"],
                                   label=int(temp[2])))
    ds.set_input("input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids", "label")
    ds.set_target("label")
    return ds


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/macbert-4-17-with-label-fineturning-enhance", _refresh=False)
def load_data_fastnlp_enhance(path, tokenizer):
    ds = FastNLPDataSet()
    samples = []
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines(), leave=True):
            temp = line.split('\t')
            if len(temp) > 2:
                data_1 = tokenize_data(temp[0], temp[1], tokenizer)
                ds.append(Instance(input_ids=data_1["input_ids"], token_type_ids=data_1["token_type_ids"],
                                   attention_mask=data_1["attention_mask"],
                                   co_ocurrence_ids=data_1["co_ocurrence_ids"],
                                   label=int(temp[2])))
                data_2 = tokenize_data(temp[1], temp[0], tokenizer)
                ds.append(Instance(input_ids=data_2["input_ids"], token_type_ids=data_2["token_type_ids"],
                                   attention_mask=data_2["attention_mask"],
                                   co_ocurrence_ids=data_2["co_ocurrence_ids"],
                                   label=int(temp[2])))

                samples.append([temp[0], temp[1]])
                samples.append([temp[1], temp[0]])
                labels.append(int(temp[2]))

    pos_G = nx.Graph()
    neg_G = nx.Graph()

    for index, label in enumerate(labels):
        if label == 1:
            pos_G.add_edge(samples[index][0], samples[index][1])
        else:
            neg_G.add_edge(samples[index][0], samples[index][1])

    pos_dataset = []
    neg_dataset = []
    for c in nx.connected_components(pos_G):
        nodes = list(c)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pos_dataset.append([nodes[i], nodes[j]])
            if nodes[i] in neg_G:
                for neiber_neg in neg_G.adj[nodes[i]]:
                    for z in range(len(nodes)):
                        if z != i:
                            neg_dataset.append([neiber_neg, nodes[z]])

    for text_1, text_2 in pos_dataset[:100000]:
        data = tokenize_data(text_1, text_2, tokenizer)
        ds.append(Instance(input_ids=data["input_ids"], token_type_ids=data["token_type_ids"],
                           attention_mask=data["attention_mask"],
                           co_ocurrence_ids=data["co_ocurrence_ids"],
                           label=1))

    for text_1, text_2 in neg_dataset[:100000]:
        data = tokenize_data(text_1, text_2, tokenizer)
        ds.append(Instance(input_ids=data["input_ids"], token_type_ids=data["token_type_ids"],
                           attention_mask=data["attention_mask"],
                           co_ocurrence_ids=data["co_ocurrence_ids"],
                           label=0))

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

            if attack_model.name == "fgm":
                loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
                attack_model.attack()
                loss, logits = model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
                attack_model.restore()  # 恢复embedding参数
            elif attack_model.name == "pgd":
                loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
                if epoch > 3:
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
            elif attack_model.name == "freelb":
                if epoch <= 3:
                    loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                         co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                    loss.backward()
                elif epoch > 3:
                    # init delta
                    embeds_init = model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,
                                                        co_ocurrence_ids=co_ocurrence_ids)
                    if attack_model.adv_init_mag > 0:
                        input_mask = attention_mask
                        input_lengths = torch.sum(input_mask, 1)
                        # check the shape of the mask here..

                        delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                        dims = input_lengths * embeds_init.size(-1)
                        mag = attack_model.adv_init_mag / torch.sqrt(dims)
                        delta = (delta * mag.view(-1, 1, 1)).detach()
                    else:
                        delta = torch.zeros_like(embeds_init)

                    for astep in range(attack_model.adv_steps):
                        # (0) forward
                        delta.requires_grad_()
                        inputs_embeds = delta + embeds_init
                        outputs = model(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        co_ocurrence_ids=co_ocurrence_ids, labels=labels, inputs_embeds=inputs_embeds)
                        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                        # (1) backward

                        if attack_model.gradient_accumulation_steps > 1:
                            loss = loss / attack_model.gradient_accumulation_steps

                        loss = loss / attack_model.adv_steps
                        loss.backward()

                        if astep == attack_model.adv_steps - 1:
                            # further updates on delta
                            break

                        # (2) get gradient on delta
                        delta_grad = delta.grad.clone().detach()
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + attack_model.adv_lr * delta_grad / denorm).detach()
                        if attack_model.adv_max_norm > 0:
                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                            exceed_mask = (delta_norm > attack_model.adv_max_norm).to(embeds_init)
                            reweights = (attack_model.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                            delta = (delta * reweights).detach()
                        embeds_init = model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,
                                                            co_ocurrence_ids=co_ocurrence_ids)

            else:
                raise NotImplementedError

            optimizer.step()
            model.zero_grad()

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
            # co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            co_ocurrence_ids = None
            output = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            pred_label = np.argmax(output_score, axis=-1)
            y_pred.append(pred_label)
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    # return roc_auc_score(y_true, y_pred)
    return accuracy_score(y_true, y_pred)


def run():
    transformers.logging.set_verbosity_error()
    args = argparse.ArgumentParser()

    args.add_argument("--train_path",
                      default="/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv")
    args.add_argument("--test_path",
                      default="/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv")

    args.add_argument("--epoches", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--fold_name", default="./model_20")
    args.add_argument("--evalution_method", default="auc")
    args.add_argument("--attack_method", default="fgm")
    args.add_argument("--model_type", default="clscat")
    args.add_argument("--data_enhance", action='store_true')

    args = args.parse_args()
    train_path = args.train_path
    test_path = args.test_path

    # tokenizer_file = '/remote-home/zyfei/project/tianchi/model_output/nezha_output_2'
    tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/macbert_base_output_without_round1"

    epochs = args.epoches
    lr = 1e-5
    batch_size = args.batch_size
    fold_path = args.fold_name
    evalution_method = args.evalution_method
    model_type = args.model_type

    fitlog.add_hyper(args)
    # fitlog.add_hyper(fold_path, "fold_path")  # 记录本文件中写死的超参数
    # fitlog.add_hyper("no co_ocurrence", "attention")

    random_seed = 2021
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    fitlog.add_hyper(random_seed, "random_seed")

    model_output_path = "/remote-home/zyfei/project/tianchi/model_output/"
    checkpoint = "macbert_base_output_without_round1/checkpoint-55000/"
    model_name_or_path = os.path.join(model_output_path, checkpoint)
    fitlog.add_hyper(checkpoint, "model_name_or_path")

    tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # train_dataset, dev_dataset = load_data_fastnlp(train_path, tokenizer).split(0.3, shuffle=True)
    # train_dataset = load_data_fastnlp(train_path, tokenizer, _cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-17-with-label-fineturning-train", _refresh=False)
    if args.data_enhance:
        train_dataset = load_data_fastnlp_enhance(train_path, tokenizer,
                                                  _cache_fp="/remote-home/zyfei/project/tianchi/cache/macbert-4-17-with-label-fineturning-train-enhance",
                                                  _refresh=False)
    else:
        train_dataset = load_data_fastnlp(train_path, tokenizer,
                                          _cache_fp="/remote-home/zyfei/project/tianchi/cache/macbert-4-17-with-label-fineturning-train",
                                          _refresh=False)

    print(train_dataset.get_length())
    train_dataset.print_field_meta()
    dev_dataset = load_data_fastnlp(test_path, tokenizer,
                                    _cache_fp="/remote-home/zyfei/project/tianchi/cache/macbert-4-17-with-label-fineturning-dev",
                                    _refresh=False)

    if model_type == "headwithmd":
        # NeZhaLabelHead for classifier with mutil dropout
        raise NotImplementedError
    elif model_type == "head":
        # only use NeZhaLabelHead for classifier
        raise NotImplementedError
    elif model_type == "clscat":
        # cat cls to classifier
        config = BertConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
        config.classifier_dropout_prob = 0.3
        model = BertForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)
    else:
        raise NotImplementedError

    fitlog.add_hyper(model_type, "model")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if args.attack_method == "fgm":
        attack_model = FGM(model)
        model.set_attack()
    elif args.attack_method == "pgd":
        attack_model = PGD(model)
        model.set_attack()
    elif args.attack_method == "freelb":
        attack_model = FreeLB()
        model.set_attack()

    else:
        raise NotImplementedError

    # fitlog.add_hyper(attack_model.name, "attack_model")
    tokenizer.save_pretrained(fold_path)

    best_result = 0
    best_model = None
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
            best_model = model
            best_model.save_pretrained(fold_path)
            # torch.save(model.state_dict(), current_fold_path)
    print(f'best result:{best_result}')
    best_model.save_pretrained(fold_path)

    fitlog.add_best_metric(str(best_result), evalution_method)
    fitlog.finish()


if __name__ == '__main__':
    run()
