# -*- coding: utf-8 -*-
"""
__title__="furnturning_n_gram"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import os
import time
import torch
import tqdm
import random
import transformers
import numpy as np
import pandas as pd
import networkx as nx

from torch.utils.data.dataset import Dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from modeling_nezha import NeZhaForSequenceClassificationWithClsCat, NeZhaForSequenceClassificationWithMutilDropout
from configuration_nezha import NeZhaConfig
from transformers import BertTokenizer, DataCollatorWithPadding


class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx][0] + ' [SEP] ' + self.texts[idx][1]

        sentence_list1, sentence_list2 = text.split("[SEP]")
        sentence_list1, sentence_list2 = sentence_list1.strip().split(" "), sentence_list2.strip().split(
            " ")
        sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
        # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
        # 1 is word in other sentence and 0 is not in other sentence
        co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
            1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]

        examples = self.tokenizer(text, add_special_tokens=True, truncation=True)
        if isinstance(self.labels, np.ndarray):
            examples['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        examples['token_type_ids'] = co_ocurrence_list

        return examples

    def __len__(self):
        return len(self.texts)


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

    def attack(self, epsilon=0.2, alpha=0.3, emb_name='word_embeddings.', is_first_attack=False):
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


def train(model, attack_model, train_loader, optimizer, device, epoch=0, epochs=0):
    model.train()  # Turn on the train mode
    start_time = time.time()
    with tqdm.tqdm(train_loader, leave=False) as phar:
        phar.set_description("epoch " + str(epoch) + "/" + str(epochs))
        for batch in phar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            co_ocurrence_ids = batch['token_type_ids'].to(device)

            labels = batch['labels'].to(device)
            loss, logits = model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
            loss.backward()

            if attack_model.name == "fgm":
                attack_model.attack()
                loss, logits = model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
                attack_model.restore()  # 恢复embedding参数
            elif attack_model.name == "pgd":
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


def evaluate(eval_model, val_loader, device):
    eval_model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            co_ocurrence_ids = batch['token_type_ids'].to(device)
            output = eval_model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            y_pred.append(output_score[:, 1] / output_score.sum(axis=1))
            y_true.append(batch['labels'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true, y_pred)


def predict_test(model, test_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)[0]
            output = output.cpu().numpy()
            y_pred.append(output)
    y_pred = np.concatenate(y_pred)
    return y_pred


def run():
    transformers.logging.set_verbosity_error()

    train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
    test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

    vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence/tokens.txt'

    epochs = 5
    lr = 1e-5
    batch_size = 64
    n_split = 5
    fold_path = "./kfold-19"

    random_seed = 2021
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    model_name_or_path = "/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence/output/checkpoint-30000"
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    config = NeZhaConfig.from_pretrained(model_name_or_path, num_labels=2, output_hidden_states=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_collator = DataCollatorWithPadding(tokenizer)

    train_texts, train_labels, _ = load_data(train_path)
    test_texts, _, _ = load_data(test_path)

    pos_G = nx.Graph()
    neg_G = nx.Graph()

    for index, label in enumerate(train_labels):
        if label == 1:
            pos_G.add_edge(train_texts[index][0], train_texts[index][1])
        else:
            neg_G.add_edge(train_texts[index][0], train_texts[index][1])

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

    train_texts_aug = pos_dataset + neg_dataset
    train_labels_aug = [1] * len(pos_dataset) + [0] * len(neg_dataset)

    # 合并原始的负样本
    for index, label in enumerate(train_labels):
        if label == 0:
            train_texts_aug.append(train_texts[index])
            train_labels_aug.append(0)

    train_texts_aug = np.array(train_texts_aug)
    train_labels_aug = np.array(train_labels_aug)

    skf = StratifiedKFold(n_splits=n_split, random_state=1017, shuffle=True)

    print(
        f"traing on {skf.n_splits} fold, {device} , saving k-fold path is {fold_path}, which checkpoint is {model_name_or_path}.")
    global_best_auc = 0
    global_best_fold = -1
    for i, (train_index, val_index) in enumerate(skf.split(train_texts_aug, train_labels_aug)):
        print("FOLD | ", i + 1)
        print("###" * 35)

        train_dataset = MatchingDataset(train_texts_aug[train_index], train_labels_aug[train_index], tokenizer)
        val_dataset = MatchingDataset(train_texts_aug[val_index], train_labels_aug[val_index], tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        # model init is here
        config.classifier_dropout_prob = 0.2
        config.is_co_ocurrence = True
        model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        attack_model = FGM(model)
        # attack_model = PGD(model)

        best_auc = 0
        current_fold_path = os.path.join(fold_path, f'fold_co_{i}')
        for i in range(epochs):
            train(model, attack_model, train_loader, optimizer, device, i, epochs)
            auc = evaluate(model, val_loader, device)
            print(f'epoch:{i},auc:{auc}')
            if auc > best_auc:
                best_auc = auc
                model.save_pretrained(current_fold_path)
                # torch.save(model.state_dict(), current_fold_path)
        if best_auc > global_best_auc:
            global_best_auc = best_auc
            global_best_fold = i
        print(f'best auc:{best_auc}')

    print(f"global best fold is {global_best_fold}, and best auc is {global_best_auc}")


if __name__ == '__main__':
    run()
