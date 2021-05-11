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
import math

from fastNLP import DataSet as FastNLPDataSet
from fastNLP.core import Instance, DataSetIter, RandomSampler, cache_results
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from modeling_nezha import NeZhaForSequenceClassificationWithHeadClass, NeZhaForSequenceClassificationWithHeadClassMD, \
    NeZhaForSequenceClassificationWithClsCat, NeZhaForSequenceClassificationWithClsCatForOnnx
from configuration_nezha import NeZhaConfig
from transformers import BertTokenizer, get_scheduler

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

    def attack(self, epsilon=0.3, alpha=0.2, emb_name='word_embeddings.', is_first_attack=False):
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


class NoneAttack():
    def __init__(self):
        self.name = 'none'


def tokenize_data(text_1, text_2, tokenizer):
    sentence_list1, sentence_list2 = text_1.strip().split(" "), text_2.strip().split(
        " ")
    sentence_set1, sentence_set2 = set(sentence_list1), set(sentence_list2)
    # [CLS] + sentence1 + [SEP] + sentence2 + [SEP]
    # 1 is word in other sentence and 0 is not in other sentence
    co_ocurrence_list = [0] + [1 if i in sentence_set2 else 0 for i in sentence_list1] + [0] + [
        1 if i in sentence_set1 else 0 for i in sentence_list2] + [0]
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True, max_length=64)
    return {
        "input_ids": sample.input_ids,
        "token_type_ids": sample.token_type_ids,
        "attention_mask": sample.attention_mask,
        "co_ocurrence_ids": co_ocurrence_list
    }


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-27-with-label-fineturning", _refresh=False)
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


@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-4-27-with-label-fineturning-kfold",
               _refresh=False)
def load_data(path, tokenizer):
    samples = []
    input_ids = []
    token_type_ids = []
    attention_masks = []
    co_ocurrence_ids = []

    labels = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().splitlines()):
            temp = line.split('\t')
            samples.append([temp[0], temp[1]])
            labels.append(int(temp[2]))

            data = tokenize_data(temp[0], temp[1], tokenizer)
            input_ids.append(data["input_ids"])
            token_type_ids.append(data["token_type_ids"])
            attention_masks.append(data["attention_mask"])
            co_ocurrence_ids.append(data["co_ocurrence_ids"])

    return np.array(samples), np.array(input_ids), np.array(token_type_ids), np.array(attention_masks), np.array(
        co_ocurrence_ids), np.array(labels)


def train(model, attack_model, train_dataset, optimizer, device, epoch=0, epochs=0, batch_size=64, lr_scheduler=None):
    model.train()  # Turn on the train mode
    start_time = time.time()
    samplr = RandomSampler()
    train_loader = DataSetIter(dataset=train_dataset, batch_size=batch_size, sampler=samplr)
    with tqdm.tqdm(train_loader, leave=False) as phar:
        phar.set_description("epoch " + str(epoch) + "/" + str(epochs))
        for batch, _ in phar:
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
                if epoch > 2:
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
                if epoch < 0:
                    loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                         co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                    loss.backward()
                elif epoch >= 0:
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
                            reweights = (attack_model.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(
                                -1, 1, 1)
                            delta = (delta * reweights).detach()
                        embeds_init = model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids,
                                                            co_ocurrence_ids=co_ocurrence_ids)
            elif attack_model.name == "none":
                # not use attack to use mutil dropout
                loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     co_ocurrence_ids=co_ocurrence_ids, labels=labels)[:2]
                loss.backward()
            else:
                raise NotImplementedError

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

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
    args.add_argument("--checkpoint", default="nezha_base_output_5_3_clean_round2data_3")
    args.add_argument("--epoches", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--fold_name", default="./nezha_5_1_2")
    args.add_argument("--evalution_method", default="auc")
    args.add_argument("--attack_method", default="fgm")
    args.add_argument("--model_type", default="clscat")
    args.add_argument("--n_splits", type=int, default=3)
    args.add_argument("--classifier_dropout", default=None, type=float)
    args.add_argument("--hidden_dropout", default=None, type=float)
    args.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    args.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    args.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    args.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )


    args = args.parse_args()
    train_path = args.train_path

    epochs = args.epoches
    lr = 1e-5
    batch_size = args.batch_size
    fold_path = args.fold_name
    evalution_method = args.evalution_method
    model_type = args.model_type

    fitlog.add_hyper(args)

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    fitlog.add_hyper(random_seed, "random_seed")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30_v2_round2data"

    model_output_path = "/remote-home/zyfei/project/tianchi/model_output"
    checkpoint = args.checkpoint
    model_name_or_path = os.path.join(model_output_path, checkpoint)
    fitlog.add_hyper(checkpoint, "model_name_or_path")
    print("traing in checkpoint " + checkpoint)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

    train_samples, train_input_ids, train_token_type_ids, train_attention_masks, train_co_ocurrence_ids, train_labels = load_data(
        train_path, tokenizer=tokenizer,
        _cache_fp="/remote-home/zyfei/project/tianchi/cache/nezha-5-1-fineturning-kfold",
        _refresh=False)

    skf = StratifiedKFold(n_splits=args.n_splits, random_state=42, shuffle=True)

    for fold, (train_index, val_index) in enumerate(skf.split(train_samples, train_labels)):
        print("FOLD | ", fold + 1)
        print("###" * 35)

        if model_type == "headwithmd":
            # NeZhaLabelHead for classifier with mutil dropout
            config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
            model = NeZhaForSequenceClassificationWithHeadClassMD.from_pretrained(model_name_or_path, config=config)
        elif model_type == "head":
            # only use NeZhaLabelHead for classifier
            config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
            model = NeZhaForSequenceClassificationWithHeadClass.from_pretrained(model_name_or_path, config=config)
        elif model_type == "clscat":
            # cat cls to classifier
            config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
            if args.classifier_dropout is not None:
                config.classifier_dropout_prob = args.classifier_dropout
            if args.hidden_dropout is not None:
                config.hidden_dropout_prob = args.hidden_dropout
            model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_name_or_path, config=config)
        else:
            raise NotImplementedError

        model.to(device)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

        if args.attack_method == "fgm":
            attack_model = FGM(model)
        elif args.attack_method == "pgd":
            attack_model = PGD(model)
            model.set_attack()
        elif args.attack_method == "pgd-md":
            attack_model = PGD(model)
        elif args.attack_method == "freelb":
            attack_model = FreeLB()
            model.set_attack()
        elif args.attack_method == "freelb-md":
            attack_model = FreeLB()
        elif args.attack_method == "mutildrop":
            attack_model = NoneAttack()
        else:
            raise NotImplementedError

        tokenizer.save_pretrained(fold_path)

        train_data = {"samples": train_samples[train_index],
                      "input_ids": train_input_ids[train_index],
                      "token_type_ids": train_token_type_ids[train_index],
                      "attention_mask": train_attention_masks[train_index],
                      "co_ocurrence_ids": train_co_ocurrence_ids[train_index],
                      "label": train_labels[train_index]}
        train_dataset = FastNLPDataSet(train_data)
        train_dataset.set_input("input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids", "label")
        train_dataset.set_target("label")

        dev_data = {"samples": train_samples[val_index],
                    "input_ids": train_input_ids[val_index],
                    "token_type_ids": train_token_type_ids[val_index],
                    "attention_mask": train_attention_masks[val_index],
                    "co_ocurrence_ids": train_co_ocurrence_ids[val_index],
                    "label": train_labels[val_index]}
        dev_dataset = FastNLPDataSet(dev_data)
        dev_dataset.set_input("input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids", "label")
        dev_dataset.set_target("label")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        args.max_train_steps = epochs * num_update_steps_per_epoch

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        best_result = 0
        best_model = None
        current_fold_path = os.path.join(fold_path, f'fold_co_{fold + 1}')
        for i in range(epochs):
            train(model, attack_model, train_dataset, optimizer, device, i, epochs, batch_size, lr_scheduler)
            if evalution_method == "auc":
                result = evaluate(model, dev_dataset, device, batch_size)
            else:
                result = evaluate_acc(model, dev_dataset, device, batch_size)

            print(f'fold:{fold + 1}, epoch:{i}, dev {evalution_method}:{result}, test {evalution_method}:{result}')
            fitlog.add_metric({"dev": {f"{fold + 1}-dev-{evalution_method}": result}}, step=i)

            if result > best_result:
                best_result = result
                best_model = model
                model.save_pretrained(current_fold_path)
                # torch.save(model.state_dict(), current_fold_path)
        print(f'fold:{fold + 1}, best test result:{best_result}')

        onnx_file = os.path.join(fold_path, f"model-{str(fold)}.onnx")

        input_dict = (torch.tensor([train_input_ids[0]], device=device),
                      {
                          'token_type_ids': torch.tensor([train_token_type_ids[0]], device=device),
                          'co_ocurrence_ids': torch.tensor([train_co_ocurrence_ids[0]], device=device)
                      })

        input_names = ["input_ids", "token_type_ids", "co_ocurrence_ids"]
        output_names = ["logtis"]

        onnx_model = NeZhaForSequenceClassificationWithClsCatForOnnx.from_pretrained(current_fold_path)
        onnx_model.to(device)

        torch.onnx.export(onnx_model, input_dict, onnx_file, verbose=True, input_names=input_names,
                          output_names=output_names, dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                                                   'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                                                                   'co_ocurrence_ids': {0: 'batch_size',
                                                                                        1: 'sequence'}},
                          opset_version=10)

        fitlog.add_best_metric(str(best_result), f"fold-{fold + 1} {evalution_method}")

    fitlog.finish()


if __name__ == '__main__':
    run()
