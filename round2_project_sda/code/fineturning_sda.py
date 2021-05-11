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
import logging
import math

logging.disable(0)

import numpy as np

from fastNLP import DataSet as FastNLPDataSet
from fastNLP.core import Instance, DataSetIter, RandomSampler, cache_results
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from modeling_nezha import NeZhaForSequenceClassification, NeZhaConfig
from modeling_bert import BertForSequenceClassification, BertConfig
from modeling_roberta import RobertaForSequenceClassification, RobertaConfig

from modeling_sda import SDAModel

from transformers import BertTokenizer, get_scheduler

class FGM():
    def __init__(self, model):
        self.name = "fgm"
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.2, emb_name='word_embeddings.'):

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and "teacher" not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and "teacher" not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class NoneAttack():
    def __init__(self):
        self.name = 'none'


def tokenize_data(text_1, text_2, tokenizer):
    sample = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True, max_length=64)
    return {
        "input_ids": sample.input_ids,
        "token_type_ids": sample.token_type_ids,
        "attention_mask": sample.attention_mask
    }


def load_data(path, tokenizer):
    samples = []
    input_ids = []
    token_type_ids = []
    attention_masks = []

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

    return np.array(samples), np.array(input_ids), np.array(token_type_ids), np.array(attention_masks), np.array(labels)


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
            labels = batch['label'].to(device)

            if attack_model.name == "fgm":
                loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     labels=labels)[:2]
                loss.backward()
                attack_model.attack()
                loss, logits = model(input_ids, attention_mask, labels=labels)[:2]
                loss.backward()
                attack_model.restore()  # 恢复embedding参数
            elif attack_model.name == "none":
                loss, logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     labels=labels)[:2]
                loss.backward()
            else:
                raise NotImplementedError

            optimizer.step()
            model.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            model.update_teacher()

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
            output = eval_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            y_pred.append(output_score[:, 1] / output_score.sum(axis=1))
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true, y_pred)


def run():
    transformers.logging.set_verbosity_error()

    train_path = "../tcdata/gaiic_track3_round2_train_20210407.tsv"

    epochs = 10
    lr = 1e-5
    batch_size = 112
    attack_method = "fgm"
    n_split = 4
    onnx_path = "../onnx_models"

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer_path = "../origin_model/tokenizer"
    nezha_model_path = "../model_output/nezha"
    roberta_model_path = "../model_output/roberta"
    bert_model_path = "../model_output/bert"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    train_samples, train_input_ids, train_token_type_ids, train_attention_masks, train_labels = load_data(train_path,
                                                                                                          tokenizer=tokenizer)

    skf = StratifiedKFold(n_splits=n_split, random_state=42, shuffle=True)

    for fold, (train_index, val_index) in enumerate(skf.split(train_samples, train_labels)):
        print("FOLD | ", fold)

        train_data = {"samples": train_samples[train_index],
                      "input_ids": train_input_ids[train_index],
                      "token_type_ids": train_token_type_ids[train_index],
                      "attention_mask": train_attention_masks[train_index],
                      "label": train_labels[train_index]}
        train_dataset = FastNLPDataSet(train_data)
        train_dataset.set_input("input_ids", "token_type_ids", "attention_mask", "label")
        train_dataset.set_target("label")

        dev_data = {"samples": train_samples[val_index],
                    "input_ids": train_input_ids[val_index],
                    "token_type_ids": train_token_type_ids[val_index],
                    "attention_mask": train_attention_masks[val_index],
                    "label": train_labels[val_index]}
        dev_dataset = FastNLPDataSet(dev_data)
        dev_dataset.set_input("input_ids", "token_type_ids", "attention_mask", "label")
        dev_dataset.set_target("label")

        if fold == 0:
            stu_model = NeZhaForSequenceClassification.from_pretrained(nezha_model_path)
        elif fold == 1:
            stu_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)
        elif fold == 2:
            stu_model = BertForSequenceClassification.from_pretrained(bert_model_path)
        else:
            stu_model = NeZhaForSequenceClassification.from_pretrained(nezha_model_path)

        model = SDAModel(stu_model)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        max_train_steps = epochs * num_update_steps_per_epoch

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=1,
            num_training_steps=max_train_steps,
        )

        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        if attack_method == "fgm":
            attack_model = FGM(model)
        else:
            attack_model = NoneAttack()

        model.to(device)

        best_result = 0.0
        best_model = None
        for i in range(epochs):
            train(model, attack_model, train_dataset, optimizer, device, i, epochs, batch_size, lr_scheduler)
            result = evaluate(model, dev_dataset, device, batch_size)
            print(result)

            if result > best_result:
                best_result = result
                best_model = model
                # torch.save(model.state_dict(), current_fold_path)
        print(f'fold:{fold}, best test result:{best_result}')

        # convert model to onnx types

        onnx_file = os.path.join(onnx_path, f"fold-{str(fold)}.onnx")

        input_dict = (
            torch.tensor([train_input_ids[0]], device=device),
            torch.tensor([train_token_type_ids[0]], device=device),
        )

        input_names = ["input_ids", "token_type_ids"]
        output_names = ["logtis"]

        torch.onnx.export(best_model,
                          input_dict,
                          onnx_file,
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                                        'token_type_ids': {0: 'batch_size', 1: 'sequence'}},
                          opset_version=11)


if __name__ == '__main__':
    run()
