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
from modeling_nezha import NeZhaForSequenceClassificationWithHeadClass
from configuration_nezha import NeZhaConfig
from transformers import BertTokenizer
from torch.nn import MSELoss
from torch import log_softmax, softmax
from torch.nn.functional import mse_loss

fitlog.set_log_dir("/remote-home/zyfei/project/tianchi/round2-code/logs")


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


def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = softmax(beta_logits_T, dim=-1)
    loss = -(p_T * log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def hid_mse_loss(state_S, state_T, mask=None):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.
    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        loss = mse_loss(state_S, state_T)
    else:
        mask = mask.to(state_S.device)
        valid_count = mask.sum() * state_S.size(-1)
        loss = (mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss


def distill_train(teacher_model, model, train_dataset, optimizer, device, device_t, epoch=0, epochs=0,
                  batch_size=64, kd_weight=1, hard_label_weight=1):
    model.train()  # Turn on the train mode
    teacher_model.eval()
    start_time = time.time()
    samplr = RandomSampler()
    train_loader = DataSetIter(dataset=train_dataset, batch_size=batch_size, sampler=samplr)
    with tqdm.tqdm(train_loader, leave=False) as phar:
        phar.set_description("epoch " + str(epoch) + "/" + str(epochs))
        for batch, _ in phar:
            optimizer.zero_grad()
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device_t)
                attention_mask = batch['attention_mask'].to(device_t)
                token_type_ids = batch['token_type_ids'].to(device_t)
                co_ocurrence_ids = batch['co_ocurrence_ids'].to(device_t)
                labels = batch['label'].to(device_t)

                loss_teacher, teacher_logits, teacher_state = teacher_model(input_ids, attention_mask=attention_mask,
                                                                            token_type_ids=token_type_ids,
                                                                            co_ocurrence_ids=co_ocurrence_ids,
                                                                            labels=labels)
            teacher_logits = teacher_logits.to(device)
            teacher_states = tuple(teacher_state[i].to(device) for i in range(1, len(teacher_state), 2))

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            labels = batch['label'].to(device)

            student_loss, student_logits, student_states = model(input_ids, attention_mask=attention_mask,
                                                                 token_type_ids=token_type_ids,
                                                                 co_ocurrence_ids=co_ocurrence_ids, labels=labels)

            state_loss = 0
            state_count = 0
            for student_state, teacher_state in zip(student_states[1:], teacher_states):
                state_loss += hid_mse_loss(student_state, teacher_state, mask=attention_mask)
                state_count += 1
            state_loss = state_loss / state_count
            kd_loss = kd_ce_loss(student_logits, teacher_logits)

            loss = state_loss + kd_loss*kd_weight + student_loss*hard_label_weight

            loss.backward()
            optimizer.step()

            phar.set_postfix({"loss": float(loss.cpu().detach())})

    spend_time = time.time() - start_time
    print("model traing finish and use time " + str(spend_time) + " s")


def evaluate(teacher_model, eval_model, val_dataset, device, batch_size=64):
    teacher_model.eval()
    eval_model.eval()

    sampler = RandomSampler()
    val_loader = DataSetIter(val_dataset, batch_size=batch_size, sampler=sampler)

    y_pred = []
    y_true = []
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
    student_auc = roc_auc_score(y_true, y_pred)

    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch, _ in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(device)
            output = teacher_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            y_pred.append(output_score[:, 1] / output_score.sum(axis=1))
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    teacher_auc = roc_auc_score(y_true, y_pred)
    return teacher_auc, student_auc


def evaluate_acc(teacher_model, eval_model, val_dataset, device, batch_size=64):
    teacher_model.eval()
    eval_model.eval()

    sampler = RandomSampler()
    val_loader = DataSetIter(val_dataset, batch_size=batch_size, sampler=sampler)

    y_pred = []
    y_true = []
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
            pred_label = np.argmax(output_score, axis=-1)
            y_pred.append(pred_label)
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    student_acc = accuracy_score(y_true, y_pred)

    teacher_device = teacher_model.device
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch, _ in tqdm.tqdm(val_loader, leave=False):
            input_ids = batch['input_ids'].to(teacher_device)
            attention_mask = batch['attention_mask'].to(teacher_device)
            token_type_ids = batch['token_type_ids'].to(teacher_device)
            co_ocurrence_ids = batch['co_ocurrence_ids'].to(teacher_device)
            output = teacher_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   co_ocurrence_ids=co_ocurrence_ids)[0]

            output = output.cpu().numpy()
            # output_score = np.exp(output[:, 1])/ (np.exp(output).sum(axis=1))
            output_score = np.exp(output) / (np.exp(output).sum(axis=1, keepdims=True))
            pred_label = np.argmax(output_score, axis=-1)
            y_pred.append(pred_label)
            y_true.append(batch['label'].numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    teacher_acc = accuracy_score(y_true, y_pred)
    return student_acc, teacher_acc


def run():
    transformers.logging.set_verbosity_error()

    train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv'

    # tokenizer_file = '/remote-home/zyfei/project/tianchi/model_output/nezha_output_2'
    tokenizer_file = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_with_label_2"

    epochs = 10
    lr = 1e-5
    batch_size = 128
    fold_path = "./model_9"
    evalution_method = "acc"

    fitlog.add_hyper(fold_path, "fold_path")

    random_seed = 2021
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_with_label_2/checkpoint-50000"
    teacher_model_path_1 = "./model_8"
    fitlog.add_hyper(model_name_or_path, "model_name_or_path")
    fitlog.add_hyper(teacher_model_path_1, "teacher model path")

    tokenizer = BertTokenizer.from_pretrained(tokenizer_file,
                                              model_input_names=["input_ids", "attention_mask", "token_type_ids",
                                                                 "co_ocurrence_ids"])

    device = torch.device('cuda:0')  # if torch.cuda.is_available() else torch.device('cpu')
    device_t = torch.device('cuda:1')

    train_dataset, dev_dataset = load_data_fastnlp(train_path, tokenizer).split(0.3, shuffle=True)

    config = NeZhaConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
    config.num_hidden_layers = 6  # student nezha is 6 layers
    model = NeZhaForSequenceClassificationWithHeadClass.from_pretrained(model_name_or_path, config=config)
    model.to(device)

    teacher_model = NeZhaForSequenceClassificationWithHeadClass.from_pretrained(teacher_model_path_1)
    teacher_model.to(device_t)  # teacher nezha is 12 layers or 24 layers

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_result = 0
    best_model = None
    for i in range(epochs):
        distill_train(teacher_model, model, train_dataset, optimizer, device, device_t, i, epochs, batch_size)
        if evalution_method == "auc":
            student_result, teacher_result = evaluate(teacher_model, model, dev_dataset, device, batch_size)
        else:
            student_result, teacher_result = evaluate_acc(teacher_model, model, dev_dataset, device, batch_size)
        print(f'epoch:{i},student {evalution_method}:{student_result}, teacher {evalution_method}:{teacher_result}')
        fitlog.add_metric({"dev": {f"student {evalution_method}": str(student_result),
                                   f"teacher {evalution_method}": str(teacher_result)}}, step=i)
        if student_result > best_result:
            best_result = student_result
            best_model = model
            model.save_pretrained(fold_path)
            # torch.save(model.state_dict(), current_fold_path)
    print(f'best result:{best_result}')
    best_model.save_pretrained(fold_path)
    fitlog.add_best_metric(str(best_result), evalution_method)
    fitlog.finish()


if __name__ == '__main__':
    run()
