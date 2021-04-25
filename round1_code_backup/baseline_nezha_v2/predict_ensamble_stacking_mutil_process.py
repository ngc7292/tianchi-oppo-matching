# -*- coding: utf-8 -*-
"""
__title__="predict_ensamble"
__author__="ngc7293"
__mtime__="2021/4/6"
"""
import os
import torch
import json
import numpy as np
# import multiprocessing

from tqdm import tqdm
from furnturning_nezha import MatchingDataset, load_data, DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding
from modeling_nezha import NeZhaForSequenceClassificationWithClsCat
from modeling_electra import ElectraForSequenceClassificationWithClsCat
from modeling_roberta import BertForSequenceClassificationClsCat

from torch import nn
# from multiprocessing import Process
from torch.multiprocessing import Process
from torch import multiprocessing

train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

# vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence_by_zzdu/tokens_ngram.txt'

nezha_vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence/tokens.txt'
electra_vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence/tokens.txt'
roberta_vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_v2/roberta_tokens.txt'

is_co_ocurrence = True


def predict_test(pred_model, test_data_loader, run_device):
    pred_model.eval()
    pred = []
    with torch.no_grad():
        for batch_index, batch in enumerate(test_data_loader):
            input_ids = batch['input_ids'].to(run_device)
            attention_mask = batch['attention_mask'].to(run_device)

            co_ocurrence_ids = batch['token_type_ids'].to(run_device)
            model_output = pred_model(input_ids, attention_mask, co_ocurrence_ids=co_ocurrence_ids)[0]

            model_output = model_output.cpu().numpy()
            pred.append(model_output)
    pred = np.concatenate(pred)
    return pred


def run_nezha_kfold_pred(test_dataloader, test_data, model_dir, device, is_save_transformer=True):
    """
    This function for k fold evaluation training predict
    :param test_dataloader:
    :param test_data:
    :return:
    """
    model_path_list = os.listdir(model_dir)
    k = len(model_path_list)
    # model_pred = np.zeros((len(test_data), 2))
    model_pred = None
    for model_path in model_path_list:
        model_path = os.path.join(model_dir, model_path)
        # test_model.load_state_dict(torch.load('./kfold/fold_{i}.pkl'))
        if is_save_transformer is True:
            test_model = NeZhaForSequenceClassificationWithClsCat.from_pretrained(model_path)
            # test_model = NeZhaForSequenceClassificationWithClSCat.from_pretrained(model_path)
        else:
            test_model = torch.load(model_path)
        test_model.to(device)
        model_output = predict_test(test_model, test_dataloader, device)
        model_softmax_output = np.exp(model_output) / (np.exp(model_output).sum(axis=1, keepdims=True))  # softmax
        model_softmax_output = model_softmax_output[:, 1] / model_softmax_output.sum(axis=1)
        model_softmax_output = np.expand_dims(model_softmax_output, axis=-1)

        if model_pred is None:
            model_pred = model_softmax_output
        else:
            model_pred = np.concatenate([model_pred, model_softmax_output], axis=1)
        # break
    # model_pred = model_pred / k
    # model_pred = model_pred[:, 1] / model_pred.sum(axis=1)
    return model_pred


def run_electra_kfold_pred(test_dataloader, test_data, model_dir, device, is_save_transformer=True):
    """
    This function for k fold evaluation training predict
    :param test_dataloader:
    :param test_data:
    :return:
    """
    model_path_list = os.listdir(model_dir)
    k = len(model_path_list)
    model_pred = None
    for model_path in model_path_list:
        model_path = os.path.join(model_dir, model_path)
        # test_model.load_state_dict(torch.load('./kfold/fold_{i}.pkl'))
        if is_save_transformer is True:
            test_model = ElectraForSequenceClassificationWithClsCat.from_pretrained(model_path)
            # test_model = NeZhaForSequenceClassificationWithClSCat.from_pretrained(model_path)
        else:
            test_model = torch.load(model_path)
        test_model.to(device)
        model_output = predict_test(test_model, test_dataloader, device)
        model_softmax_output = np.exp(model_output) / (np.exp(model_output).sum(axis=1, keepdims=True))  # softmax
        model_softmax_output = model_softmax_output[:, 1] / model_softmax_output.sum(axis=1)
        model_softmax_output = np.expand_dims(model_softmax_output, axis=-1)

        if model_pred is None:
            model_pred = model_softmax_output
        else:
            model_pred = np.concatenate([model_pred, model_softmax_output], axis=1)
    # model_pred = model_pred / k
    # model_pred = model_pred[:, 1] / model_pred.sum(axis=1)
    return model_pred


def run_roberta_kfold_pred(test_dataloader, test_data, model_dir, device, is_save_transformer=True):
    """
    This function for k fold evaluation training predict
    :param test_dataloader:
    :param test_data:
    :return:
    """
    model_path_list = os.listdir(model_dir)
    k = len(model_path_list)
    model_pred = None
    for model_path in model_path_list:
        model_path = os.path.join(model_dir, model_path)
        # test_model.load_state_dict(torch.load('./kfold/fold_{i}.pkl'))
        if is_save_transformer is True:
            test_model = BertForSequenceClassificationClsCat.from_pretrained(model_path)
            # test_model = NeZhaForSequenceClassificationWithClSCat.from_pretrained(model_path)
        else:
            test_model = torch.load(model_path)
        test_model.to(device)
        model_output = predict_test(test_model, test_dataloader, device)
        model_softmax_output = np.exp(model_output) / (np.exp(model_output).sum(axis=1, keepdims=True))  # softmax
        model_softmax_output = model_softmax_output[:, 1] / model_softmax_output.sum(axis=1)
        model_softmax_output = np.expand_dims(model_softmax_output, axis=-1)

        if model_pred is None:
            model_pred = model_softmax_output
        else:
            model_pred = np.concatenate([model_pred, model_softmax_output], axis=1)
    # model_pred = model_pred / k
    # model_pred = model_pred[:, 1] / model_pred.sum(axis=1)
    return model_pred


batch_size = 256
train_texts, train_labels, _ = load_data(train_path)
test_texts, _, _ = load_data(test_path)

nezha_tokenizer = BertTokenizer(vocab_file=nezha_vocab_file,
                                model_input_names=["input_ids", "token_type_ids", "attention_mask", "co_ocurrence_ids"])
electra_tokenizer = BertTokenizer(vocab_file=electra_vocab_file,
                                  model_input_names=["input_ids", "token_type_ids", "attention_mask",
                                                     "co_ocurrence_ids"])
roberta_tokenizer = BertTokenizer(vocab_file=roberta_vocab_file,
                                  model_input_names=["input_ids", "token_type_ids", "attention_mask",
                                                     "co_ocurrence_ids"])

nezha_data_collator = DataCollatorWithPadding(nezha_tokenizer)
nezha_train_dataset = MatchingDataset(train_texts, None, nezha_tokenizer)
nezha_test_dataset = MatchingDataset(test_texts, None, nezha_tokenizer)
nezha_train_loader = DataLoader(nezha_train_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=nezha_data_collator)
nezha_test_loader = DataLoader(nezha_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=nezha_data_collator)

electra_data_collator = DataCollatorWithPadding(electra_tokenizer)
electra_train_dataset = MatchingDataset(train_texts, None, electra_tokenizer)
electra_test_dataset = MatchingDataset(test_texts, None, electra_tokenizer)
electra_train_loader = DataLoader(electra_train_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=electra_data_collator)
electra_test_loader = DataLoader(electra_test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=electra_data_collator)

roberta_data_collator = DataCollatorWithPadding(roberta_tokenizer)
roberta_train_dataset = MatchingDataset(train_texts, None, roberta_tokenizer)
roberta_test_dataset = MatchingDataset(test_texts, None, roberta_tokenizer)
roberta_train_loader = DataLoader(roberta_train_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=roberta_data_collator)
roberta_test_loader = DataLoader(roberta_test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=roberta_data_collator)

device_0 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device_1 = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
device_2 = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


def run_proc(train_data_loader, test_data_loader, train_text, test_text, device, model_dir, run_func, i):
    train_save_path = f"./train_enhance_data/train_{i}.txt"
    test_save_path = f"./train_enhance_data/train_{i}.txt"
    print(f"running {i} models which model_path is {model_dir}, device is {device}, saving in {train_save_path}")
    train_pred = run_func(train_data_loader, train_text, device=device, model_dir=model_dir)
    test_pred = run_func(test_data_loader, test_text, device=device, model_dir=model_dir)
    print(f"{i} models running is down")

    with open(train_save_path, "w") as f:
        f.write(json.dumps(train_pred.tolist()))

    with open(test_save_path, "w") as f:
        f.write(json.dumps(test_pred.tolist()))


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    p_1 = Process(target=run_proc, args=(
        nezha_train_loader, nezha_test_loader, train_texts, test_texts, device_0, './kfold-11', run_nezha_kfold_pred,
        1))

    p_2 = Process(target=run_proc, args=(
        electra_train_loader, electra_test_loader, train_texts, test_texts, device_1, './kfold-16',
        run_electra_kfold_pred,
        2))

    p_3 = Process(target=run_proc, args=(
        roberta_train_loader, roberta_test_loader, train_texts, test_texts, device_2, './kfold-14',
        run_roberta_kfold_pred,
        3))

    p_1.start()
    p_1.join()

    p_2.start()
    p_2.join()

    p_3.start()
    p_3.join()

# train_1 = run_nezha_kfold_pred(nezha_train_loader, train_texts, device=device, model_dir='./kfold-11')
# train_2 = run_electra_kfold_pred(electra_train_loader, train_texts, device=device, model_dir='./kfold-16')
# train_3 = run_roberta_kfold_pred(roberta_train_loader, train_texts, device=device, model_dir='./kfold-14')
#
# test_1 = run_nezha_kfold_pred(nezha_test_loader, test_texts, device=device, model_dir='./kfold-11')
# test_2 = run_electra_kfold_pred(electra_test_loader, test_texts, device=device, model_dir='./kfold-16')
# test_3 = run_roberta_kfold_pred(roberta_test_loader, test_texts, device=device, model_dir='./kfold-14')
#
# train_data = np.concatenate([train_1, train_2, train_3], axis=-1)
# test_data = np.concatenate([test_1, test_2, test_3], axis=-1)
#
# with open("./train_enhance_data/train_x.txt", "w") as f:
#     f.write(json.dumps(train_data.tolist()))
#
# with open("./train_enhance_data/train_y.txt", "w") as f:
#     f.write(json.dumps(train_labels.tolist()))
#
# with open("./train_enhance_data/test.txt", "w") as f:
#     f.write(json.dumps(test_data.tolist()))
