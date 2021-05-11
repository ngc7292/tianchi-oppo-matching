# -*- coding: utf-8 -*-
"""
__title__="predict"
__author__="ngc7293"
__mtime__="2021/3/28"
"""
import os
import torch
import numpy as np

from tqdm import tqdm
from furnturning_nezha import MatchingDataset, load_data, DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding
from modeling_electra import ElectraForSequenceClassification

test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

# vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence_by_zzdu/tokens_ngram.txt'

vocab_file = '/remote-home/zyfei/project/tianchi/baseline_nezha_with_token_coocurrence/tokens.txt'

is_co_ocurrence = True


def predict_test(pred_model, test_data_loader, run_device):
    pred_model.eval()
    pred = []
    with torch.no_grad():
        for batch_index, batch in tqdm(enumerate(test_data_loader),leave=False):
            input_ids = batch['input_ids'].to(run_device)
            attention_mask = batch['attention_mask'].to(run_device)

            model_output = pred_model(input_ids, attention_mask)[0]

            model_output = model_output.cpu().numpy()
            pred.append(model_output)
    pred = np.concatenate(pred)
    return pred


def run_kfold_pred(test_dataloader, test_data, model_dir, device, is_save_transformer=True):
    """
    This function for k fold evaluation training predict
    :param test_dataloader:
    :param test_data:
    :return:
    """
    model_path_list = os.listdir(model_dir)
    k = len(model_path_list)
    model_pred = np.zeros((len(test_data), 2))
    for model_path in model_path_list:
        model_path = os.path.join(model_dir, model_path)
        # test_model.load_state_dict(torch.load('./kfold/fold_{i}.pkl'))
        if is_save_transformer is True:
            test_model = ElectraForSequenceClassification.from_pretrained(model_path)
            # test_model = NeZhaForSequenceClassificationWithClSCat.from_pretrained(model_path)
        else:
            test_model = torch.load(model_path)
        test_model.to(device)
        model_output = predict_test(test_model, test_dataloader, device)
        model_output = np.exp(model_output) / (np.exp(model_output).sum(axis=1, keepdims=True))
        model_pred += model_output
    model_pred = model_pred / k
    model_pred = model_pred[:, 1] / model_pred.sum(axis=1)
    return model_pred


test_texts, _, _ = load_data(test_path)

tokenizer = BertTokenizer(vocab_file=vocab_file)

data_collator = DataCollatorWithPadding(tokenizer)

test_dataset = MatchingDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=data_collator)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


pred = run_kfold_pred(test_loader, test_texts, device=device, model_dir='./kfold-9')

with open('./result-9.txt', 'w') as f:
    for val in pred:
        f.write(str(val) + '\n')
