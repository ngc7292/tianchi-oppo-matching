# -*- coding: utf-8 -*-
"""
__title__="get_word"
__author__="ngc7293"
__mtime__="2021/3/11"
"""
import torch
from transformers import BertTokenizer, BertForMaskedLM

from model import ERINE_Matching

tokenizer = BertTokenizer.from_pretrained('nghuyong/ernie-1.0')

# model = BertForMaskedLM.from_pretrained('nghuyong/ernie-1.0')

model = torch.load(
    "/remote-home/zyfei/project/tianchi/data_process/model/best_ERINE_Matching_acc_2021-03-15-03-49-16-452614")


# model = torch.load("/remote-home/zyfei/project/tianchi/data_process/model/best_ERINE_Matching_acc_2021-03-11-09-14-40-595158")

def get_mask_word(text_list, id_list):
    """
    :param text_list: like ['你', ' ', 14, 15]
    :param word_type: eeach word get the charcter
    :return:
    """
    # input_tx = "[CLS] [MASK] [MASK] [MASK] 是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。[SEP]"
    input_list = ["[CLS]"]
    input_list.extend(text_list)
    input_list.append("[SEP]")
    id_list = id_list[::-1]

    # input_tx = "[CLS] [MASK]是什么意思 [SEP]"
    input_tx = "".join(input_list)
    tokenized_text = tokenizer.tokenize(input_tx)
    # attention_mask = [i == "[MASK]" for i in tokenized_text]
    attention_mask = []
    type_list = []
    for i in tokenized_text:
        if i == "[MASK]":
            attention_mask.append(0)
            while True:
                char = id_list.pop()
                if char != -1:
                    type_list.append(char)
                    break
        else:
            type_list.append(-1)
            attention_mask.append(1)


    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    attention_mask = torch.tensor([attention_mask])

    segments_tensors = torch.tensor([[0] * len(tokenized_text)])

    # model = torch.load("/remote-home/zyfei/project/tianchi/data_process/model/best_ERINE_Matching_acc_2021-03-11-09-14-40-595158")
    model.eval()

    with torch.no_grad():
        outputs = model.predict(token_ids=tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)['pred']
        # outputs = model(tokens_tensor, token_type_ids=segments_tensors, attention_mask=attention_mask)
        predictions = outputs

    predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0, (len(tokenized_text)))]
    predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in
                       range(0, (len(tokenized_text)))]

    # print('Predicted token is:', "".join(predicted_token))

    result = {}
    for i, char_type in zip(predicted_token, type_list):
        if char_type == -1:
            continue
        else:
            if char_type not in result:
                result[char_type] = []
            result[char_type].append(i)
    for i in result:
        result[i] = "".join(result[i])
    return result


def replace_id_to_mask(text):
    replace_result = [[]]
    id_list = [[]]
    for i, t in enumerate(text):
        if isinstance(t, int):
            l = len(replace_result)
            for j in range(l):
                replace_result[j].append("[MASK]")
                id_list[j].append(t)
                t_list = replace_result[j].copy()
                t_ids = id_list[j].copy()
                t_list.append("[MASK]")
                t_ids.append(t)
                replace_result.append(t_list)
                id_list.append(t_ids)
        else:
            for ins in range(len(replace_result)):
                replace_result[ins].append(t)
                id_list[ins].append(-1)

    return replace_result, id_list




if __name__ == '__main__':
    # a = get_mask_word(text_list=['喜欢', '吧', '怎么', 11, '了'], word_type=2)
    # print(a)
    a = replace_id_to_mask(['手机', '能', '能', '手机', 'OPPO', '没有', '帮', '打开', 49, 159, 622])
    for i, j in zip(a[0], a[1]):
        d = get_mask_word(text_list=i, id_list=j)
        print(d)
