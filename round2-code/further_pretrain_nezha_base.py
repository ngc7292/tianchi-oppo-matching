# -*- coding: utf-8 -*-
"""
__title__="further_pretrain_nezha"
__author__="ngc7293"
__mtime__="2021/4/15"
"""
import os
import numpy as np
import torch
import random
import tqdm
from typing import Dict

from fastNLP import cache_results
from transformers import BertTokenizer, PreTrainedTokenizer
from torch.utils.data.dataset import Dataset
from modeling_nezha import NeZhaForMaskedLM, NeZhaForPreTrainingWithLabel
from configuration_nezha import NeZhaConfig
from transformers import Trainer, TrainingArguments

from transformers import AlbertForSequenceClassification
from DataCollator import DataCollatorForLanguageModelingNgram

# vocab_file = './nezha_base_vocab.txt'  # vocab file

raw_text = './data/train-dual.tsv' # line by line file
# raw_text = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
# raw_text = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv'

# model_name_or_path = "/remote-home/zyfei/project/tianchi/output/nezha-pretrained-round1/checkpoint-30000"
# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_output_with_label"
model_name_or_path = "/remote-home/zyfei/project/tianchi/models/nezha-base-www"
# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_with_label_2/checkpoint-10000"
tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1"

new_model_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1_v2"


cache_path = "/remote-home/zyfei/project/tianchi/cache/nezha-base-4-28"


class LineByLineTextDataset(Dataset):
    """
    This dataset is
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):

        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        lines_1 = []
        lines_2 = []
        with open(file_path, encoding="utf-8") as f:
            # lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            for line in tqdm.tqdm(f.read().splitlines(), desc="loading data", leave=False):
                if len(line) > 0 and not line.isspace():
                    text1, text2, label = line.split("\t")
                    lines_1.append(text1)
                    lines_2.append(text2)

        batch_encoding = tokenizer(text=lines_1, text_pair=lines_2, add_special_tokens=True, truncation=True,
                                   max_length=block_size)

        example_ids = batch_encoding["input_ids"]
        example_tokens = batch_encoding["token_type_ids"]
        self.examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long),
             "token_type_ids": torch.tensor(t, dtype=torch.long)} for e, t in zip(example_ids, example_tokens)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


# using cache_result increase speed of loadding data if want to change cache use _refresh=True
@cache_results(_cache_fp=cache_path, _refresh=True)
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=32  # maximum sequence length
    )


tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# 转移权重
config = NeZhaConfig.from_pretrained(model_name_or_path)
config.vocab_size = tokenizer.vocab_size
model = NeZhaForMaskedLM(config=config)
# model.resize_token_embeddings(tokenizer.vocab_size)

# pretrained_dict = torch.load("/remote-home/zyfei/project/tianchi/output/nezha-pretrained-round1/checkpoint-30000/pytorch_model.bin")
# pretrained_model = NeZhaForPreTrainingWithLabel.from_pretrained(model_name_or_path)
pretrained_model = NeZhaForMaskedLM.from_pretrained(model_name_or_path)
pretrained_model.resize_token_embeddings(tokenizer.vocab_size)
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()

for k,v in pretrained_dict.items():
    if k in model_dict:
        if v.size() == model_dict[k].size():
            model_dict[k] = v

model.load_state_dict(model_dict)

# model = NeZhaForPreTrainingWithLabel.from_pretrained(model_name_or_path, config=config)

# model.resize_token_embeddings(tokenizer.vocab_size)

dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text)

tokenizer.save_pretrained(new_model_path)

random_seed = 2021
random.seed(random_seed)
np.random.seed(random_seed)
torch.random.manual_seed(random_seed)

data_collator = DataCollatorForLanguageModelingNgram(
    data_tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=new_model_path,
    overwrite_output_dir=True,
    num_train_epochs=300,
    per_device_train_batch_size=256,
    save_steps=10_000,
    learning_rate=5e-5,
    max_steps=60000,
    dataloader_num_workers=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("*" * 35)
print("traing...")
trainer.train()

trainer.save_model(new_model_path)
