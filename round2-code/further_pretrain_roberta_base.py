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
from modeling_roberta import RobertaForMaskedLM
from modeling_roberta import RobertaConfig
from transformers import Trainer, TrainingArguments

from transformers import AlbertForSequenceClassification
from DataCollator import DataCollatorForLanguageModelingNgram

# vocab_file = '/remote-home/zyfei/project/tianchi/round2-code/raw_text/roberta_base_vocab.txt'  # vocab file

# raw_text = './data/train_true.tsv'  # line by line file
# raw_text = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
raw_text = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round2_train_20210407.tsv'
raw_text_clean = '/remote-home/zyfei/project/tianchi/round2-code/data/train_clean.tsv'
# model_name_or_path = "/remote-home/zyfei/project/tianchi/output/nezha-pretrained-round1/checkpoint-30000"
# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_output_with_label"
# model_name_or_path = "/remote-home/zyfei/project/tianchi/models/nezha-base-www"
# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_without_round1_v3/checkpoint-30000"
tokenizer_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_5_2_v3_clean_round2data/checkpoint-20000"

# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30"
# /remote-home/zyfei/project/tianchi/models/chinese-roberta-wwm-ext
model_name_or_path = "/remote-home/zyfei/project/tianchi/models/chinese-roberta-wwm-ext"
new_model_path = "/remote-home/zyfei/project/tianchi/model_output/roberta_base_output_5_3_clean_round2data"

# model_name_or_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30_v2_round2data/checkpoint-20000"
# new_model_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_base_output_4_30_v2_round2data_1"

cache_path_clean = "/remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_cleandata"
cache_path_2 = "/remote-home/zyfei/project/tianchi/cache/nezha-base-5-2_v2_round2data"

print(
    f"data path is {raw_text}, origin model is {model_name_or_path}, and trained model saved {new_model_path}, data cache is {cache_path_clean}, {cache_path_2}, tokenizer is {tokenizer_path}")


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

        self.tokenizer = tokenizer
        self.block_size = block_size

        # batch_encoding = tokenizer(text=lines_1, text_pair=lines_2, add_special_tokens=True, truncation=True,
        #                            max_length=block_size)
        #
        # example_ids = batch_encoding["input_ids"]
        # example_tokens = batch_encoding["token_type_ids"]
        self.examples = [
            {"text_1": e,
             "text_2": t} for e, t in zip(lines_1, lines_2)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        text_1 = self.examples[i]["text_1"]
        text_2 = self.examples[i]["text_2"]

        batch_encoding = tokenizer(text=text_1, text_pair=text_2, add_special_tokens=True, truncation=True,
                                   max_length=self.block_size)
        result = {
            "input_ids": batch_encoding["input_ids"],
            "token_type_ids": batch_encoding["token_type_ids"]
        }
        return result


class LineByLineTextDataset_2(Dataset):
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
@cache_results(_cache_fp=cache_path_clean, _refresh=False)
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=64  # maximum sequence length
    )


@cache_results(_cache_fp=cache_path_2, _refresh=False)
def load_data_2(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=64  # maximum sequence length
    )


tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

model = RobertaForMaskedLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(tokenizer.vocab_size)
dataset_clean = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text_clean)
dataset = load_data_2(data_tokenizer=tokenizer, raw_text_path=raw_text)
tokenizer.save_pretrained(new_model_path)

random_seed = 42
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
    max_steps=10000,
    dataloader_num_workers=16,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_clean,
    tokenizer=tokenizer
)

print("*" * 35)
print("traing...")
trainer.train()

training_args_2 = TrainingArguments(
    output_dir=new_model_path,
    overwrite_output_dir=True,
    num_train_epochs=300,
    per_device_train_batch_size=256,
    save_steps=10_000,
    learning_rate=5e-5,
    max_steps=60000,
    dataloader_num_workers=16,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args_2,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
)

print("*" * 35)
print("traing...")
trainer.train(resume_from_checkpoint=True)

trainer.save_model(new_model_path)
