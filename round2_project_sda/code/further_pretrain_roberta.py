# -*- coding: utf-8 -*-
"""
__title__="further_pretrain_nezha"
__author__="ngc7293"
__mtime__="2021/4/30"
"""
import os
import time
import numpy as np
import torch
import random
import tqdm
from typing import Dict
import transformers

from transformers import BertTokenizer, PreTrainedTokenizer
from torch.utils.data.dataset import Dataset
from modeling_roberta import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments

from DataCollator import DataCollatorForLanguageModelingNgram

raw_text = '../tcdata/gaiic_track3_round2_train_20210407.tsv'

tokenizer_path = "../origin_model/tokenizer"

model_path = "../origin_model/chinese-roberta-wwm-ext"

new_model_path = "../model_output/roberta"

print(
    f"data path is {raw_text}, origin model path is {model_path}, and trained model saved {new_model_path}, tokenizer is {tokenizer_path}")


class LineByLineTextDataset(Dataset):
    """
    This dataset is
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        lines_1 = []
        lines_2 = []
        with open(file_path, encoding="utf-8") as f:
            for line in tqdm.tqdm(f.read().splitlines(), desc="loading data", leave=False):
                if len(line) > 0 and not line.isspace():
                    text1, text2, label = line.split("\t")
                    lines_1.append(text1)
                    lines_2.append(text2)

        self.tokenizer = tokenizer
        self.block_size = block_size

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


# using cache_result increase speed of loadding data if want to change cache use _refresh=True
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=32  # maximum sequence length
    )


tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = RobertaForMaskedLM.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))

dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text)

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
    num_train_epochs=100,
    per_device_train_batch_size=300,
    learning_rate=5e-5,
    dataloader_num_workers=16,
    # disable_tqdm=True,
    weight_decay=0.01,
    seed=42,
    fp16=True,
    save_steps=20_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    tokenizer=tokenizer
)

print("*" * 35)
print("traing...")
start_time = time.time()
trainer.train()
print("training finish and time is " + str(time.time() - start_time))

trainer.save_model(new_model_path)
