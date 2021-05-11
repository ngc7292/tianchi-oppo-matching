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

from fastNLP import cache_results
from transformers import BertTokenizer, LineByLineTextDataset
from modeling_nezha import NeZhaForMaskedLM
from configuration_nezha import NeZhaConfig
from transformers import Trainer, TrainingArguments

from DataCollator import DataCollatorForLanguageModelingNgram

vocab_file = './vocab.txt' # vocab file

raw_text = '/remote-home/zyfei/project/tianchi/data/raw_text_nezha.txt' # line by line file

model_name_or_path = "/remote-home/zyfei/project/tianchi/models/nezha-large-www"
new_model_path = "/remote-home/zyfei/project/tianchi/model_output/nezha_output_2"

cache_path = "/remote-home/zyfei/project/tianchi/cache/nezha-4-15-rawtext-2"

# using cache_result increase speed of loadding data if want to change cache use _refresh=True
@cache_results(_cache_fp=cache_path, _refresh=True)
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=32  # maximum sequence length
    )


tokenizer = BertTokenizer(vocab_file=vocab_file)

config = NeZhaConfig.from_pretrained(model_name_or_path, vocab_size=tokenizer.vocab_size)

model = NeZhaForMaskedLM.from_pretrained(model_name_or_path)
model.resize_token_embeddings(tokenizer.vocab_size)

# model = NeZhaForMaskedLM.from_pretrained(model_name_or_path, config=config)


dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text, _refresh=False)


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
    num_train_epochs=200,
    per_device_train_batch_size=128,
    save_steps=10_000,
    prediction_loss_only=True,
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("*"*35)
print("traing...")
trainer.train()

trainer.save_model(new_model_path)