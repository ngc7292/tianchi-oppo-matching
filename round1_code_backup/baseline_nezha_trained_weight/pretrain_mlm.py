# -*- coding: utf-8 -*-
"""
__title__="pretrain_mlm"
__author__="ngc7293"
__mtime__="2021/3/16"
"""
import torch
import transformers

from transformers import LineByLineTextDataset, BertTokenizer
from transformers import BertConfig, BertForMaskedLM
from modeling_nezha import NeZhaForMaskedLM
from configuration_nezha import NeZhaConfig

from DataCollator import DataCollatorForLanguageModelingWithNgram


transformers.set_seed(42)

train_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"
test_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv"

vocab_data_path = "/remote-home/zyfei/project/tianchi/baseline/vocab.txt"

raw_text = "/remote-home/zyfei/project/tianchi/baseline/raw_text.txt"

nezha_model_path = "/remote-home/zyfei/project/tianchi/baseline_nezha_trained_weight/nezha-base-www"

print("create tokenizer...")
tokenizer = BertTokenizer(vocab_file=vocab_data_path)

print("load data...")
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=raw_text,
    block_size=128  # maximum sequence length
)

print("create model...")


config = NeZhaConfig.from_pretrained(nezha_model_path)

model = NeZhaForMaskedLM.from_pretrained(nezha_model_path)

print("model's parameters number is:")
print(model.num_parameters())

data_collator = DataCollatorForLanguageModelingWithNgram(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, n_gram=1
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./mybert/',
    overwrite_output_dir=True,
    num_train_epochs=500,
    do_train=True,
    per_device_train_batch_size=128,
    learning_rate=5e-5,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./mybert")
