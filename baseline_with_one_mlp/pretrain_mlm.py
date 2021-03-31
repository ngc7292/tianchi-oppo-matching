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

from DataCollator import DataCollatorForLanguageModelingWithNgram


transformers.set_seed(42)

train_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv"
test_data_path = "/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv"

vocab_data_path = "./vocab.txt"

raw_text = './raw_text.txt'

# train_mlm_data = []
# vocab = set()
# with open(train_data_path, encoding="utf-8") as f:
#     for line in f.readlines():
#         rows = line.split('\t')
#         for sent in rows[0:2]:
#             vocab.update(sent.split(' '))
#         train_mlm_data.append(rows[0] + ' [SEP] ' + rows[1])
#         train_mlm_data.append(rows[1] + ' [SEP] ' + rows[0])
#
# test_mlm_data = []
# with open(test_data_path, encoding="utf-8") as f:
#     for line in f.readlines():
#         rows = line.split('\t')
#         for sent in rows[0:2]:
#             vocab.update(sent.split(' '))
#         test_mlm_data.append(rows[0] + ' [SEP] ' + rows[1])
#         test_mlm_data.append(rows[1] + ' [SEP] ' + rows[0])
#
# all_mlm_data = train_mlm_data + test_mlm_data
# with open(raw_text, 'w') as f:
#     for key in all_mlm_data:
#         f.write(str(key) + '\n')
#
# vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(vocab)
#
# with open(vocab_data_path, "w") as fp:
#     for key in vocab:
#         fp.write(str(key) + "\n")


print("create tokenizer...")
tokenizer = BertTokenizer(vocab_file=vocab_data_path)

print("load data...")
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=raw_text,
    block_size=128  # maximum sequence length
)

# config = BertConfig(
#     vocab_size=tokenizer.vocab_size+1000,
#     hidden_size=768,
#     num_hidden_layers=6,
#     num_attention_heads=12,
#     max_position_embeddings=512
# )

print("create model...")
# pretrained_model_name_or_path = "uer/chinese_roberta_L-12_H-512"

# uer_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)

config = BertConfig(
    vocab_size=tokenizer.vocab_size + 1000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512
)

model = BertForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModelingWithNgram(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, n_gram=3
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./mybert/',
    overwrite_output_dir=True,
    num_train_epochs=500,
    per_device_train_batch_size=128,
    learning_rate=5e-5,
    save_steps=10_000,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./mybert")
