# -*- coding: utf-8 -*-
"""
__title__="futher_pertrain_n_gram"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import torch
import transformers
import numpy as np

from fastNLP import cache_results
from transformers import BertTokenizer, LineByLineTextDataset

from modeling_nezha import NeZhaForMaskedLM
from configuration_nezha import NeZhaConfig
from DataCollator import DataCollatorForLanguageModelingNgram

# transformers.logging.set_verbosity_error()

train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

vocab_file = './tokens.txt'

raw_text = './raw_text_ngram.txt'

# using cache_result increase speed of loadding data if want to change cache use _refresh=True
@cache_results(_cache_fp="./cache", _refresh=True)
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=32  # maximum sequence length
    )


tokenizer = BertTokenizer(vocab_file=vocab_file)

print("loading data...")
dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text, _refresh=False)

print("*"*35)
config = NeZhaConfig.from_pretrained('/remote-home/zyfei/project/tianchi/models/nezha-large-www')

model = NeZhaForMaskedLM.from_pretrained('/remote-home/zyfei/project/tianchi/models/nezha-large-www', config=config)

random_seed = 42
np.random.seed(random_seed)

data_collator = DataCollatorForLanguageModelingNgram(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./output/',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=128,
    save_steps=5_000,
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

trainer.args.learning_rate = 1e-5
trainer.train()

trainer.save_model("./models/")
tokenizer.save_pretrained("./models/")
