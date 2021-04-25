# -*- coding: utf-8 -*-
"""
__title__="futher_pertrain_n_gram"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import numpy as np
import torch
import random

from fastNLP import cache_results
from transformers import BertTokenizer, LineByLineTextDataset
from  modeling_electra import ElectraForMaskedLM
from configuration_electra import ElectraConfig
from modeling_nezha import NeZhaForMaskedLM
from configuration_nezha import NeZhaConfig

from DataCollator import DataCollatorForLanguageModelingNgram

train_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_train_20210228.tsv'
test_path = '/remote-home/zyfei/project/tianchi/data/gaiic_track3_round1_testA_20210228.tsv'

vocab_file = "/remote-home/zyfei/project/tianchi/baseline_nezha_v2/electra_tokens.txt"

raw_text = '/remote-home/zyfei/project/tianchi/baseline_nezha_v2/electra_raw_text_ngram.txt'

# using cache_result increase speed of loadding data if want to change cache use _refresh=True
@cache_results(_cache_fp="/remote-home/zyfei/project/tianchi/cache/cache_electra", _refresh=True)
def load_data(data_tokenizer, raw_text_path):
    return LineByLineTextDataset(
        tokenizer=data_tokenizer,
        file_path=raw_text_path,
        block_size=32  # maximum sequence length
    )


tokenizer = BertTokenizer(vocab_file=vocab_file)

dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text, _refresh=True)


# config = NeZhaConfig.from_pretrained('/remote-home/zyfei/project/tianchi/models/nezha-large-www')

config = ElectraConfig.from_pretrained("hfl/chinese-electra-180g-large-discriminator")

model = ElectraForMaskedLM.from_pretrained('hfl/chinese-electra-180g-large-discriminator', config=config)

random_seed = 2021
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(2021)


data_collator = DataCollatorForLanguageModelingNgram(
    data_tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir='/remote-home/zyfei/project/tianchi/model_output/electra_output',
    overwrite_output_dir=True,
    num_train_epochs=300,
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

trainer.save_model("/remote-home/zyfei/project/tianchi/models/electra/")
tokenizer.save_pretrained("/remote-home/zyfei/project/tianchi/models/electra")
