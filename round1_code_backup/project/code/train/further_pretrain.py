# -*- coding: utf-8 -*-
"""
__title__="futher_pertrain_n_gram"
__author__="ngc7293"
__mtime__="2021/3/25"
"""
import numpy as np
import torch
import argparse

from fastNLP import cache_results
from transformers import BertTokenizer, LineByLineTextDataset
from modeling_nezha import NeZhaForMaskedLM
from modeling_roberta import BertForMaskedLM
from modeling_electra import ElectraForMaskedLM

from configuration_nezha import NeZhaConfig
from configuration_roberta import BertConfig
from configuration_electra import ElectraConfig

from DataCollator import DataCollatorForLanguageModelingNgram

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input model name")

    parser.add_argument("--model_name", type=str, default="nezha")

    args = parser.parse_args()

    train_path = '../../tcdata/gaiic_track3_round1_train_20210228.tsv'
    test_path = '../../tcdata/gaiic_track3_round1_testA_20210228.tsv'

    if args.model_name == "nezha":
        vocab_file = '../../user_data/tmp_data/nezha_tokens.txt'
        raw_text = '../../user_data/tmp_data/nezha_raw_text_ngram.txt'

        model_path = "../../user_data/model_data/nezha-large-www"
        config = NeZhaConfig.from_pretrained(model_path)
        model = NeZhaForMaskedLM.from_pretrained(model_path, config=config)

        model_saved_path = "../../user_data/tmp_data/saved_model/nezha/"

    elif args.model_name == "roberta":
        vocab_file = '../../user_data/tmp_data/roberta_tokens.txt'
        raw_text = '../../user_data/tmp_data/roberta_raw_text_ngram.txt'

        model_path = "../../user_data/model_data/chinese-roberta-wwm-ext-large"
        config = BertConfig.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path, config=config)

        model_saved_path = "../../user_data/tmp_data/saved_model/roberta/"
    elif args.model_name == "electra":
        vocab_file = '../../user_data/tmp_data/electra_tokens.txt'
        raw_text = '../../user_data/tmp_data/electra_raw_text_ngram.txt'

        model_path = "../../user_data/model_data/chinese-electra-180g-large-discriminator"
        config = ElectraConfig.from_pretrained(model_path)
        model = ElectraForMaskedLM.from_pretrained(model_path, config=config)

        model_saved_path = "../../user_data/tmp_data/saved_model/electra/"
    else:
        raise NotImplementedError

    def load_data(data_tokenizer, raw_text_path):
        return LineByLineTextDataset(
            tokenizer=data_tokenizer,
            file_path=raw_text_path,
            block_size=32  # maximum sequence length
        )


    tokenizer = BertTokenizer(vocab_file=vocab_file)

    dataset = load_data(data_tokenizer=tokenizer, raw_text_path=raw_text, _refresh=False)

    random_seed = 42
    np.random.seed(random_seed)

    data_collator = DataCollatorForLanguageModelingNgram(
        data_tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=model_saved_path,
        overwrite_output_dir=True,
        num_train_epochs=100,
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

    print("*" * 35)
    print(f"traing {model_path} models...")
    trainer.train()

    trainer.save_model(model_saved_path)
    tokenizer.save_pretrained(model_saved_path)
