# -*- coding: utf-8 -*-
"""
__title__="data_process"
__author__="ngc7293"
__mtime__="2021/4/9"
"""
import os
import argparse

# 不删除低频词
def load_data_pair_sent(path, result):
    if path.split(".")[-1] == "tsv":
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                rows = line.split('\t')[0:2]
                a = []
                for key in rows[0].split(' '):
                    key = key.strip()
                    a.append(key)
                b = []
                for key in rows[1].split(' '):
                    key = key.strip()
                    b.append(key)
                result.append(' '.join(a) + ' [SEP] ' + ' '.join(b))
                result.append(' '.join(b) + ' [SEP] ' + ' '.join(a))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input model name")

    parser.add_argument("--model_name", type=str, default="nezha")

    args = parser.parse_args()

    train_path = '../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
    test_path = '../tcdata/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'

    if args.model_name == "nezha":
        origin_vocab_file = '../user_data/model_data/nezha-large-www/vocab.txt'

        vocab_file = '../user_data/tmp_data/tokens/nezha_tokens.txt'

        raw_text = '../user_data/tmp_data/raw_data/nezha_raw_text_ngram.txt'
    elif args.model_name == "roberta":
        origin_vocab_file = '../user_data/model_data/chinese-roberta-wwm-ext-large/vocab.txt'

        vocab_file = '../user_data/tmp_data/tokens/roberta_tokens.txt'

        raw_text = '../user_data/tmp_data/raw_data/roberta_raw_text_ngram.txt'
    elif args.model_name == "electra":
        origin_vocab_file = '../user_data/model_data/chinese-electra-180g-large-discriminator/vocab.txt'

        vocab_file = '../user_data/tmp_data/tokens/electra_tokens.txt'

        raw_text = '../user_data/tmp_data/raw_data/electra_raw_text_ngram.txt'
    else:
        raise NotImplementedError

    # 统计词频
    vocab_frequence = {}

    with open(train_path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            rows = line.split('\t')[0:2]
            for sent in rows:
                for key in sent.split(' '):
                    key = key.strip()
                    vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

    with open(test_path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            rows = line.split('\t')[0:2]
            for sent in rows:
                for key in sent.split(' '):
                    key = key.strip()
                    vocab_frequence[key] = vocab_frequence.get(key, 0) + 1

    vocab_frequence = sorted(vocab_frequence.items(), key=lambda s: -s[1])

    nezha_orgin_vocab = []
    with open(origin_vocab_file, encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if line != '!':
                nezha_orgin_vocab.append(line)
            else:
                break

    vocab = nezha_orgin_vocab + [key[0] for key in vocab_frequence]

    train_result = []
    test_result = []
    load_data_pair_sent(train_path, train_result)
    load_data_pair_sent(test_path, test_result)

    all_result = train_result + test_result
    with open(raw_text, 'w') as f:
        for key in all_result:
            f.write(str(key) + '\n')

    with open(vocab_file, 'w') as f:
        for key in vocab:
            f.write(str(key) + '\n')
