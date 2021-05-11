# -*- coding: utf-8 -*-
"""
__title__="test_func"
__author__="ngc7293"
__mtime__="2021/3/13"
"""


def get_most_vocab(vocab):
    """
    取出vocab中频数最大的字。
    :param vocab:{'xxx':x,'xxx':x}
    :return:
    """
    a = sorted(vocab.items(), key=lambda s: -s[1])
    return a[0][0]


if __name__ == '__main__':
    a = {"你": 1, "我": 2}
    get_most_vocab(a)
