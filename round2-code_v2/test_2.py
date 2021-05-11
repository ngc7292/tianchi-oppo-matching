# -*- coding: utf-8 -*-
"""
__title__="test_2"
__author__="ngc7293"
__mtime__="2021/5/6"
"""
class c1():
    def __init__(self):
        self.data = [1,2,3]

    def __getitem__(self, item):
        return self.data[item]

a = c1()
b = [a, a, a]
print([j for i in b for j in i])

print([i for i in a])
for i in a:
    print(i)