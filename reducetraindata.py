# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:05:23 2021

@author: vvtes
"""


import pickle
from sys import exit

train_data = pickle.load(open('data/lmg/train.pkl', 'rb'))
l = []

l1 = []
l2 = []

l.append(l1)
l.append(l2)
i = -1

for d in train_data:
    di = 0
    i = i + 1
    for item in d:        
        di = di+1
        if (di > 100):
            break
        l[i].append(item)
        
file = open('data/lmg/smalltrain.pkl', 'wb')
pickle.dump(l, file)
file.close()
