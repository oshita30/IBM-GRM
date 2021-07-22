# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:49:52 2021

@author: my
"""

import pickle
from sys import exit

test_data = pickle.load(open('data/lmg/test.pkl', 'rb'))
l = []

l1 = []
l2 = []

l.append(l1)
l.append(l2)
i = -1

for d in test_data:
    di = 0
    i = i + 1
    for item in d:        
        di = di+1
        if (di > 100):
            break
        l[i].append(item)
        
file = open('data/lmg/smalltest.pkl', 'wb')
pickle.dump(l, file)
file.close()
