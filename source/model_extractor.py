#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:33:27 2020

@author: jderoo
"""


import pickle
import time
import numpy as np
import pandas as pd
import csv



path  = '../data/5primeUTR_human_kmers_fulllen.csv'


df    = pd.read_csv(path)
 
IDs = df.values[:,1]       
X = df.values[:,3:]
rows, cols = X.shape

print('reached point 1\n')


loaded_model = pickle.load(open('../models/knn_model.p', 'rb'))

st = time.time()
positives = []
negatives = []
split5050 = []

print('reach point 2\n')


for i in range(0,rows):
    result = loaded_model.predict_proba(X[i].reshape(1,-1))
    result = result[0][0]
    if result == 1:
        positives.append(IDs[i])
    elif result == 0:
        negatives.append(IDs[i])
    elif result == 0.5:
        split5050.append(IDs[i])

    
'''

result output: 1x2. [1,0] is positive
result output: 1x2. [0,1] is negative
result output: 1x2. [0.5,0.5] is unknown


'''    

with open('kNN_negatives.csv','w',newline='') as myfile:
    wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
    wr.writerow(negatives)
    
with open('kNN_positives.csv','w',newline='') as myfile:
    wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
    wr.writerow(positives)

with open('kNN_undecided.csv','w',newline='') as myfile:
    wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
    wr.writerow(split5050)    
    
    
    