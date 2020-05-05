#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:33:27 2020

@author: jderoo
"""


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

positive_path  = '../data/RS_3mers.csv'
negative_path  = "../data/ncRNA_3mers_lengthnorm.csv"
testing_frac  = .70

df1 = pd.read_csv(positive_path)
df2 = pd.read_csv(negative_path)

df = [df1, df2]
df = pd.concat(df)

rows, cols = df.shape

Y = []

for row in range(0,rows):
    if row <= df1.shape[0]:
        Y.append(1)
    elif row > df1.shape[0]:
        Y.append(0)
        
X = df.values[:,3:]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=testing_frac,random_state=42, stratify=Y)

print('reached point 1\n')


loaded_model = pickle.load(open('../models/knn_model.p', 'rb'))
print('reach point 2\n')
result = loaded_model.predict(X_test[0])
print('reach point 3\n')