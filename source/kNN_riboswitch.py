#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:17:36 2020

kNN operates most efficently if values are 0-1.

@author: jderoo
"""

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

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.7,random_state=42, stratify=Y)

print('reached point 1, data built\n')

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
print('reached point 2, knn fit\n')
acc = knn.score(X_test,y_test)
print('reached point 3, knn scored: %d\n'%acc*100)

y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
print('reached point 4, confusion matrix prepared')

y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors = 2) ROC curve')
plt.show()


roc_score = roc_auc_score(y_test,y_pred_proba)
print('the roc_score is: %d\n'%roc_score)


