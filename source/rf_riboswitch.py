#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:21:28 2020


for help and inspiration: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

@author: jderoo
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib




model_name = '../models/RF_model1.sav'
num_trees = 50
save_model = 0





# Generate the data

df1 = pd.read_csv('../data/RS_3mers.csv', delimiter=',')
df2 = pd.read_csv('../data/ncRNA_3mers_lengthnorm.csv', delimiter=',')



df = [df1, df2]
df = pd.concat(df)
X = df.values[:,3:]

rows, cols = df.shape

Y = []

for row in range(0,rows):
    if row <= df1.shape[0]:
        Y.append(1)
    elif row > df1.shape[0]:
        Y.append(0)
        


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2 ,random_state=42)

np.random.seed(5)
np.random.shuffle(X_train)
np.random.seed(5)
np.random.shuffle(y_train)

np.random.seed(9)
np.random.shuffle(X_test)
np.random.seed(9)
np.random.shuffle(y_test)


print('data made\n')


# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=num_trees, 
                               bootstrap = True,
                               max_features = 'sqrt')

print('model made\n')
# Fit on training data
model.fit(X_train, y_train)

print('model fit\n')
# Actual class predictions
y_pred = model.predict(X_test)
# Probabilities for each class
y_probs = model.predict_proba(X_test)[:, 1]


# Actual class predictions
y_train_preds = model.predict(X_train)
# Probabilities for each class
y_train_probs = model.predict_proba(X_train)[:, 1]


# Calculate roc auc
roc_value = roc_auc_score(y_test, y_probs)


counter = 0
for i in range(0,len(y_pred)):
    if y_pred[i] == y_test[i]:
        counter +=1
        
acc = counter/len(y_test)

print("Accuracy: %.2f%%\n" % (acc*100))




def evaluate_model(probs, test_labels):
    print(type(test_labels))
    print(type(test_labels))
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    '''
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    '''
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();
    
    
evaluate_model( y_probs, y_test)

if save_model == 1:
    joblib.dump(model, model_name) 




  