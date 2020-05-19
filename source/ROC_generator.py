#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:47:19 2020

@author: jderoo
"""

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

from keras.models import model_from_json
from scipy.special import softmax

from cycler import cycler
seed = 9



df = pd.read_csv('../data/test_data_set_3mer_CNN.csv')
X_test = df.values[:,4:68]
y_test = df.values[:,68]



def evaluate_model(probs, test_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    if type(probs) != list:
        probs = list(probs)

    if type(test_labels) != list:
        test_labels = list(test_labels)
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
    return model_fpr, model_tpr

    '''
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();
    '''

def k_modeller(json_str,h5_str):
    json_file = open(json_str, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(h5_str)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(X_test, y_test, verbose=0)
    print("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    y_probs = model.predict_proba(X_test)[:,0]
    return  list(y_probs), model


# load lstm model
lstm_probs = np.loadtxt('../data/lstm_test_data_scores.txt')    
lstm_tpr, lstm_fpr = evaluate_model(lstm_probs, y_test)


# load ffNN model
json_ffnn = '../models/ffNN_model2_3mer.json'
h5_ffnn   = '../models/ffNN_model2_5mer.h5'
ffnn_prob, ffnn = k_modeller(json_ffnn,h5_ffnn)
ffnn_tpr, ffnn_fpr = evaluate_model(ffnn_prob, y_test)


# load CNN probability list
UTR_hits_CNN = pd.read_csv('../data/test_data_set_3mer_CNN.csv')
prs = softmax(np.array(UTR_hits_CNN.iloc[:,-2:]),axis=1)[:,0]
cnn_tpr, cnn_fpr = evaluate_model( prs, y_test)


# load RF 
rf = joblib.load('../models/RF_model1_3mer.sav')
rf_proba = rf.predict_proba(list(X_test))[:,1]
rf_tpr, rf_fpr = evaluate_model(list(rf_proba), y_test)


# load kNN
plt.figure(figsize = (8, 6))
plt.rcParams['font.size'] = 16
knn_tpr = np.loadtxt('knn_tpr.txt')
knn_fpr = np.loadtxt('knn_fpr.txt')




# plot

colors = ['#5e1150', '#d90000', '#16ba1c','#ff8c00','#1039c2']

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rcParams.update({'font.size': 22, 'font.weight':'bold' }   )
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
plt.rcParams.update({'lines.linewidth': 2})

plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})
plt.rcParams.update({'axes.prop_cycle':cycler(color=colors)})


plt.rcParams.update({'xtick.major.width'   : 2.8 })
plt.rcParams.update({'xtick.labelsize'   : 14 })



plt.rcParams.update({'ytick.major.width'   : 2.8 })
plt.rcParams.update({'ytick.labelsize'   : 14})


plt.rcParams.update({'axes.linewidth':2.8})
plt.rcParams.update({'axes.labelpad':8})
plt.rcParams.update({'axes.titlepad':10})
plt.rcParams.update({'figure.dpi':300})

plt.plot([0,1],[0,1],'k--')
plt.plot(lstm_tpr,lstm_fpr, label='RNN-LSTM')
plt.plot(cnn_tpr,cnn_fpr, label='CNN')
plt.plot(knn_fpr,knn_tpr, label='KNN')
plt.plot(rf_tpr,rf_fpr, label='RF')
plt.plot(ffnn_tpr,ffnn_fpr, label='FFNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Classifer ROC Curves')
plt.legend(loc = 'lower right')
plt.show()

