#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:00:53 2020

@author: jderoo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import mean_squared_error

np.random.seed(1)

# Jacob's first LSTM


positive_path  = '../data/RS_db.csv'
negative_path   = "../data/ncrna_db_length_norm.csv"

training_fraction = .8
vectors_per_char  = 100  #i don't really know what this does
max_RS_length     = 300
n_neurons         = 100
n_epoch           = 1
batch             = 500
seq_or_kmer       = 1  # for seq: 0, for Kmer: 1

if seq_or_kmer == 0:
    RS_size = 4
elif seq_or_kmer == 1:
    RS_size = 64
    
    

# load in the data
# first column is ID
# second column is actual string
# 4th onward is sliding window # of components


def data_prep(path, training_fraction , pone):
    df1 = pd.read_csv(positive_path)
    dataset = df1.values
    IDs = dataset[:,1]
    seqs = dataset[:,2]
    data = dataset[:, 3:]

      
    # split the data, I know William will hate me for doing this so simply
    # these are positives
    
    if seq_or_kmer == 1:
        train_size = int(len(data) * training_fraction)
        test_size = len(data) - train_size
        train, test = data[0:train_size], data[train_size:len(data)]
        print(train_size)
        
    if seq_or_kmer == 0:
        train_size = int(len(seqs) * training_fraction)
        test_size = len(seqs) - train_size
        train, test = seqs[0:train_size], seqs[train_size:len(seqs)]
        print(train_size) 
        
    
    def seq_nummer(seq):
        
        '''
        
        A = 1, U = 2, C = 3, G = 4
        
        '''
        out_list = []
        for i in seq:
            if i == 'A' or i == 'a':
                out_list.append(1)
            elif i == 'U' or i == 'u':
                out_list.append(2)
            elif i == 'C' or i == 'c':
                out_list.append(3)    
            elif i == 'G' or i == 'g':
                out_list.append(4)
                
        return out_list
    
    X_train = []
    Y_train = []
    x_test = []
    y_test = []
    
    if seq_or_kmer == 0:
        for s in train:
            converted_aucg_1234 = seq_nummer(s)
            X_train.append(converted_aucg_1234)
            Y_train.append(pone)
        for s in test:
            converted_aucg_1234 = seq_nummer(s)
            x_test.append(converted_aucg_1234)
            y_test.append(pone)
    
    if seq_or_kmer == 1:
        X_train = train; x_test = test;
        
    for s in range(0,len(X_train)):
        Y_train.append(pone)
    for s in range(0,len(x_test)):
        y_test.append(pone)
    
    if seq_or_kmer == 0:
        X_train = sequence.pad_sequences(X_train, maxlen=max_RS_length)
        x_test = sequence.pad_sequences(x_test, maxlen=max_RS_length)
        
    return X_train, Y_train, x_test, y_test
    
X_train_pos, Y_train_pos, x_test_pos, y_test_pos = data_prep(positive_path, training_fraction, 1) 
X_train_neg, Y_train_neg, x_test_neg, y_test_neg = data_prep(negative_path, training_fraction, 0) 

X_train = []; Y_train = []; x_test = []; y_train = []

X_train = np.concatenate ( (X_train_pos , X_train_neg), axis = 0)
Y_train = Y_train_pos + (Y_train_neg)
x_test  = np.concatenate  ( (x_test_pos , x_test_neg), axis=0)
y_test  = y_test_pos + (y_test_neg)

maxml = 0

for i in range(0, len(X_train)):
    if (max(X_train[i])) > maxml:
        maxml = max(X_train[i])
        
max_RS_freq = maxml  # what is the highest value that occurrs in dataset; for a seq should be 4, kmers 31.


1/0

'''
X_train = sequence.pad_sequences(X_train, maxlen = max_RS_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_RS_len)
'''

# create the model

'''

The first layer of our model is an embedded layer that uses vectors_per_char vectors to represent each character/kmer.
The next layer is the LSTM layer with n_neurons neurons.
Because this is a classification problem, the final output is a Dense layer w/ sigmodial activation.

'''
   

model = Sequential()
model.add(Embedding(max_RS_freq, vectors_per_char, input_length=RS_size)) 
model.add(LSTM(n_neurons))   
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=n_epoch, batch_size=batch)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#model.save('model1.h5')

    
    



