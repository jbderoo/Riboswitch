#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:49:35 2020

@author: jderoo

Hugely helpful reference for values and model building:
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

Data prep was unique to our problem and solved internally.

"""



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from datetime import datetime
from keras.models import model_from_json


# Jacob's first LSTM


positive_path  = '../data/RS_5mers.csv'
negative_path   = "../data/ncRNA_5mers_lengthnorm.csv"

training_fraction = .80  # what % of the data should be training
vectors_per_char  = 32   # i don't really know what this does, "inteligence of each word" maybe?
n_neurons         = 100  # number of neurons in 2nd layer of model
n_epoch           = 2    # this value can be too high; for a CNN solving this problem 200 epochs was appropriate.
batch             = 500  # see above: too high is not true.
save_model        = 1    # 0: do not save the model (.h5)
max_RS_length     = 300  # how long is each RNA sequence forced to be. Padded proximally with 0's.

    
    

# load in the data
# first column is ID
# second column is actual string
# 4th onward is sliding window # of components


def data_prep(path, training_fraction , pone):
    '''
    path: the path to the file in use, typically .csv
    training fraction: percentage split into training
    pone: positive or negative: yes it's a riboswitch or 0 it isn't
    
    '''
    
    df1 = pd.read_csv(path)
    dataset = df1.values
    IDs  = dataset[:,1]  # unique URS code assigned to each string
    seqs = dataset[:,2]  # the actual raw sequence itself
    data = dataset[:,3:] # the frequency in that string that Kmer appears
    kmer_list = list((df1.columns[3:]))
    n = len(df1.columns[3]) # length of each Kmer (i.e. 3mer n =3, 4mer n = 4)

      
    # split the data, I know William will hate me for doing this so simply
    # these are positives
        

    train_size = int(len(seqs) * training_fraction)
    test_size  = len(seqs) - train_size
    train, test = seqs[0:train_size], seqs[train_size:len(seqs)]
    print(train_size) 
        
    
    def seq_nummer(seq):
        
        Kmer_str = [] 
        for i in range(0,len(seq)-n):   
            Kmer_str.append( int(kmer_list.index(seq[i:i+n])) )

                
        return Kmer_str
    
    X_train = []
    Y_train = []
    x_test  = []
    y_test  = []
    

    for s in train:
        converted_aucg_1234 = seq_nummer(s)
        X_train.append(converted_aucg_1234)
        Y_train.append(pone)
    for s in test:
        converted_aucg_1234 = seq_nummer(s)
        x_test.append(converted_aucg_1234)
        y_test.append(pone)
    
    
    X_train = sequence.pad_sequences(X_train, maxlen=max_RS_length)
    x_test  = sequence.pad_sequences(x_test,  maxlen=max_RS_length)
       
        
    return X_train, Y_train, x_test, y_test


X_train_pos, Y_train_pos, x_test_pos, y_test_pos = data_prep(positive_path, training_fraction, 1) 
X_train_neg, Y_train_neg, x_test_neg, y_test_neg = data_prep(negative_path, training_fraction, 0) 


# catenate positives and negatives into singular arrays

X_train = np.concatenate  ( (X_train_pos , X_train_neg), axis = 0)
Y_train = np.concatenate  ( (Y_train_pos , Y_train_neg), axis = 0)
x_test  = np.concatenate  ( (x_test_pos  , x_test_neg),  axis = 0)
y_test  = np.concatenate  ( (y_test_pos  , y_test_neg),  axis = 0)

       
maxes = [X_train.max(), Y_train.max(), x_test.max(), y_test.max()]
maxml = max(maxes)        
        
max_RS_freq = maxml+1  # what is the highest value that occurrs in dataset; for a seq should be 4, 3mers 33.
                       # this is exclusive, add 1 




# shuffle data identically.

np.random.seed(5)
np.random.shuffle(X_train)
np.random.seed(5)
np.random.shuffle(Y_train)

np.random.seed(9)
np.random.shuffle(x_test)
np.random.seed(9)
np.random.shuffle(y_test)


# create the model
'''

The first layer of our model is an embedded layer that uses vectors_per_char vectors to represent each character/kmer.
The next layer is the LSTM layer with n_neurons neurons.
Because this is a classification problem, the final output is a Dense layer w/ sigmodial activation.
This is copy-pasted from Reference 1.

'''
   

model = Sequential()
model.add(Embedding(max_RS_freq, vectors_per_char, input_length=max_RS_length)) 
model.add(LSTM(n_neurons))   
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=n_epoch, batch_size=batch)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

dt = datetime.now()
if save_model == 1:
    model_json = model.to_json()
    with open("../data/model3_5mer.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('../data/model3_5mer.h5')
    #model.save(r'model_5mer_{dt.strftime("YmdHMs")}.h5')

    