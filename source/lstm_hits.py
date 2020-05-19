# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:52:35 2020

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow import keras
Sequential =  keras.models.Sequential  
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM 
Embedding =  keras.layers.Embedding 
sequence =  keras.preprocessing.sequence
from sklearn.metrics import mean_squared_error

#np.random.seed(1)

# Jacob's first LSTM




df = pd.read_csv('test_data_set_3mer_CNN.csv')
df = df.drop(['Unnamed: 0.1'],axis=1)
df2 = pd.read_csv('5primeUTR_human_5mers_fulllen.csv.csv')

training_fraction = .8
vectors_per_char  = 32   # i don't really know what this does, "inteligence of each word" maybe?
n_neurons         = 100
n_epoch           = 5  # this value can be too high; for a CNN solving this problem 200 epochs was appropriate.
batch             = 500  # see above: too high is not true.
seq_or_kmer       = 1    # for seq: 0, for Kmer: 1
save_model        = 1    # 0: do not save the model (.h5)

if seq_or_kmer == 0:
    max_RS_length = 300
elif seq_or_kmer == 1:
    max_RS_length = 300
    
    
model = Sequential()
model.add(Embedding(1024, vectors_per_char, input_length=max_RS_length)) 
model.add(LSTM(n_neurons))   
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.load_weights('lstm_model2_5mer.h5')


dataset = df.values
IDs  = dataset[:,1]
seqs = dataset[:,2]
data = dataset[:,3:]


kmer_list = list((df2.columns[3:]))

def convert_to_kmer_ids(seq, kmer_list, maxlen=300):
    inds = [0,4,16,64,4**4,4**5]
    n = inds.index(len(kmer_list))

    kmer_inds = np.zeros(len(seq) - n)

    for i in range(0,len(seq)-n):   
        kmer_inds[i] = int(kmer_list.index(seq[i:i+n])) 
    kmer_inds = sequence.pad_sequences([kmer_inds], maxlen=maxlen)
    return kmer_inds 



scores = np.zeros(len(seqs))

for i in range(len(seqs)):
    if i%1000 == 0:
        print('tested:')
        print(i)
    seq = seqs[i]
    
    if len(seq) > 300:
        nwindows = int(np.ceil(len(seq)/300))
        maxscore = -1
        for j in range(nwindows-1):
            sub_seq = seq[j*300:j*300+300]
            
            kid = convert_to_kmer_ids(sub_seq,kmer_list)
            score = model.predict(kid)[0][0]    
            if score > maxscore :
                maxscore = score
                
        sub_seq = seq[-300:]
        kid = convert_to_kmer_ids(sub_seq,kmer_list)
        score = model.predict(kid)[0][0]    
        if score > maxscore :
            maxscore = score      
            
        scores[i] = maxscore
    elif len(seq) < 10:
        scores[i] = 0
        
    else:
        kid = convert_to_kmer_ids(seq,kmer_list)
        scores[i] = model.predict(kid)[0][0]
    



