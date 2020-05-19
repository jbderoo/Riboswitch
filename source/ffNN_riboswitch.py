#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:46:52 2020

for help and inspiration: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

@author: jderoo
"""

# first neural network with keras make predictions
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


n_epochs   = 20
batch      = 500
model_save = 1

# load the dataset

df1 = pd.read_csv('../data/RS_3mers.csv', delimiter=',')
df2 = pd.read_csv('../data/ncRNA_3mers_lengthnorm.csv', delimiter=',')



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

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2 ,random_state=42)

np.random.seed(5)
np.random.shuffle(X_train)
np.random.seed(5)
np.random.shuffle(y_train)

np.random.seed(9)
np.random.shuffle(X_test)
np.random.seed(9)
np.random.shuffle(y_test)



# rescale so that all values exist 0 --> 1

X_train = X_train/max([X_train.max(), X_test.max()])
X_test = X_test/max([X_train.max(), X_test.max()])



# define the keras model
model = Sequential()
model.add(Dense(100, input_dim=64, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=n_epochs, batch_size=batch, verbose=1)
# make class predictions with the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
if model_save == 1:
    model_json = model.to_json()
    with open("../models/ffNN_model2_3mer.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('../models/ffNN_model2_5mer.h5')