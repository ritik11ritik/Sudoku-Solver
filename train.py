#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:14:37 2021

@author: rg
"""

import numpy as np
import pandas as pd
import warnings
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

Y = df.iloc[:,0].values
X = df.iloc[:,1:].values

XX = []
for xseq in X:
   tmp = np.asarray(xseq).reshape(28,28)
   tmp.astype("int")/255.0
   XX.append(tmp.astype('int64'))

Y = to_categorical(Y, dtype='int64')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(XX, Y, test_size=0.2)
sample_shape = X_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

X_train = np.array(X_train).reshape(len(X_train),input_shape[0], input_shape[1], input_shape[2])
X_test = np.array(X_test).reshape(len(X_test),input_shape[0], input_shape[1], input_shape[2])

model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(32, 4, 1, input_shape = (28, 28, 1), activation = 'relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 4, 1, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,2), padding='same'))
model.add(Dropout(0.2))

model.add(Convolution2D(32, 4, 1, activation = 'relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (4,2), padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(10, activation = 'softmax'))

model.summary()

opt = Adam(lr=0.00001)

# Compiling the CNN
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



hist = model.fit(np.array(X_train), np.array(Y_train),
                 batch_size=8, 
                 epochs = 1000,
                 validation_data = (np.array(X_test), np.array(Y_test)))

model.evaluate(X_test, Y_test)

model.save("model.h5")
