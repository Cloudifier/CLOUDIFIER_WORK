# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:08:06 2018

@author: damia
"""


from keras.layers import LSTM, Dense
from keras.models import Sequential
import numpy as np

with open('text.txt','rt') as f:
    lines = f.readlines()


full_text =" ".join(lines)
vocab = list(set(full_text))
vsize =len(vocab)

def get_one_hot(letter, vocab):    
    np_vocab = np.eye(len(vocab))
    i = vocab.index(letter)
    return np_vocab[i]

def get_letter(ohv, vocab):
    i = np.argmax(ohv)
    return vocab[i]
    

data_list=[]
for t in full_text:
    data_list.append( get_one_hot(t, vocab))

data_list.append(data_list[0])
y_data = data_list[1:]
    

model = Sequential()

model.add(LSTM(128, input_shape=(vsize,)))
model.add(Dense(vsize, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x=data_list, y=y_data, batch_size=32, epochs=5)
