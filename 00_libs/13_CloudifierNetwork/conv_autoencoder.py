# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:54:51 2018

@author: Andrei Ionut Damian
"""
from keras.datasets import mnist
from keras.models import Model
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Y_train = np_utils.to_categorical(y_train, 10)

y_train_dense = np.zeros(X_train.shape) - 1
nonzero_positions = np.where(X_train > 0)
y_train_dense[nonzero_positions] = y_train[nonzero_positions[0]]

#for i in tqdm(range(X_train.shape[0])):
#  pos_h,pos_w = np.where(X_train[i]>0)
#  y_train_dense[i][pos_h, pos_w] = y_train[i]

X_train = np.expand_dims(X_train,axis=3)
y_train_dense = np.expand_dims(y_train_dense,axis=3)

x_input = Input((28,28,1))

x = Conv2D(32, 3, padding='same')(x_input)
x = Conv2D(48, 3, strides=(2,2), padding='same')(x)
x = Conv2D(64, 3, padding='same')(x)
encoded = Conv2D(64, 3, strides=(2,2), padding='same')(x)

decoded1 = Conv2DTranspose(64, 3, strides=(2,2), padding='same')(encoded)
decoded2 = Conv2DTranspose(1, 3, strides=(2,2), padding='same')(decoded1)
decoded3 = Conv2DTranspose(1, 3, strides=(4,4), padding='same')(encoded)

model1 = Model(inputs=x_input, outputs=decoded2)
model2 = Model(inputs=x_input, outputs=decoded3)
model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')
model1.fit(y_train_dense,y_train_dense, batch_size=128, epochs=2)
model2.fit(y_train_dense,y_train_dense, batch_size=128, epochs=2)

