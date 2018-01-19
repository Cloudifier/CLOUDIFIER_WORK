# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:11:41 2017

@author: Andrei
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:01:48 2017

@author: Andrei
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np


import tensorflow as tf


def softmax(z):
  z = z.copy()
  z -= z.max(axis = 1, keepdims = True)
  return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims = True)


def accuracy(y_hat, y_test):
  pred = np.argmax(y_hat, axis = 1) == np.argmax(y_test, axis = 1)
  return (np.sum(pred) / y_hat.shape[0]) * 100
  
  

mnist = fetch_mldata('mnist-original')
X = mnist.data
y = mnist.target
n_classes = np.unique(y).shape[0]
Y = np.zeros((y.shape[0], n_classes))
for i in range(y.shape[0]):
    Y[i, int(y[i])] = 1

m, n = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state = 1234)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

m_train, n_features = X_train.shape

batch_size = 512

optimizer = tf.keras.optimizers.SGD(lr = 0.001)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation = 'softmax', input_shape = (n_features,),
                                kernel_regularizer = tf.keras.regularizers.l2(0.05),
                                kernel_initializer = 'zeros'))
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy')
model.fit(X_train, y_train, epochs = 3, batch_size = batch_size, validation_split = 0.1)

yhat = model.predict(X_test)
print("Test accuracy: {:.2f}%".format(accuracy(yhat,y_test)))

