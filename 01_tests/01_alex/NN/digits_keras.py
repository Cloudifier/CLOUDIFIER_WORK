# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:31:45 2017

@author: Andrei
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np

def meanNormalization(X):
    mean = X.mean()
    _max = X.max()
    _min = X.min()
    norm = (X - mean) / (_max -_min)
    return norm, mean, _max, _min

def meanNormalizationWithParams(X, mean, _max, _min):
    norm = (X - mean) / (_max - _min)
    return norm

if __name__=='__main__':
  digits = load_digits()
  n_samples = len(digits.images)
  X = digits.images.reshape((n_samples, -1))
  y = digits.target.reshape(n_samples, 1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  
  X_train, mean, _max, _min = meanNormalization(X_train)
  X_test = meanNormalizationWithParams(X_test, mean, _max, _min)
  m_train, n_train = X_train.shape

  Y_train = np.zeros((len(y_train), len(digits.target_names)))
  
  for i in range(len(y_train)):
      Y_train[i, y_train[i]] = 1  
      
  model = Sequential()
  model.add(Dense(32, input_shape = (64,), activation = "relu"))
  model.add(Dense(16, activation = "relu"))
  model.add(Dense(10, activation = "softmax"))
  opt = SGD(lr = 0.005)
  model.compile(optimizer = opt, loss = "categorical_crossentropy")
  model.fit(X_train, Y_train, batch_size = 8, epochs = 10)
  
  yhat = model.predict(X_test)

  predict = (np.argmax(yhat, axis=1)).reshape(m, 1)
  print(np.mean((predict == y_test) * 1) * 100)  
  
  yhat_train = model.predict(X_train)
  print("Train acc: {:.2f}%".format(
      100*((np.argmax(yhat_train, axis=1)==y_train.ravel()).sum()/y_train.shape[0])))