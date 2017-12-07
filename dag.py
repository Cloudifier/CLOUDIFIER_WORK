# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:12:08 2017

@author: AndreiS

"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
import platform
from importlib.machinery import SourceFileLoader
import pandas as pd

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  
def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val
  
def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "Google Drive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

def fetch_data():
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "base.py")).load_module()
  logger = logger_lib.Logger(lib='CNN')

  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)

  X = mnist.data
  y = mnist.target

  X = min_max_scaler(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)
  X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                test_size=0.5,
                                                                random_state=42)

  return dotdict({'train': (X_train, y_train),
                  'test': (X_test, y_test),
                  'validation': (X_validation, y_validation)}), logger

import keras
from keras.layers import Dense, Conv2D, Input, GlobalMaxPool2D, MaxPooling2D, concatenate, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_ = Input((None, None, 1))
input_x1 = input_

x1 = Conv2D(32, 3, padding= "same", activation='relu')(input_x1)
x1 = Conv2D(32, 3, padding= "same", activation='relu')(x1)
x1 = Conv2D(64, 3, padding= "same", activation='relu')(x1)
x1 = Conv2D(64, 3, padding= "same", activation='relu')(x1)
x1 = Conv2D(128, 3, padding= "same", activation='relu')(x1)
x1 = Conv2D(128, 3, padding= "same", activation='relu')(x1)
x1 = GlobalMaxPool2D()(x1)

input_x2 = MaxPooling2D()(input_)

x2 = Conv2D(32, 3, padding= "same", activation='relu')(input_x2)
x2 = Conv2D(32, 3, padding= "same", activation='relu')(x2)
x2 = Conv2D(64, 3, padding= "same", activation='relu')(x2)
x2 = Conv2D(64, 3, padding= "same", activation='relu')(x2)
x2 = Conv2D(128, 3, padding= "same", activation='relu')(x2)
x2 = Conv2D(128, 3, padding= "same", activation='relu')(x2)
x2 = GlobalMaxPool2D()(x2)

input_x3 = UpSampling2D()(input_)

x3 = Conv2D(32, 3, padding= "same", activation='relu')(input_x3)
x3 = Conv2D(32, 3, padding= "same", activation='relu')(x3)
x3 = Conv2D(64, 3, padding= "same", activation='relu')(x3)
x3 = Conv2D(64, 3, padding= "same", activation='relu')(x3)
x3 = Conv2D(128, 3, padding= "same", activation='relu')(x3)
x3 = Conv2D(128, 3, padding= "same", activation='relu')(x3)
x3 = GlobalMaxPool2D()(x3)

concat_layer =  concatenate(inputs= [x1, x2, x3])
dense1 = Dense(1024, activation='relu')(concat_layer)
read_out = Dense(10, activation='softmax')(dense1)

model = Model(inputs = [input_], outputs = [read_out])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model.fit(X_train, y_train,
          batch_size=128,
          epochs=5,
          verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

_, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
data = fetch_data()

test50 = np.load(os.path.join(mnist_path, 'test_50x50.npy'))

test50 = test50.reshape(test50.shape[0], 50, 50, 1)
test50 = test50.astype("float32")
test50 /= 255

y = np_utils.to_categorical(data[0].test[1], 10)

score = model.evaluate(test50, y, verbose = 0)
print(score)









