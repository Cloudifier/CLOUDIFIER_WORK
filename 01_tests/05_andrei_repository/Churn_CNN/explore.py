# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:04:43 2017

@author: AndreiS
"""

import pandas as pd
import numpy as np

df = pd.read_csv("D:\Google Drive\\_hyperloop_data\churn_v3\\_dbcache\EXEC_SP_GET_CHURN_@TRAN_PER_ID_=_2_@CHURN_PER_ID_=_3_@SGM_ID=22.csv")

df.head()
df.describe()

df.fillna(0, inplace= True)

cols = list(df.columns)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist

batch_size = 128
epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(filters= 32, kernel_size= (3, 3), padding="same", activation= "relu", input_shape= (28, 28, 1)))























