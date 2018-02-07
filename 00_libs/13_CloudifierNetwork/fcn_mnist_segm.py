# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:54:51 2018

@author: Andrei Ionut Damian
"""
from keras.datasets import mnist
from keras.models import Model
import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Input, Activation, add, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

from log_util import LoadLogger

if __name__ == '__main__':

  log = LoadLogger('MFCN','config.txt')  
  
  log.P("Loading mnist...")  
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = np.lib.pad(X_train, ((0,0),(6,6),(6,6)),'edge')
  X_test  = np.lib.pad(X_test, ((0,0),(6,6),(6,6)),'edge')
  
  H = X_train.shape[1]
  W = X_train.shape[2]
  
  log.P("Loading Y_train...")  
  Y_train = np_utils.to_categorical(y_train, 11)
  sh = X_train.shape
  y_train_dense = np.zeros((sh[0],sh[1],sh[2],11))
  y_train_dense[:,:,:] = [0,0,0,0,0,0,0,0,0,0,1]
  nonzero_positions = np.where(X_train > 0)
  y_train_dense[nonzero_positions] = Y_train[nonzero_positions[0]]
  
  
  log.P("Loading Y_test...")  
  Y_test = np_utils.to_categorical(y_test, 11)
  sh = X_test.shape
  y_test_dense = np.zeros((sh[0],sh[1],sh[2],11))
  y_test_dense[:,:,:] = [0,0,0,0,0,0,0,0,0,0,1]
  nonzero_positions = np.where(X_test > 0)
  y_test_dense[nonzero_positions] = Y_test[nonzero_positions[0]]


  X_train = np.expand_dims(X_train,axis=3)
  X_test  = np.expand_dims(X_test,axis=3)
  
  #for i in tqdm(range(X_train.shape[0])):
  #  pos_h,pos_w = np.where(X_train[i]>0)
  #  y_train_dense[i][pos_h, pos_w] = y_train[i]
  
  
  log.P("Praparing network...")
  x_input = Input((None,None,1))
  
  x = Conv2D(32, 3, padding='same', activation='relu', name='c1')(x_input)
  x = Conv2D(16, 1, activation='relu', name='c1x1_1')(x)  
  skip_depth = 64
  x = Conv2D(skip_depth, 3, strides=(2,2), padding='same', activation=None, name='downsample1')(x) 
  
  enc_1 = x
  skip_1 = enc_1
  x = Conv2D(16, 1, activation='relu', name='c1x1_2')(x)
  x = Conv2D(64, 3, padding='same', activation='relu', name='c2')(x)
  x = Conv2D(skip_depth, 1, activation='relu', name='c1x1_3')(x)
  x = add([skip_1, x]) # skip connections
  x = Conv2D(64, 3, strides=(2,2), padding='same', activation=None, name='downsample2')(x)
  enc_2 = x
  
  skip_depth = 128
  skip_2 = Conv2D(skip_depth,1)(x)
  x = Conv2D(32, 1, activation='relu', name='c1x1_4')(x)
  x = Conv2D(skip_depth, 3, padding='same', activation='relu', name='c3')(x)
  x = add([skip_2, x])

  x = Conv2D(256, 3, strides=(2,2), padding='same', activation=None, name='downsample3')(x)
  skip_depth = 256
  skip_3 = x
  x = Conv2D(32, 1, activation='relu', name='c1x1_5')(x)
  x = Conv2D(skip_depth, 3, padding='same', activation='relu', name='c4')(x)
  x = add([skip_3, x])
  
  x = Conv2D(64, 1, activation=None, name='c1x1_6')(x)
  final_conv = x
  
  #x = Dropout(0.5)
  
  prev = final_conv
  dec_1 = Conv2DTranspose(64, 4,
                          strides=(2,2),
                          padding='same',
                          use_bias=False,
                          name='deconv_1')(prev)
  fuse_1 = add([dec_1, enc_2])

  prev = fuse_1
  dec_2 = Conv2DTranspose(64, 4,
                          strides=(2,2),
                          padding='same',
                          use_bias=False,
                          name='deconv_2')(prev)
  fuse_2 = add([dec_2, enc_1])
  
  prev = fuse_2
  dec_final = Conv2DTranspose(11, 4, 
                              strides=(2,2), 
                              padding='same', 
                              use_bias=False,
                              name='deconv_preds')(prev)
  
  preds = Activation('softmax',  name='softmax_act')(dec_final)
  
  model = Model(inputs=x_input, outputs=preds)
  log.P(log.GetKerasModelSummary(model))
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['acc'])
  
  log.P("Training ...")
  model.fit(X_train,y_train_dense, batch_size=128, epochs=3,
            validation_data=(X_test,y_test_dense))
  
  
  log.P("Testing...")
  
  test_size = 112 # 112 % 2**4 must be 0
  new_image = np.zeros((test_size,test_size))
  
  nr_images = 10
  
  indices = np.arange(X_test.shape[0])
  
  orig_h, orig_w = (X_test.shape[1],X_test.shape[2])
  
  #np.random.seed(1234)
  sampled = np.random.choice(indices,nr_images, replace=False)
  log.P("Samples: {}".format(sampled))
  labels = []
  pos_w = np.random.randint(1,10)
  pos_h = np.random.randint(1,10)
  for i in range(nr_images):
    ind = sampled[i]
    x_cur = X_test[ind]
    if (pos_h<=(test_size-orig_h)):
      new_image[pos_h:(pos_h+orig_h),pos_w:(pos_w+orig_w)] = x_cur.reshape((orig_h,orig_w))
      labels.append(np.argmax(y_test[ind]))
      pos_w += 28 + np.random.randint(-5,10)
    if pos_w >= (test_size - orig_w):
      pos_w = np.random.randint(1,10)
      pos_h += 28 + np.random.randint(-3,8)
  plt.matshow(new_image,cmap="gray")
  plt.show()
  
  
  new_image = np.expand_dims(new_image, axis = 0)
  new_image = np.expand_dims(new_image, axis = 3)
  y_preds = model.predict(new_image)
  y_hat = y_preds.argmax(axis=3)
  plt.matshow(y_hat.squeeze())
  plt.show()
