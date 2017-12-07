# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:00:40 2017

@author: Andrei
"""

import tensorflow as tf
import os
import numpy as np

if __name__=="__main__":
  
  smodel1 = 'model1_fc'
  smodel1_fn = smodel1 + '.h5'
  smodel2 = 'model2_cnn'
  smodel2_fn = smodel2 + '.h5'
  
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  X_train = (x_train / 255.).reshape((-1,28,28,1))
  X_test = (x_test / 255.).reshape((-1,28,28,1))
  nb_classes = 10
  
  Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
  Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

  X_test_small = X_test[:5000]
  y_test_small = y_test[:5000]
  Y_test_small = Y_test[:5000]
  
  print("Pass 1: loading/training-testing phase")
  if os.path.isfile(smodel1_fn):
    print(" Loading model1", flush = True)
    model1 = tf.keras.models.load_model(smodel1_fn)
  else:
    print(" Training [{}]".format(smodel1))
    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
    model1.add(tf.keras.layers.Flatten())
    model1.add(tf.keras.layers.Dense(512, activation = "relu"))
    model1.add(tf.keras.layers.Dense(256, activation = "relu"))
    model1.add(tf.keras.layers.Dense(10, activation = "softmax"))
    model1.compile(optimizer = "adam", loss = "categorical_crossentropy",
                   metrics = ['accuracy'])
    model1.fit(X_train, Y_train, epochs = 3, batch_size = 128,)
    score = model1.evaluate(X_test, Y_test)
    print("\n Model [{}] {}: {}".format(smodel1, model1.metrics_names, score), flush = True)
    model1.save(smodel1_fn)
    
  
  
  if os.path.isfile(smodel2_fn):
    print(" Loading model2", flush = True)
    model2 = tf.keras.models.load_model(smodel2_fn)
  else:
    print(" Training [{}]".format(smodel2))
    X_inp = tf.keras.layers.Input((28,28,1))

    X = tf.keras.layers.Conv2D(32, (3,3))(X_inp)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.MaxPool2D((2,2))(X)

    X = tf.keras.layers.Conv2D(64, (3,3))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.MaxPool2D((2,2))(X)

    X = tf.keras.layers.Conv2D(128, (3,3))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(512, activation = 'relu')(X)
    X = tf.keras.layers.Dense(512, activation = 'relu')(X)
    X = tf.keras.layers.Dense(10, activation = 'softmax')(X)
    
    model2 = tf.keras.models.Model(inputs = X_inp, outputs = X)
    model2.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy",
                   metrics = ['accuracy'])
    model2.fit(X_train, y_train, epochs = 3, batch_size = 128, )
    score = model2.evaluate(X_test, y_test)
    print("\n Model [{}] {}: {}".format(smodel2, model2.metrics_names, score), flush = True)
    model2.save(smodel2_fn)
  
  print("Pass 2: testing phase (evaluate)", flush = True) 
  score1 = model1.evaluate(X_test, Y_test, batch_size = 128)
  score2 = model2.evaluate(X_test, y_test, batch_size = 256)
  print("")
  print("  Model [{}] {}: {}".format(smodel1, model1.metrics_names,score1))
  print("  Model [{}] {}: {}".format(smodel2, model2.metrics_names,score2))


  print("Pass 2.1: testing phase (evaluate subset)", flush = True) 
  score1 = model1.evaluate(X_test_small, Y_test_small, batch_size = 128)
  score2 = model2.evaluate(X_test_small, y_test_small, batch_size = 256)
  print("")
  print("  Model [{}] {}: {}".format(smodel1, model1.metrics_names,score1))
  print("  Model [{}] {}: {}".format(smodel2, model2.metrics_names,score2))
  
  print("Pass 3: testing phase with tensor evaluation", flush = True) 
  model1_output = model1.outputs[0]
  model1_input = model1.inputs[0]
  sess = tf.keras.backend.get_session()
  y_hat1 = sess.run([model1_output], feed_dict = { model1_input : X_test_small })
  y_hat1 = y_hat1[0]
  acc1 = np.sum(np.argmax(y_hat1, axis = 1) == y_test_small) / y_test_small.shape[0]
  
  model2_output = model2.outputs[0]
  model2_input = model2.inputs[0]
  model2_learning_phase = tf.keras.backend.learning_phase()
  sess = tf.keras.backend.get_session()
  y_hat2 = sess.run([model2_output], feed_dict = { model2_input : X_test_small,
                    model2_learning_phase : 0})
  y_hat2 = y_hat2[0]
  acc2 = np.sum(np.argmax(y_hat2, axis = 1) == y_test_small) / y_test_small.shape[0]
  print("")
  print("  Model [{}] {:.3f}".format(smodel1, acc1))
  print("  Model [{}] {:.3f}".format(smodel2, acc2))
  