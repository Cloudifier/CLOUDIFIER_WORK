# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:48:33 2017

@author: Andrei
"""


import numpy as np

X = np.array(
    [
     [0, 1, 100, 0,  32], # obs 1
     [0, 1, 200, 1,  47], # obs 2 ..
     [1, 0, 300, 0,  54],
     [1, 0, 200, 1,  23],
     [1, 0, 200, 0,  15],
     [0, 2, 200, 0,  18],
     [0, 2, 200, 0,  47],
     [3, 1, 200, 0,  30],
     [0, 1, 200, 1,  47],
     [5, 2, 200, 1,  60], # obs 10
        ]
    )

y = np.array([
    1000,
    1200,
    2000,
    1900,
    1800,
    1400,
    1700,
    2100,
    900,
    4000,    
    ])

y = y.reshape((10,1))

theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

yhat = X.dot(theta)


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()

X_boston = data.data
y_boston = np.expand_dims(data.target,1)

X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston,
                                                    test_size = 0.3)

theta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

yhat_boston = X_train.dot(theta)
yhat_test = X_test.dot(theta)
print("TRAIN RMSE: {:.2f}".format(np.sqrt(np.mean((yhat_boston - y_train)**2))))
print("TEST  RMSE: {:.2f}".format(np.sqrt(np.mean((yhat_test - y_test)**2))))


import tensorflow as tf


tf_graph = tf.Graph()

with tf_graph.as_default():
  tf_x_input = tf.placeholder(dtype = tf.float32, 
                              shape = (None, X_boston.shape[1]),
                              name = "x_input")
  tf_y_input = tf.placeholder(dtype = tf.float32, shape = (None, 1),
                              name = "y_input")
  
  tf_theta = tf.Variable(initial_value = np.zeros((X_boston.shape[1],1)),
                                                   dtype = tf.float32)
  
  tf_x_t = tf.transpose(tf_x_input)
  tf_x_sq = tf.matmul(tf_x_t, tf_x_input)
  tf_x_sq_inv = tf.matrix_inverse(tf_x_sq)
  tf_x_res = tf.matmul(tf_x_sq_inv, tf_x_t)
  tf_theta_mul = tf.matmul(tf_x_res, tf_y_input)
  tf_theta_calc = tf.assign(tf_theta, tf_theta_mul)
  
  tf_yhat = tf.matmul(tf_x_input, tf_theta)

sess = tf.Session(graph = tf_graph)

_ = sess.run(tf_theta_calc,
             feed_dict={
                 "x_input:0" : X_train,
                 "y_input:0" : y_train
                 })

theta_res = tf_theta.eval(session = sess)

yhat_train_tf = tf_yhat.eval(session = sess,
                          feed_dict = {
                              "x_input:0" : X_train
                              })

yhat_test_tf = tf_yhat.eval(session = sess,
                          feed_dict = {
                              "x_input:0" : X_test,
                              })

print("TF TRAIN RMSE: {:.2f}".format(np.sqrt(np.mean((yhat_train_tf - y_train)**2))))
print("TF TEST  RMSE: {:.2f}".format(np.sqrt(np.mean((yhat_test_tf - y_test)**2))))





