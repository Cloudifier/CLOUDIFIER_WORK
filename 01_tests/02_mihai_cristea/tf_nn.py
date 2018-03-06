# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:23:15 2018

@author: Andrei Ionut Damian
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

import tensorflow as tf


mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target.astype(int)
label_to_oh=lambda x: np.eye(10)[x]
y = label_to_oh(y)
X /= 255
batch_size = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
m, n = X_train.shape

h1 = 50

g = tf.Graph()
with g.as_default():  
  tf_x = tf.placeholder(dtype=tf.float32, shape=[None,n], name='x_input')
  tf_y = tf.placeholder(dtype=tf.float32, shape=[None,10], name='y_input')
  
  tf_Wxh1 = tf.Variable(np.random.uniform(-0.01, 0.01, size=(n, h1)).astype(np.float32), name='weights_h1')
  tf_bxh1 = tf.Variable(np.random.uniform(-0.01, 0.01, size=(h1,)).astype(np.float32), name='bias_h1')
  
  tf_Wh1s = tf.Variable(np.random.uniform(-0.01, 0.01, size=(h1,10)).astype(np.float32), name='softmax_weights')
  tf_bh1s = tf.Variable(np.random.uniform(-0.01, 0.01, size=(10,)).astype(np.float32), name='softmax_bias')
  
  tf_zh1 = tf.identity(tf.matmul(tf_x, tf_Wxh1) + tf_bxh1, name='z_h1')
  tf_ah1 = tf.nn.relu(tf_zh1, name='relu_h1')
  
  tf_logits = tf.identity(tf.matmul(tf_ah1, tf_Wh1s) + tf_bh1s, name='logits')
  
  tf_yhat = tf.nn.softmax(tf_logits)
  
  tf_loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=tf_logits, name='loss_pre_mean')
  tf_loss = tf.reduce_mean(tf_loss1, name='loss')
  
  tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
  tf_opt_op = tf_optimizer.minimize(tf_loss)
  
  tf_init = tf.global_variables_initializer()

sess = tf.Session(graph=g)
sess.run(tf_init)

for epoch in range(10):
  for i in range(m//100 - 1):
    x_batch = X_train[i*100 : (i+1)*100]
    y_batch = y_train[i*100 : (i+1)*100]
    
    _, loss = sess.run([tf_opt_op, tf_loss], feed_dict={
                  tf_x: x_batch,
                  tf_y: y_batch
        })
    if i % 10 == 0:
      print("Evaluation at Epoch{} Iter #{}:".format(epoch, i,))
      yhat = sess.run(tf_yhat, feed_dict={
            tf_x: X_test,
            tf_y: y_test
          })
      acc = (np.argmax(yhat, axis=1) == y_test.argmax(axis=1)).sum() / y_test.shape[0]
      print("    Loss:{:.3f} Acc:{:.2f}%".format(loss, acc * 100))
  