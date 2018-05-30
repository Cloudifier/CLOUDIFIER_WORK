# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:42:27 2018

@author: Andrei Ionut Damian
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

w1_t = 74.3
w2_t = 28.1

x1=[]
x2=[]
y=[]

for xx1 in range(5):
  for xx2 in range(7):
    y.append(xx1*w1_t +xx2*w2_t)
    x1.append(xx1)
    x2.append(xx2)
    


tf_g = tf.Graph()

with tf_g.as_default():
  tf_w1 = tf.Variable(initial_value=0,dtype=tf.float32, name='w1')
  tf_w2 = tf.Variable(initial_value=0,dtype=tf.float32, name='w2')
  tf_x1 = tf.placeholder(dtype=tf.float32, name='x1')
  tf_x2 = tf.placeholder(dtype=tf.float32, name='x2')
  tf_y = tf.placeholder(dtype=tf.float32, name='y')
  #tf_lr = tf.placeholder(dtype=tf.float32, name='lr')
  tf_yhat = tf_x1 * tf_w1 + tf_x2 * tf_w2
  tf_err = tf.pow(tf_yhat - tf_y, 2)
  #tf_deriv_w1 = 2*(tf_yhat-tf_y)*tf_x1
  #tf_deriv_w2 = 2*(tf_yhat-tf_y)*tf_x2
  #or
  #tf_deriv_w1 = tf.gradients(tf_err, tf_w1)
  #tf_deriv_w2 = tf.gradients(tf_err, tf_w2)
  
  #tf_update_w1 = tf.assign(tf_w1,tf_w1 - tf_lr*tf_deriv_w1)
  #tf_update_w2 = tf.assign(tf_w2,tf_w2 - tf_lr*tf_deriv_w2)
  
  tf_optimizer = tf.train.AdamOptimizer(learning_rate=1)
  tf_opt_op = tf_optimizer.minimize(tf_err)

  tf_initializer = tf.global_variables_initializer()  
  
tf_sess = tf.Session(graph=tf_g)
tf_sess.run(tf_initializer)

epochs = 20
errs=[]
for epoch in range(epochs):
  for i in range(len(y)):  
    xx1 = x1[i]
    xx2 = x2[i]
    yy = y[i]
    #_,_,w1,w2,err = tf_sess.run([tf_update_w1,tf_update_w2, tf_w1, tf_w2,tf_err], 
    #                        feed_dict={tf_x1:xx1,tf_x2:xx2,tf_y:yy,tf_lr:0.005})
    _, w1,w2,err = tf_sess.run([tf_opt_op,tf_w1,tf_w2,tf_err],
                               feed_dict={tf_x1:xx1,tf_x2:xx2,tf_y:yy})
    errs.append(err)
    
    print("Epoch {} Iter {} err={:.2f} w1={:.2f} w2={:.2f}".format(epoch, i, err, w1,w2))

plt.plot(range(len(errs)),errs)