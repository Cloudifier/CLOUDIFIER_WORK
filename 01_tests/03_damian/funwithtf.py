# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:20:13 2018

@author: Andrei Ionut Damian
"""

import numpy as np
import tensorflow as tf

a = np.random.randint(1,10,size=(3,2))
b = np.random.randint(1,5, size=(2,3))
c = np.random.randint(1,4, size=(1,3))

g = tf.Graph()

with g.as_default():  
  tf_a = tf.constant(a)
  tf_b = tf.constant(b)
  tf_c = tf.constant(c)
  tf_d = tf.matmul(tf_a, tf_b)
  tf_e = tf.add(tf_d, tf_c)

sess = tf.Session(
    graph = g,
    config=tf.ConfigProto(log_device_placement=True))

e = sess.run(tf_e)



