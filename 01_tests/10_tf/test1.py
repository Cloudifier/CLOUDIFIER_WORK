# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:51:29 2017

@author: Andrei
"""


import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

A = np.array([1,2,2])
B = np.array([1,1,1])
C=A.dot(B)
print(C)

tf_A = tf.constant([[1,2,2]], dtype = tf.float32, name = "A")
tf_B = tf.constant([[1,1,1]], dtype = tf.float32, name = "B")

tf_C = tf.matmul(tf_A, tf.transpose(tf_B), name = "C")

print(tf_C)

sess = tf.Session()
tf_out_C = sess.run(tf_C)

print(tf_out_C)

tf_A2 = tf.placeholder(dtype=tf.float32, shape=(1,3))
tf_B2 = tf.placeholder(dtype=tf.float32, shape=(1,3))

tf_C2 = tf.matmul(tf_A2, tf.transpose(tf_B2), name = "C2")

print(tf_C2)

tf_out_C2 = sess.run(tf_C2, 
                     feed_dict = {  
                                   tf_A2 : [[1,2,2]], 
                                   tf_B2 : [[1,1,1]]
                                  })
print(tf_out_C2)

mnist = fetch_mldata("MNIST original")
y = mnist.target
y = np.eye(10)[y.astype(int)]

X_train, X_test, y_train, y_test = train_test_split(mnist.data, y,
                                                    test_size = 0.16)

batch_size = 16
drop_keep = 0.5
nr_epochs = 20
h1_size = 256
sess.close()

mnist_graph = tf.Graph()

with mnist_graph.as_default():
  tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=())
  
  tf_X_batch = tf.placeholder(dtype = tf.float32, shape = (None,784))
  print(tf_X_batch)
  tf_y_batch = tf.placeholder(dtype = tf.float32, shape = (None,10))
  print(tf_y_batch)
  
  tf_weights_h1 = tf.get_variable(name = "h1_weights",dtype=tf.float32,
                               shape = (784,h1_size),
                               initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
  print(tf_weights_h1)
  tf_bias_h1 = tf.Variable(initial_value = np.zeros(shape=(h1_size)), 
                           dtype=tf.float32)
  print(tf_bias_h1)
  
  tf_weights_h2 = tf.get_variable(name="h2_weights",dtype=tf.float32,
                                  shape=(h1_size,10),
                                  initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
  print(tf_weights_h2)
  
  tf_bias_h2 = tf.Variable(initial_value = np.zeros(shape=(10)), 
                           dtype = tf.float32)
  print(tf_bias_h2)
  
  z1 = tf.matmul(tf_X_batch, tf_weights_h1) + tf_bias_h1
  print(z1)
  a1 = tf.nn.relu(z1)
  print(a1)
  
  a1drop = tf.nn.dropout(a1, keep_prob = tf_keep_prob)
  
  z2 = tf.matmul(a1drop, tf_weights_h2) + tf_bias_h2
  print(z2)
  
  tf_output = tf.nn.softmax(z2)
  print(tf_output)
  
  J = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=tf_y_batch, 
                                        logits = z2))
  print(J)
  
  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
  print(optimizer)
  optimizer_op = optimizer.minimize(J)
  print(optimizer_op)
  
  print("Creating variable initializer within graph")
  init_op = tf.global_variables_initializer()

print("Creating session")
session = tf.Session(graph = mnist_graph)

print("Initializing variables")
init_op.run(session = session)
  
for epoch in range(nr_epochs):
  minibatches =  int(X_train.shape[0] / batch_size)
  for mini in range(minibatches):
    X_batch = X_train[mini*batch_size:(mini+1)*batch_size,:]
    y_batch = y_train[mini*batch_size:(mini+1)*batch_size,:]

    _, batch_loss = session.run([optimizer_op, J], feed_dict={
                                                        tf_X_batch:X_batch,
                                                        tf_y_batch:y_batch,
                                                        tf_keep_prob:drop_keep
                                                      })
    if ((mini % 1000)==0):
      print("epoch {} minibatch {} loss: {:.3f}".format(epoch,mini,batch_loss))
  test_output = tf_output.eval(session = session,
                          feed_dict = {
                            tf_X_batch:X_test,
                            tf_y_batch:y_test,
                            tf_keep_prob:1
                          })
  preds_test = np.sum(np.argmax(test_output, axis=1)==np.argmax(y_test, axis=1))
  acc_test = preds_test / y_test.shape[0]
  print("Epoch {} test accuracy {:.2f}".format(epoch,acc_test*100))

  train_output = session.run(tf_output,
                          feed_dict = {
                            tf_X_batch:X_train,
                            tf_y_batch:y_train,
                            tf_keep_prob:1
                          })  
  preds_train = np.sum(np.argmax(train_output, axis=1)==np.argmax(y_train, axis=1))
  acc_train = preds_train / y_train.shape[0]
  print("Epoch {} train accuracy {:.2f}".format(epoch,acc_train*100))
  
session.close()
    

  
  
  

