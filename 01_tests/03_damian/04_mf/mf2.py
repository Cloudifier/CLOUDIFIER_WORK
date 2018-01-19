# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:13:10 2018

@author: damia
"""

import tensorflow as tf
import numpy as np

np.set_printoptions(precision=2, suppress=True)

UxP =[
      [100, 30, 40,  0,  0,  0],
      [ 50, 30,  0,  0,  0, 10],
      [ 20, 20, 90,100,  0, 70],
      [  0,  0, 50, 20, 10,  0],
      [  0,  0,  0,  0, 20,  0]
      ]

UxP = np.array(UxP, dtype=np.float32)
mask = UxP>0
prod_means = np.average(UxP, axis=0, weights=mask)
UxP_norm1 = UxP #- prod_means
UxP_norm2 = UxP_norm1 * mask
UNK = (~mask) * prod_means
UxP_norm3 = UxP_norm2 + UNK

input_data = UxP_norm3.copy()


print(input_data)
nr_users = UxP.shape[0]
nr_prods = UxP.shape[1]
nr_feats = 2 
lmbd = 0.01
lr = 0.005
epochs = 6000
steps = 5

g = tf.Graph()
with g.as_default():
    tf_u_feats = tf.Variable(initial_value=np.random.uniform(0.05,0.1, 
                                                             size=(nr_users, nr_feats)), 
                          dtype=tf.float32, name='user_feats')
    tf_p_feats = tf.Variable(initial_value=np.random.uniform(0.05,0.1, 
                                                             size=(nr_prods, nr_feats)),
                          dtype=tf.float32, name='prod_feats')

    #tf_prod_means = tf.constant(prod_means, dtype=tf.float32)        
    tf_y = tf.constant(input_data, dtype=tf.float32, name='ground_truth')

    tf_y_hat = tf.matmul(tf_u_feats, tf.transpose(tf_p_feats)) 
    #tf_y_hat += + tf_prod_means
    #tf_y_hat = tf.maximum(tf.zeros_like(tf_y_hat), tf_y_hat)
    #tf_y_masked = tf.boolean_mask(tf_y, mask)
    #tf_y_hat_masked = tf.boolean_mask(tf_y_hat, mask)
    tf_sub = tf.subtract(tf_y_hat, tf_y)
    #tf_sub = tf.subtract(tf_y_hat_masked, tf_y_masked)
    tf_loss = tf.reduce_sum(tf.pow(tf_sub,
                                   2), 
                            name = 'basic_loss')
    tf_reg = tf.add(tf.reduce_sum(tf.matmul(tf_u_feats, tf_u_feats, transpose_b=True)),
                    tf.reduce_sum(tf.matmul(tf_p_feats, tf_p_feats, transpose_b=True)),
                    name = 'reg')
          
    tf_loss = tf.add(tf_loss, lmbd * tf_reg, 'loss')
    opt = tf.train.AdamOptimizer(learning_rate = lr)
    opt_op = opt.minimize(tf_loss)
    
    tf_op_nn_U = tf.assign(tf_u_feats, 
                           tf.maximum(tf.zeros_like(tf_u_feats), tf_u_feats))
    tf_op_nn_P = tf.assign(tf_p_feats, 
                           tf.maximum(tf.zeros_like(tf_p_feats), tf_p_feats))
    tf_op_nn = tf.group(tf_op_nn_U, tf_op_nn_P)
    
    init = tf.global_variables_initializer()

sess = tf.Session(graph=g)
init.run(session=sess)
show_range = epochs // steps
for epoch in range(epochs):    
    loss, _ = sess.run([tf_loss, opt_op])
    tf_op_nn.run(session=sess)
    if (epoch % show_range) == 0:
        print("Epoch {} loss {:.2f}".format(epoch, loss))

u_feats = tf_u_feats.eval(session=sess)
p_feats = tf_p_feats.eval(session=sess)
ypred = u_feats.dot(p_feats.T)
print(ypred)
print(ypred+prod_means)