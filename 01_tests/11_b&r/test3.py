# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:16:24 2018

@author: Andrei Ionut Damian
"""


from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.width', 500)
data = load_boston()
df = pd.DataFrame(data["data"],columns=data["feature_names"])
df["target"] = data["target"]
print(df.head(10))
X = df.iloc[:,:-1].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
n = X.shape[1]
Y = df.iloc[:,-1:].values
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1)

tf_g = tf.Graph()

with tf_g.as_default():  
  tf_w = tf.Variable(initial_value=np.zeros(n),dtype=tf.float32, name='w') 
  tf_b = tf.Variable(initial_value=0,dtype=tf.float32,name='b')
  tf_x = tf.placeholder(dtype=tf.float32,shape=[n],name='x')  
  tf_y = tf.placeholder(dtype=tf.float32, name='y')

  tf_yhat = tf.reduce_sum(tf_w * tf_x) + tf_b
  tf_err = tf.pow(tf_yhat - tf_y, 2)
  
  tf_optimizer = tf.train.AdamOptimizer(learning_rate=0.025)
  tf_opt_op = tf_optimizer.minimize(tf_err)

  tf_initializer = tf.global_variables_initializer()  
  
tf_sess = tf.Session(graph=tf_g)
tf_sess.run(tf_initializer)
  
epochs = 20
errs=[]
for epoch in range(epochs):
  epoch_error = 0 
  for i in range(X_train.shape[0]):
    X_obs = X_train[i]
    Y_obs = y_train[i]
    err, _ = tf_sess.run([tf_err,tf_opt_op],feed_dict={tf_x:X_obs,tf_y:Y_obs})
    epoch_error += err[0]
  print("Epoch: {} error= {:.2f} ".format(epoch,epoch_error/X_train.shape[0]))
  errs.append(epoch_error/X_train.shape[0])

plt.plot(errs)
  
for i in range(X_test.shape[0]):
  Xt_obs = X_test[i]
  y = y_test[i][0]
  yhat = tf_sess.run(tf_yhat, feed_dict={tf_x:Xt_obs})
  rmse = np.sqrt((yhat-y)**2)
  print("Obs: {} Real= {:.1f} Predicted= {:.1f} rmse= {}".format(i, y, yhat, rmse))
  
    
  
  

