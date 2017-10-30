# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 08:12:18 2017

@author: Andrei
"""

import numpy as np

import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt


from keras.datasets import mnist

from tqdm import tqdm
import socket

from datetime import datetime as dt

FULL_DEBUG = False
app_log = list()
SHOW_TIME = False
file_prefix = dt.now().strftime("%Y%m%d_%H%M%S") 
log_file = '_log.txt'
log_results_file = file_prefix + "_RESULTS.txt"
__lib__= "FCNK"

def _logger(logstr, show = False):
  """
  log processing method 
  """
  nowtime = dt.now()
  strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(__lib__))
  if SHOW_TIME:
    logstr = strnowtime + logstr
  app_log.append(logstr)
  if show:
      print(logstr, flush = True)
  try:
      log_output = open(log_file, 'w')
      for log_item in app_log:
        log_output.write("%s\n" % log_item)
      log_output.close()
  except:
      print(strnowtime+"Log write error !", flush = True)
  return


  
def get_machine_name():
  if socket.gethostname().find('.')>=0:
      name=socket.gethostname()
  else:
      name=socket.gethostbyaddr(socket.gethostname())[0]
  return name

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__=="__main__":
  
  
  nr_epochs = 10
  nr_channels = 1
  nr_classes = 10
  batch_size = 128
  test_size = 2500 #limit images per test (full GPU loaded)
  nr_tests = test_size
  img_h = 28
  img_w = 28
  
  filter1_h = 3
  filter1_w = 3  
  depth1 = 16
  
  drop_rate = 0.5 
  
  filter2_h = 4
  filter2_w = 4 
  depth2 = 32
  
  filter3_h = 4
  filter3_w = 4   
  depth3 = 512
  
  
  
  print("Loading data...", flush=True)
  (X_train, y_tr), (X_test, y_ts) = mnist.load_data()
  X_train = X_train.astype("float32") / 255
  X_test = X_test.astype("float32") / 255
  nr_train = X_train.shape[0]
  nr_test = X_test.shape[0]
  X_train = X_train.reshape(nr_train, img_h,img_w,nr_channels)
  X_test = X_test.reshape(nr_test,img_h,img_w,nr_channels)
  np.set_printoptions(suppress = True, edgeitems = 5, linewidth = 100,
                      formatter={'float':'{: 0.2f}'.format})

  y_train = np_utils.to_categorical(y_tr, nr_classes)
  y_test = np_utils.to_categorical(y_ts, nr_classes)  
  
  if get_machine_name() in ['DAMIAN']:
    test_size = 250
    batch_size = 64
    nr_epochs = 2
    nr_tests = 200 
    
  # random image preparation
  #np.random.seed(1234)
  np_idx = np.random.randint(0,9999,size=test_size)
  test_images = list()  
  x_samples = X_test[np_idx]
  y_samples = y_test[np_idx]
  print("Test config: {}".format(np.argmax(y_samples[1:10], 1)))
  
  FULL_DISPLAY = False
  test_images = list()
  
  for i in tqdm(range(np_idx.shape[0])):
     
    new_h = np.random.randint(50,100)+img_h
    new_w = np.random.randint(50,100)+img_w
    test_img = np.zeros(shape=(new_h,new_w))
    pos_r = np.random.randint(0, new_h-img_h-1)
    pos_c = np.random.randint(0, new_w-img_w-1)
    src_h = x_samples.shape[1]
    src_w = x_samples.shape[2]
    
    test_img[pos_r:(pos_r+src_h), pos_c:(pos_c+src_w)] = x_samples[i,:,:,0]
    
    test_img = test_img.reshape(1,new_h,new_w,1)
    test_images.append(test_img)


  
  init_tnorm = 0
  init_xavie = 1
  init_he = 2
  inits = ["truncated_normal", "xavier", "he"]
  for WEIGHTS_INIT in inits:
    print("Preparing graph with {} init...".format(WEIGHTS_INIT), flush=True)  
    graph = tf.Graph()
    
    with graph.as_default():
      
      def GetConvFilter(h,w,d1,d2, name):
        assert WEIGHTS_INIT in inits
        if WEIGHTS_INIT == inits[init_xavie]:
          tf_filter = tf.get_variable(name=name, shape=[h,w,d1,d2], dtype=tf.float32, 
                                      initializer = tf.contrib.layers.xavier_initializer_conv2d() )
        elif WEIGHTS_INIT == inits[init_tnorm]:
          tf_filter = tf.Variable(initial_value = tf.truncated_normal([h,w, d1, d2], stddev=0.1),
                                  name = name)
        elif WEIGHTS_INIT == inits[init_he]:
          # To get Delving Deep into Rectifiers, use (Default):
          # factor=2.0 mode='FAN_IN' uniform=False
          # To get Convolutional Architecture for Fast Feature Embedding, use:
          # factor=1.0 mode='FAN_IN' uniform=True
          # To get Understanding the difficulty of training deep feedforward neural networks, use:
          # factor=1.0 mode='FAN_AVG' uniform=True.        
          # To get xavier_initializer use either:
          # factor=1.0 mode='FAN_AVG' uniform=True, or
          # factor=1.0 mode='FAN_AVG' uniform=False.        
          tf_filter = tf.get_variable(name=name, shape=[h,w,d1,d2], dtype=tf.float32, 
                initializer = tf.contrib.layers.variance_scaling_initializer(
                                                        factor=2.0,
                                                        mode='FAN_IN',
                                                        uniform=False,
                                                     ))
  
        return tf_filter
      
      tf_test_dataset = tf.placeholder(dtype=tf.float32, shape=(None,None,None,1), name = "Test_data")
      he_init = tf.contrib.layers.variance_scaling_initializer()
      
      tf_inputs = tf.placeholder(dtype=tf.float32, shape=(None,None,None,1), name = "Input_data")
      tf_labels = tf.placeholder(dtype=tf.float32, shape=(None, nr_classes), name = "Labels" )
      
      tf_dropout = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")
      
      conv1_filters = GetConvFilter(filter1_h,filter1_w,nr_channels,depth1,"conv1_filter")
      conv1_bias = tf.Variable(tf.constant(0.01, shape=[depth1]), name = "conv1_bias")
      
      conv2_filters = GetConvFilter(filter2_h,filter2_w,depth1,depth2,"conv2_filter")
      conv2_bias = tf.Variable(tf.constant(0.01, shape=[depth2]), name = "conv2_bias")
      
  
      conv3_filters = GetConvFilter(filter3_h,filter3_w,depth2,depth3,"conv3_filter")
      conv3_bias = tf.Variable(tf.constant(0.01, shape=[depth3]), name = "conv3_bias")
      
      final_weights = tf.Variable(initial_value = tf.truncated_normal(
                                        [depth3, nr_classes], 
                                        stddev=0.1), name= "readout_weights")
      final_bias = tf.Variable(tf.constant(0.01, shape=[nr_classes]), name = "readout_bias")
      
      # model
      def fcn_model(input_data, drop_rate):
      
        conv1_layer = tf.nn.elu(tf.nn.conv2d(input_data,conv1_filters,strides = [1, 1, 1, 1], padding='SAME') + conv1_bias)
        conv2_layer = tf.nn.elu(tf.nn.conv2d(conv1_layer, conv2_filters,strides = [1, 1, 1, 1], padding='SAME') + conv2_bias)   
        drop_layer = tf.nn.dropout(conv2_layer, keep_prob = drop_rate)      
        conv3_layer = tf.nn.elu(tf.nn.conv2d(drop_layer, conv3_filters,strides = [1, 1, 1, 1], padding='SAME') + conv3_bias)      
        global_pool_layer = tf.reduce_max(conv3_layer, reduction_indices=[1,2], keep_dims=False) # max pool layer basically
        return tf.matmul(global_pool_layer, final_weights) + final_bias
      
      logits = fcn_model(tf_inputs, 0.5)
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits)
          ) 
      optimizer = tf.train.AdamOptimizer(0.01)
      
      optimizer_op  = optimizer.minimize(loss=loss)
  
      test_prediction = tf.nn.softmax(fcn_model(tf_test_dataset, 1.0))
  
    loss_list = []
    print("Starting training...", flush=True)  
    with tf.Session(graph = graph) as sess:
      tf.global_variables_initializer().run()
      for epoch in range(nr_epochs):
        print("Training epoch {}/{}".format(epoch+1,nr_epochs), flush = True)
        nr_steps = y_train.shape[0]  // batch_size
        tpbar = tqdm(range(nr_steps))
        for step in tpbar:       
          offset = (step * batch_size) % (y_train.shape[0] - batch_size)
          batch_data = X_train[offset:(offset + batch_size), :, :, :]
          batch_labels = y_train[offset:(offset + batch_size), :]
          feed_dict = {tf_inputs : batch_data, tf_labels : batch_labels}
          _, l = sess.run([optimizer_op, loss], feed_dict = feed_dict)
          loss_list.append(l)
          tpbar.set_description("Batch {} loss: {:.2f}".format(step,l))
        print("Evaluation test data:{}...".format(x_samples.shape), flush=True)
        test_preds = test_prediction.eval(feed_dict={tf_test_dataset:x_samples})
        print("Epoch {} summary:\n Last batch loss = {:.2f}\n Partial test accuracy: {:.3f}%".format(
                epoch,l,accuracy(test_preds,y_samples)), flush=True)
        ## done current epoch
      ## done all epochs  
      plt.plot(loss_list)
      
      
      #NEXT VARIABLE SIZE :))))
      
      preds = []
      y_preds = []
      for i in tqdm(range(nr_tests)):
        test_img = test_images[i]
        test_pred = test_prediction.eval(feed_dict={tf_test_dataset: test_img}).ravel()
        y_preds.append(test_pred)
        y_t = np.argmax(y_samples[i])
        yh = np.argmax(test_pred)
        pred = np.argmax(y_samples[i]) == np.argmax(test_pred)
        preds.append(pred)
        np.set_printoptions(formatter={'float':'{: 0.2f}'.format}) 
        if FULL_DEBUG:
          if not pred:
            _logger("Label/Prediction: {}/{} Correct: {} Imagesize: {}".format(
                y_t,yh,pred, test_img.shape))
            _logger("  Prediction: {}".format(test_pred))
            _logger("  y_test:     {}".format(np.array(y_samples[i],
                                                 dtype=np.float32)))  
      print("Accuracy {:.3f} for initialization {}".format(
                np.sum(preds)/len(preds), WEIGHTS_INIT))
                   
            
            
          
    for l in app_log:
      print(l)
      

  
  
  
  
  
  