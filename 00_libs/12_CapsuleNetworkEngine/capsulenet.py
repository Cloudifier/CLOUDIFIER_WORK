# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:23:24 2018

@author: Andrei Ionut Damian
"""

import tensorflow as tf

__version__   = "0.1"
__author__    = "Cloudifier"
__copyright__ = "(C) Cloudifier SRL"
__project__   = "Cloudifier"  
__module__    = "CapsuleNetworkEngine"
__reference__ = "Based on G. Hinton Capsule Networks, https://arxiv.org/abs/1710.09829"

import os

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.", flush = True)
    #raise Exception("Couldn't find google drive folder!")
  else:  
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib

class SimpleLogger:
  def __init__(self):
    return
  def VerboseLog(self, _str, show_time):
    print(_str, flush = True)

def LoadLogger(lib_name, config_file):
  module = load_module("logger", "logger.py")
  if module is not None:
    logger = module.Logger(lib_name = lib_name, config_file = config_file)
  else:
    logger = SimpleLogger()
  return logger

class ImageCapsuleNetwork:
  
  def __init__(self, n_classes, input_shape, logger,
               n_primary_maps=32, n_primary_dims=8, n_secondary_dims=16,
               conv1_kernel=9, conv1_strides=1,
               conv2_kernel=9, conv2_strides=2,
               ):
    assert len(input_shape)==3, "Input shape must be (H,W,C)"
    H,W,C = input_shape
    self.logger = logger
    self.H = H
    self.W = W
    self.C = C
    self.B = None
    self.conv1_kernel = conv1_kernel
    self.conv2_kernel = conv2_kernel
    self.conv1_strides = conv1_strides
    self.conv2_strides = conv2_strides
    self.n_primary_maps = n_primary_maps
    self.n_primary_dims = n_primary_dims    
    self.n_classes = n_classes
    self.n_secondary_caps = self.n_classes
    self.n_secondary_dims = n_secondary_dims
    

    self._build_graph()
    return
  
  def _build_graph(self):
    # following code works only for H==W
    conv1_h = (self.H - self.conv1_kernel + 1) // self.conv1_strides
    conv2_h = (conv1_h - self.conv2_kernel + 1) // self.conv2_strides
    final_conv_h = conv2_h
    final_conv_w = conv2_h
    self.n_primary_caps = self.n_primary_maps * final_conv_h * final_conv_w  
    

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.tf_x_train = tf.placeholder(dtype=tf.float32, 
                                       shape=(self.B,self.H,self.W,self.C),
                                       name = 'x_train')
      self.tf_y_train = tf.placeholder(shape=[None], dtype=tf.int64, name="y_train")
      
      
      tf_conv1 = tf.layers.conv2d(self.tf_x_train, name="conv1",
                                  filters=256, kernel_size=self.conv1_kernel,
                                  strides=self.conv1_strides, padding='valid',
                                  activation=tf.nn.relu)
      conv2filters = self.primarycaps_n_maps * self.primarycaps_n_dims
      tf_conv2 = tf.layers.conv2d(tf_conv1, name="conv2", 
                                  filters=conv2filters, kernel_size=self.conv2_kernel,
                                  strides=self.conv2_strides, padding='valid',
                                  activation=tf.nn.relu)
      
      tf_primcap_raw = tf.reshape(tf_conv2, 
                                  shape=[-1, self.n_primary_caps, self.n_primary_dims],
                                  name="primary_capsule_raw")
      
      tf_primcap_output = self.squash(tf_primcap_raw, name="primary_capsule_output")
      
      init_sigma = 0.01
      tf_W_init = tf.random_normal(
          shape=(1, self.n_primary_caps, self.n_secondary_caps, 
                 self.n_secondary_dims, self.n_primary_dims),
          stddev=init_sigma, dtype=tf.float32, name="W_init")
      tf_W = tf.Variable(tf_W_init, name="W")      
      
      batch_size = tf.shape(self.tf_x_train)[0]
      tf_W_tiled = tf.tile(tf_W, [batch_size, 1, 1, 1, 1], name="W_tiled")      

      tf_primcap_output_expanded = tf.expand_dims(tf_primcap_output, -1,
                                                  name="primary_capsule_output_expanded")
      tf_primcap_output_tile = tf.expand_dims(tf_primcap_output_expanded, 2,
                                              name="primary_capsule_output_tile")
      tf_primcap_output_tiled = tf.tile(tf_primcap_output_tile, 
                                        [1, 1, self.n_secondary_caps, 1, 1],
                                        name="primary_capsule_output_tiled")
      
      tf_seccap_predicted = tf.matmul(tf_W_tiled, tf_primcap_output_tiled,
                                      name="secondary_capsule_predicted")
      
      # routing stage 1
      
      tf_raw_weights = tf.zeros([batch_size, 
                                 self.n_primary_caps, 
                                 self.n_secondary_caps, 
                                 1, 
                                 1],
                                dtype=tf.float32, name="raw_weights")
      
      tf_routing_weights = tf.nn.softmax(tf_raw_weights, dim=2, 
                                         name="routing_weights")
      

      tf_weighted_predictions = tf.multiply(tf_routing_weights, 
                                            tf_seccap_predicted,
                                            name="weighted_predictions")
      tf_weighted_sum = tf.reduce_sum(tf_weighted_predictions, axis=1, keep_dims=True,
                                      name="weighted_sum")      

      tf_seccaps_output_round_1 = self.squash(tf_weighted_sum, axis=-2,
                                              name="secondary_capsule_output_round_1")
      
      # round 2

      tf_seccaps_output_round_1_tiled = tf.tile(tf_seccaps_output_round_1, 
                                                [1, self.n_primary_caps, 1, 1, 1],
                                                name="secondary_capsule_output_round_1_tiled")   
      
      tf_agreement = tf.matmul(tf_seccap_predicted, tf_seccaps_output_round_1_tiled,
                               transpose_a=True, name="agreement")      
      
      tf_raw_weights_round_2 = tf.add(tf_raw_weights, tf_agreement,
                                      name="raw_weights_round_2")   
      
      tf_routing_weights_round_2 = tf.nn.softmax(tf_raw_weights_round_2,
                                                 dim=2,
                                                 name="routing_weights_round_2")
      tf_weighted_predictions_round_2 = tf.multiply(tf_routing_weights_round_2,
                                                    tf_seccap_predicted,
                                                    name="weighted_predictions_round_2")
      tf_weighted_sum_round_2 = tf.reduce_sum(tf_weighted_predictions_round_2,
                                              axis=1, keep_dims=True,
                                              name="weighted_sum_round_2")
      tf_seccap_output_round_2 = self.squash(tf_weighted_sum_round_2,
                                             axis=-2,
                                             name="caps2_output_round_2")      
       
      tf_seccap_output = tf_seccap_output_round_2
      
      
      # now the output
      
      tf_y_proba = self.safe_norm(tf_seccap_output, axis=-2, name="y_proba")
      tf_y_proba_argmax = tf.argmax(tf_y_proba, axis=2, name="y_proba")
      tf_y_pred = tf.squeeze(tf_y_proba_argmax, axis=[1,2], name="y_pred")
      
      MAYBE DO LOOP !
      
      
    return

  def squash(self, s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
      squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                   keep_dims=True)
      safe_norm = tf.sqrt(squared_norm + epsilon)
      squash_factor = squared_norm / (1. + squared_norm)
      unit_vector = s / safe_norm
    return squash_factor * unit_vector    
  
  
  def safe_norm(self,s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
      squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                   keep_dims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)  

  def log(self,_str,_st=False):
    self.logger.VerboseLog(_str,_st)
    return
  
  


if __name__ == '__main__':
  logger = LoadLogger('ImageCapsNet','config.txt')
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(logger.GetDataFolder())  
