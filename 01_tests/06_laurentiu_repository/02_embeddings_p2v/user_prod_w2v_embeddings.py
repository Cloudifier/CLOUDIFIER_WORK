# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:33:56 2018

@author: LaurentiuP
"""

import os
import tensorflow as tf
import math
import h5py
import numpy as np
from time import time
import pandas as pd
from tqdm import trange

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

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
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
  module_lib   = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
  print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return module_lib


valid_embed_contexts = ["FLATTEN", "MEAN", "SUM"]
valid_embed_architectures = ["SKIP-GRAM"]  ### JUST SKIP-GRAM
valid_model_weights = ["products_embeddings", "users_embeddings", "sm_weights", "sm_biases"]


class EmbeddingsMaster:
  def __init__(self, nr_products, config_file='config.txt',
               architecture='SKIP-GRAM', nr_users=None):
    self.CONFIG = None
    logger_module = load_module('logger','logger.py')
    self.logger = logger_module.Logger(lib_name = "RECOMv3",
                                config_file = config_file,
                                TF_KERAS = True)
    self._log("Initializing Embeddings Generator (used in recommendations) ...")
  
    self.CONFIG = self.logger.config_data
    self.ARCHITECTURE = architecture
    self.nr_products = nr_products
    self.nr_users = nr_users
    
    if (self.nr_users is None) and (self.CONFIG['USERS']['ENABLE'] == 1):
      raise Exception("ERROR! Users Embeddings enabled, but unknown nr_users!")
    if (self.nr_users is not None) and (self.CONFIG['USERS']['ENABLE'] == 0):
      raise Exception("ERROR! Users Embedding not enabled, but nr_users specified!")
    
    self.str_products_embeddings = 'products_embeddings'
    self.str_users_embeddings = 'users_embeddings'
    
    self.compute_users_embeddings = bool(self.CONFIG['USERS']['ENABLE'])
    self.file_prefix = self.logger.file_prefix
    self.model_name = None
    self._base_folder = self.logger.GetBaseFolder()
    self.FRAMEWORK = self.CONFIG["FRAMEWORK"]
    self.dict_embeddings = {self.str_products_embeddings: None, self.str_users_embeddings: None}
    self.dict_norm_embeddings = {self.str_products_embeddings: None, self.str_users_embeddings: None}
    self.dict_nr_embeddings = {
        self.str_products_embeddings: self.CONFIG['PRODUCTS']['NR_EMBEDDINGS'],
        self.str_users_embeddings: None
    }
    if self.compute_users_embeddings:
      self.dict_nr_embeddings[self.str_users_embeddings] = self.CONFIG['USERS']['NR_EMBEDDINGS']
    
    self._check_config_errors()
    
    if self.FRAMEWORK["NAME"].upper() == "TENSORFLOW":
      self.tf_graph = None
      self.model_cache = None
      self._cfg_embeddings_context = self.FRAMEWORK["EMBEDDINGS_CONTEXT"]
      self._model_weights = dict()

      self._init_tensorflow_model()
    elif self.FRAMEWORK["NAME"].upper() == "KERAS":
      pass
    
    self._log("Initialized recommender model: nr_prod={}, nr_users={}, nr_p_embeds={}, "\
              "nr_u_embeds={}, architecture={}".format(nr_products, nr_users,
              self.dict_nr_embeddings[self.str_products_embeddings],
              self.dict_nr_embeddings[self.str_users_embeddings],
              self.ARCHITECTURE.upper()))
    return
    
    
  def _check_config_errors(self):

    if self.ARCHITECTURE.upper() not in valid_embed_architectures:
      err_msg = 'ERROR! Unknown architecure'
      self._log(err_msg)
      raise Exception(err_msg)
    
    if self.CONFIG["LOAD_MODEL"] != "":
      if not os.path.exists(os.path.join(self._base_folder, self.CONFIG["LOAD_MODEL"])):
        err_msg = "ERROR! Specified model to load does not exist."
        self._log(err_msg)
        raise Exception(err_msg)
      

    if (self.FRAMEWORK["NAME"].upper() == "TENSORFLOW") and (self.ARCHITECTURE.upper() == "CBOW"):
      if self.FRAMEWORK["EMBEDDINGS_CONTEXT"].upper() not in valid_embed_contexts:
        err_msg = 'ERROR! Unknown config_data["EMBEDDINGS_CONTEXT"]'
        self._log(err_msg)
        raise Exception(err_msg)
    
    return
  
  def _log(self, str_msg, results = False, show_time = False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return
  
  
  
  def _init_tensorflow_model(self):
    """
    Processes tf_config and creates the computation graph
    """

    self._log("  Using pure Tensorflow in order to use sampled_softmax ...")

    # If a previous model is specified in tf_config, we initialize the weights with the values stored in the '.h5' file
    if self.CONFIG["LOAD_MODEL"] != "":
      restore_point = os.path.join(self._base_folder, self.CONFIG["LOAD_MODEL"])
      self.model_cache = self._load_tensorflow_model(restore_point)
      self.dict_embeddings[self.str_products_embeddings] = self.model_cache[self.str_products_embeddings].value
      if self.compute_users_embeddings:
        self.dict_embeddings[self.str_users_embeddings] = self.model_cache[self.str_users_embeddings].value

    self._create_tensorflow_graph()
    return
  
  
  def __check_loaded_model_errors(self, model_cache):
    keys = list(model_cache.keys())

    for w in valid_model_weights:
      if (w is self.str_users_embeddings) and (not self.compute_users_embeddings):
        continue
      
      if w not in keys:
        err_msg = "ERROR! Weights [{}] not found in loaded model.".format(w)
        self._log(err_msg)
        raise Exception(err_msg)
    
    return

  def _load_tensorflow_model(self, restore_point):
    """
    TODO
    """
    self._log('  Restoring model weights: {}'.format(restore_point))
    model_cache = h5py.File(restore_point, 'r')
    self.__check_loaded_model_errors(model_cache)
    self._log('  Done restoring model weights: {}'.format(list(model_cache.keys())),
              show_time = True)
    return model_cache


  def _save_tensorflow_model(self, saved_model_filename):
    """
    TODO
    """
    self._log("  Saving model weights ...")
    model = h5py.File(os.path.join(self.logger._outp_dir, saved_model_filename + '.h5'), 'w')
    for key, value in self._model_weights.items():
      model.create_dataset(key, data = value)

    model.close()
    self._log("  Model {} saved.".format(saved_model_filename), show_time = True)
    return
  

  def _create_tensorflow_graph(self):
    """
    TODO
    """

    self._log("  Initializing Tensorflow graph ...")
    self.tf_graph = tf.Graph()    
    with self.tf_graph.as_default():
      self.tf_train_prod_inputs, self.tf_train_user_inputs, self.tf_train_labels = self.__create_tf_graph_placeholders()
      self.tf_prod_embeddings, self.tf_user_embeddings, self.tf_embed_context, prev_neurons_softmax = self.__create_tf_graph_embeddings_layer()
      self.tf_sm_weights, self.tf_sm_biases = self.__create_tf_graph_softmax_layer(prev_neurons_softmax)

      self.tf_loss = tf.reduce_mean(
          tf.nn.sampled_softmax_loss(weights = self.tf_sm_weights,
                         biases = self.tf_sm_biases,
                         labels = self.tf_train_labels,
                         inputs = self.tf_embed_context,
                         num_true = self.num_true,
                         num_sampled = self.FRAMEWORK["NUM_SAMPLED"],
                         num_classes = self.nr_products), name='loss')
      
      self.learning_rate = tf.placeholder(tf.float32, shape = [])
      optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
      self.train_step = optimizer.minimize(self.tf_loss)
      self.init = tf.global_variables_initializer()

    self._log("  Done initializing Tensorflow graph.", show_time = True)
    return


  def __create_tf_graph_placeholders(self):
    shape_tf_train_prod_inputs = []
    shape_tf_train_user_inputs = []
    shape_tf_train_labels = []
    self.num_true = 0
    tf_train_user_inputs = None

    if self.ARCHITECTURE.upper() == "SKIP-GRAM":
      shape_tf_train_prod_inputs = [None, 1]
      shape_tf_train_user_inputs = [None, 1]
      shape_tf_train_labels = [None, 1]
      self.num_true = 1

    tf_train_prod_inputs = tf.placeholder(tf.int32,
                                          shape=shape_tf_train_prod_inputs,
                                          name='train_prod_inputs')

    tf_train_labels = tf.placeholder(tf.int32,
                                     shape=shape_tf_train_labels,
                                     name='train_labels')

    if self.compute_users_embeddings:
      tf_train_user_inputs = tf.placeholder(tf.int32,
                                            shape=shape_tf_train_user_inputs,
                                            name='train_user_inputs')

    return tf_train_prod_inputs, tf_train_user_inputs, tf_train_labels



  def __create_tf_graph_embeddings_layer(self):
    tf_prod_embeddings = None
    tf_user_embeddings = None
    tf_prod_embed_lookup = None
    tf_user_embed_lookup = None
    tf_embed_context = None
    prev_neurons_softmax = 0

    if self.model_cache != None:
      tf_prod_embeddings = tf.Variable(self.model_cache[self.str_products_embeddings].value,
                                       name='prod_embeddings')
    else:
      tf_prod_embeddings = tf.Variable(tf.random_uniform(shape=[self.nr_products,
                                                                self.dict_nr_embeddings[self.str_products_embeddings]],
                                                         minval=-1.0,
                                                         maxval=1.0),
                                       name='prod_embeddings')
    #endif

    tf_prod_embed_lookup = tf.nn.embedding_lookup(tf_prod_embeddings,
                                                  self.tf_train_prod_inputs,
                                                  name='prod_embed_lookup')

    tf_embed_context = tf_prod_embed_lookup
    prev_neurons_softmax = self.dict_nr_embeddings[self.str_products_embeddings]

    if self.compute_users_embeddings:
      if self.model_cache != None:
        tf_user_embeddings = tf.Variable(self.model_cache[self.str_users_embeddings].value,
                                         name='user_embeddings')
      else:
        tf_user_embeddings = tf.Variable(tf.random_uniform(shape=[self.nr_users,
                                                                  self.dict_nr_embeddings[self.str_users_embeddings]],
                                                           minval=-1.0,
                                                           maxval=1.0),
                                         name='user_embeddings')
      #endif
      
      tf_user_embed_lookup = tf.nn.embedding_lookup(tf_user_embeddings,
                                                    self.tf_train_user_inputs,
                                                    name='user_embed_lookup')

      tf_embed_context = tf.concat([tf_embed_context, tf_user_embed_lookup], axis=-1)
      prev_neurons_softmax += self.dict_nr_embeddings[self.str_users_embeddings]
    #endif

    if self.ARCHITECTURE == "SKIP-GRAM":
      tf_embed_context = tf.reshape(tf_embed_context, [-1, prev_neurons_softmax], name = 'embed_context')

    return tf_prod_embeddings, tf_user_embeddings, tf_embed_context, prev_neurons_softmax


  def __create_tf_graph_softmax_layer(self, prev_neurons_softmax):
    tf_sm_weights = None
    tf_sm_biases  = None
    if self.model_cache != None:
      tf_sm_weights = tf.Variable(self.model_cache["sm_weights"].value)
      tf_sm_biases  = tf.Variable(self.model_cache["sm_biases"].value)
    else:
      tf_sm_weights = tf.Variable(tf.truncated_normal([self.nr_products, prev_neurons_softmax], 
                                  stddev=1.0 / math.sqrt(prev_neurons_softmax)),
                                  name='sm_weights')
      tf_sm_biases  = tf.Variable(tf.zeros([self.nr_products]), name='sm_biases')

    return tf_sm_weights, tf_sm_biases



  def Fit(self, X_train, y_train, epochs=5, batch_size=128, learning_rate=0.001):
    if self.compute_users_embeddings and (not (X_train.shape[1] == 2)):
      raise Exception("ERROR! X_train should have 2 columns. One for product labels and one for user labels!")

    self.X_train = X_train
    self.y_train = y_train

    self._log("Start training the model during {} epochs. "\
              "Batch_size={}, learning_rate={} ...".format(epochs,
              batch_size, learning_rate))

    if self.FRAMEWORK["NAME"].upper() == "TENSORFLOW":
      self._fit_tf_model(epochs, batch_size, learning_rate)
    elif self.FRAMEWORK["NAME"].upper() == "KERAS":
      pass
    return

  def _fit_tf_model(self, epochs, batch_size, learning_rate):
    self.model_name = self.file_prefix + '_' + self.FRAMEWORK["NAME"] +\
      '_' + self.ARCHITECTURE.upper() + '_Sampled_' + str(self.FRAMEWORK["NUM_SAMPLED"]) +\
      '_ProdEmb_' + str(self.dict_nr_embeddings[self.str_products_embeddings]) + '_Batch_' +\
      str(batch_size) + self.CONFIG["AUXILIARY_NAME"]

    if self.compute_users_embeddings:
      self.model_name += 'UserEmb' + '_' + str(self.dict_nr_embeddings[self.str_users_embeddings])

    self._log("  Model Name: {}".format(self.model_name))

    session = tf.Session(graph=self.tf_graph)
    session.run(self.init)

    nr_batches = self.X_train.shape[0] // batch_size
    self._log("  Every epoch there are processed {:,} batches.".format(nr_batches))

    start_train = time()
    
    for epoch in range(epochs):
      self._log('  Epoch {} ...'.format(epoch + 1))
      start_epoch = time()
      average_loss = 0
      t = trange(nr_batches, desc='', leave=True)
      for step in t:
        if step == nr_batches - 1:
          batch_prod_inputs = self.X_train[(step * batch_size) : self.X_train.shape[0], 0]
          batch_labels = self.y_train[(step * batch_size) : self.X_train.shape[0]]
          if self.compute_users_embeddings:
            batch_user_inputs = self.X_train[(step * batch_size) : self.X_train.shape[0], 1]
        else:
          batch_prod_inputs = self.X_train[(step * batch_size) : ((step + 1) * batch_size), 0]
          batch_labels = self.y_train[(step * batch_size) : ((step + 1) * batch_size)]
          if self.compute_users_embeddings:
            batch_user_inputs = self.X_train[(step * batch_size) : ((step + 1) * batch_size), 1]

        batch_prod_inputs = batch_prod_inputs.reshape(-1, 1)
        feed_dict = {self.tf_train_prod_inputs: batch_prod_inputs,
                     self.tf_train_labels: batch_labels,
                     self.learning_rate: learning_rate}
        if self.compute_users_embeddings:
          batch_user_inputs = batch_user_inputs.reshape(-1, 1)
          feed_dict[self.tf_train_user_inputs] = batch_user_inputs
        _, loss_val = session.run([self.train_step, self.tf_loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        t.set_description("Loss {:.3f}".format(loss_val))
        t.refresh()

        if step % 20000 == 0:
          if step > 0:
            average_loss /= 20000
          # The average loss is an estimate of the loss over the last 20000 batches.
          #self._log('    Computed cost at step {}: {:.2f}'.format(step, average_loss))
          average_loss = 0
        ### endif
      ### endfor - step

      self._model_weights[self.str_products_embeddings] = self.tf_prod_embeddings.eval(session=session)
      self._model_weights["sm_biases"] = self.tf_sm_biases.eval(session = session)
      self._model_weights["sm_weights"] = self.tf_sm_weights.eval(session = session)
      if self.compute_users_embeddings:
        self._model_weights[self.str_users_embeddings] = self.tf_user_embeddings.eval(session=session)


      if (epoch + 1) % 5 == 0:
        saved_model_filename = self.model_name + '_Ep_' + str(epoch+1).zfill(2)
        self._save_tensorflow_model(saved_model_filename)

      self.dict_embeddings[self.str_products_embeddings] = self._model_weights[self.str_products_embeddings]
      if self.compute_users_embeddings:
        self.dict_embeddings[self.str_users_embeddings] = self._model_weights[self.str_users_embeddings]

      end_epoch_min, end_epoch_seconds = divmod(time() - start_epoch, 60)
      self._log('  Epoch {} finished in {}m{:.2f}s.'.format(epoch + 1,
                int(end_epoch_min), end_epoch_seconds))
    ### endfor - epoch

    end_train_min, end_train_seconds = divmod(time() - start_train, 60)
    end_train_h, end_train_min = divmod(end_train_min, 60)
    self._log('Training finished in {}h{}m{:.2f}s.'.format(int(end_train_h),
              int(end_train_min), end_train_seconds))

    return
  

def GetGoogleDrivePath():
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
    raise Exception("ERROR! GoogleDrive not found in your computer!")
    
  return drive_path
  
if __name__ == '__main__':
  ### Data fetch and exploration
  base_folder = os.path.join(GetGoogleDrivePath(), '_hyperloop_data/recom_compl_2014_2017/_data')
  tran_folder = "all"
  app_folder = os.path.join(base_folder, tran_folder)


  trans_dataset_skip = np.load(os.path.join(app_folder, 'trans_skip.npz'))
  print('Loading training dataset ...')
  start = time()
  train = trans_dataset_skip['train']
  X = train[:, :2].reshape(-1, 2)
  y = train[:, 2].reshape(-1, 1)
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))
  
  
  prods_filename = os.path.join(app_folder, 'ITEMS.csv')
  print('Loading products dataset: {} ... '.format(prods_filename[-30:]))
  start = time()
  df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))


  users_filename = os.path.join(app_folder, 'USERS.csv')
  print('Loading users dataset: {} ... '.format(users_filename[-30:]))
  start = time()
  df_users = pd.read_csv(users_filename)
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))


  newids = np.array(df_prods['IDE'].tolist()) - 1
  newids = list(newids)
  ids = df_prods['ITEM_ID'].tolist()
  names = df_prods['ITEM_NAME'].tolist()
  id2new_id = dict(zip(ids, newids))
  new_id2prod = dict(zip(newids, names))

  architecture = 'SKIP-GRAM'
  r = EmbeddingsMaster(nr_products=len(id2new_id), nr_users=len(df_users))
  r.Fit(X_train=X, y_train=y, epochs=25, batch_size=512)
  
  
  
  