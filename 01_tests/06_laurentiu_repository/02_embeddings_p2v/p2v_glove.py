# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:08:53 2017

@author: Laurentiu Piciu
"""

import os
import tensorflow as tf
import h5py
import numpy as np
from time import time
import pandas as pd
from scipy.sparse import load_npz

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


valid_model_weights = ["focal_embeddings", "context_embeddings", "focal_bias", "context_bias"]

class RecomP2VGlove:
  def __init__(self, nr_products, config_file = 'tf_glove_config.txt',
               nr_embeddings = 64, context_window = 2, cooccurrence_cutoff = 100,
               alpha = 3/4):
    
    self.CONFIG = None
    logger_module = load_module('logger','logger.py')
    self.logger = logger_module.Logger(lib_name = "RECOMv3",
                                config_file = config_file,
                                TF_KERAS = True)
    self._log("Initializing RecomP2VGlove model ...")
    self.CONFIG = self.logger.config_data
    self.nr_products = nr_products
    self.nr_embeddings = nr_embeddings
    self.context_window = context_window
    self.cooccurrence_cutoff = cooccurrence_cutoff
    self.alpha = alpha
    self.file_prefix = self.logger.file_prefix
    self.model_name = None
    self._base_folder = self.logger.GetBaseFolder()
    self._check_config_errors()
    
    self._log("  Loading MCO from disk ...")
    self.MCO = load_npz(os.path.join(self._base_folder, self.CONFIG["MCO_CACHE"]))
    self._log("  Loaded MCO.", show_time = True)
    
    self.tf_graph = None
    self.model_cache = None
    self.Embeddings = None
    self.NormEmbeddings = None
    self.ModelWeights = dict()
    self._init_tensorflow_model()
    
    self._log("Initialized RecomP2VGlove model: nr_prod={}, nr_embeddings={}, "\
              "context_window={}, coocc_cutoff={}, alpha={:.2f}".format(nr_products,
              nr_embeddings,
              context_window,
              cooccurrence_cutoff,
              alpha))


  def _check_config_errors(self):
    mco_file = self.CONFIG["MCO_CACHE"]
    
    if mco_file == "":
      err_msg = "ERROR! Unimplemented functionality to create MCO at runtime."
      self._log(err_msg)
      raise Exception(err_msg)
      
    if not os.path.exists(os.path.join(self._base_folder, mco_file)):
      err_msg = "ERROR! MCO_CACHE does not exist."
      self._log(err_msg)
      raise Exception(err_msg)
    
    if self.CONFIG["LOAD_MODEL"] != "":
      if not os.path.exists(os.path.join(self._base_folder, self.CONFIG["LOAD_MODEL"])):
        err_msg = "ERROR! Specified model to load does not exist."
        self._log(err_msg)
        raise Exception(err_msg)

    return
  

  def _log(self, str_msg, results = False, show_time = False):
    self.logger.VerboseLog(str_msg, results, show_time)
    return


  def _init_tensorflow_model(self):
    self._log("  Using pure Tensorflow ...")
    
    if self.CONFIG["LOAD_MODEL"] != "":
      restore_point = os.path.join(self._base_folder, self.CONFIG["LOAD_MODEL"])
      self.model_cache = self._load_tensorflow_model(restore_point)
      self.Embeddings = self.model_cache["focal_embeddings"].value +\
                        self.model_cache["context_embeddings"].value

    self._create_tensorflow_graph()
    return

    
  def __check_loaded_model_errors(self, model_cache):
    keys = list(model_cache.keys())

    for w in valid_model_weights:
      if w not in keys:
        err_msg = "ERROR! Weights [{}] not found in loaded model."
        self._log(err_msg)
        raise Exception(err_msg)
    
    return
  
  def _load_tensorflow_model(self, restore_point):
    self._log("  Restoring model weights: {} ...".format(restore_point))
    model_cache = h5py.File(restore_point, 'r')
    self.__check_loaded_model_errors(model_cache)
    self._log("  Done restoring model weights: {}.".format(list(model_cache.keys())),
              show_time = True)
    return model_cache
  

  def _save_tensorflow_model(self, saved_model_filename):
    self._log("  Saving model weights ...")
    model = h5py.File(os.path.join(self.logger._outp_dir, saved_model_filename + '.h5'), 'w')
    for key, value in self.ModelWeights.items():
      model.create_dataset(key, data = value)

    model.close()
    self._log("  Model {} saved.".format(saved_model_filename), show_time = True)
    return
  
  def _create_tensorflow_graph(self):
    self._log("  Initializing Tensorflow graph ...")
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.tf_focal_input, self.tf_context_input,\
        self.tf_cooccurrence_score = self.__create_tf_graph_placeholders()

      self.tf_focal_embeddings, self.tf_context_embeddings,\
        self.tf_focal_bias, self.tf_context_bias = self.__create_tf_graph_embeddings_layer()

      self.tf_boosted_embeddings = tf.add(self.tf_focal_embeddings, self.tf_context_embeddings,
                                          name = "boosted_embeddings")

      tf_focal_embed = tf.nn.embedding_lookup(self.tf_focal_embeddings, self.tf_focal_input)
      tf_context_embed = tf.nn.embedding_lookup(self.tf_context_embeddings, self.tf_context_input)
      tf_focal_b = tf.nn.embedding_lookup(self.tf_focal_bias, self.tf_focal_input)
      tf_context_b = tf.nn.embedding_lookup(self.tf_context_bias, self.tf_context_input)
  
      tf_cutoff = tf.constant(self.cooccurrence_cutoff, dtype = tf.float32, name = 'coocc_cutoff')
      tf_alpha = tf.constant(self.alpha, dtype = tf.float32, name = 'alpha')
      tf_weighting_factor = tf.minimum(
          1.0,
          tf.pow(tf.div(self.tf_cooccurrence_score, tf_cutoff),
                 tf_alpha))

      tf_embedding_product = tf.reduce_sum(tf.multiply(tf_focal_embed, tf_context_embed), axis = 1)

      tf_log_cooccurrence_score = tf.log(tf.to_float(self.tf_cooccurrence_score))

      tf_least_squares_regression = tf.square(tf.add_n([
          tf_embedding_product,
          tf_focal_b,
          tf_context_b,
          tf.negative(tf_log_cooccurrence_score)]))

      self.tf_loss = tf.reduce_sum(tf.multiply(tf_weighting_factor,
                                               tf_least_squares_regression), name = "loss")
      self.learning_rate = tf.placeholder(tf.float32, shape = [])
      optimizer = tf.train.AdagradOptimizer(self.learning_rate)
      self.train_step = optimizer.minimize(self.tf_loss)
      self.init = tf.global_variables_initializer()
    
    self._log("  Done initializing Tensorflow graph.", show_time = True)
    return
  
  def Fit(self, epochs = 5, batch_size = 128, learning_rate = 0.05):
    self._log("Start training the model during {} epochs. "\
              "Batch_size={}, learning_rate={:.2f} ...".format(epochs,
              batch_size, learning_rate))
    
    self.model_name = self.file_prefix + '_GloVe' + '_Emb_' + str(self.nr_embeddings) +\
      '_Window_' + str(self.context_window * 2) + '_Batch_' + str(batch_size) +\
      self.CONFIG["AUXILIARY_NAME"]

    self._log("  Model Name: {}".format(self.model_name))
    
    self.X_train = None
    self._prepare_training_inputs()
    
    session = tf.Session(graph = self.tf_graph)
    session.run(self.init)
    
    nr_batches = self.X_train.shape[1] // batch_size
    self._log("  Every epoch there are processed {:,} batches.".format(nr_batches))
    start_train = time()
    for epoch in range(epochs):
      self._log('  Epoch {} ...'.format(epoch + 1))
      start_epoch = time()
      np.random.shuffle(self.X_train.T)
      average_loss = 0
      for step in range(nr_batches):
        if step == nr_batches - 1:
          inputs = self.X_train[:, (step * batch_size) : self.X_train.shape[1]]
        else:
          inputs = self.X_train[:, (step * batch_size) : ((step + 1) * batch_size)]
        
        feed_dict = {
            self.tf_focal_input: inputs[0, :].astype(np.int32),
            self.tf_context_input: inputs[1, :].astype(np.int32),
            self.tf_cooccurrence_score: inputs[2, :],
            self.learning_rate: learning_rate}
        
        _, loss_val = session.run([self.train_step, self.tf_loss], feed_dict = feed_dict)
        average_loss += loss_val
        
        if step % 20000 == 0:
          if step > 0:
            average_loss /= 20000
          # The average loss is an estimate of the loss over the last 20000 batches.
          self._log('    Computed cost at step {}: {:.2f}'.format(step, average_loss))
          average_loss = 0
        ### endif
      ### endfor - step
      
      self.ModelWeights["focal_embeddings"] = self.tf_focal_embeddings.eval(session = session)
      self.ModelWeights["context_embeddings"] = self.tf_context_embeddings.eval(session = session)
      self.ModelWeights["focal_bias"] = self.tf_focal_bias.eval(session = session)
      self.ModelWeights["context_bias"] = self.tf_context_bias.eval(session = session)
      
      if (epoch + 1) % 10 == 0:
        saved_model_filename = self.model_name + '_Ep_' + str(epoch + 1).zfill(2)
        self._save_tensorflow_model(saved_model_filename)
      
      end_epoch_min, end_epoch_seconds = divmod(time() - start_epoch, 60)
      self._log('  Epoch {} finished in {}m{:.2f}s.'.format(epoch + 1,
                int(end_epoch_min), end_epoch_seconds))
    ### endfor - epoch
    
    self.Embeddings = self.tf_boosted_embeddings.eval(session = session)
    end_train_min, end_train_seconds = divmod(time() - start_train, 60)
    end_train_h, end_train_min = divmod(end_train_min, 60)
    self._log('Training finished in {}h{}m{:.2f}s.'.format(int(end_train_h),
              int(end_train_min), end_train_seconds))
    
    return
  
  
  def _prepare_training_inputs(self):
    self._log('  Preparing training inputs ...')
    i_indices, j_indices = np.nonzero(self.MCO)
    scores = np.array(self.MCO[i_indices, j_indices]).flatten()
    self.X_train = np.array([i_indices, j_indices, scores])
    self._log('  Finished preparing {:,} training inputs.'.format(self.X_train.shape[1]),
              show_time = True)

    return


  def __create_tf_graph_placeholders(self):
    tf_focal_input = tf.placeholder(tf.int32, shape = [None], name = "focal_products")
    tf_context_input = tf.placeholder(tf.int32, shape = [None], name = "context_products")
    tf_cooccurrence_score = tf.placeholder(tf.float32, shape = [None], name = "cooccurrence_score")
    
    return tf_focal_input, tf_context_input, tf_cooccurrence_score
  
  def __create_tf_graph_embeddings_layer(self):
    focal_embeddings = None
    context_embeddings = None
    focal_bias = None
    context_bias = None
    n_prod = self.nr_products + 1
    n_emb  = self.nr_embeddings
    
    if self.model_cache != None:
      focal_embeddings = tf.Variable(self.model_cache["focal_embeddings"].value,
                                     name = "focal_embeddings")
      context_embeddings = tf.Variable(self.model_cache["context_embeddings"].value,
                                       name = "context_embeddings")
      focal_bias = tf.Variable(self.model_cache["focal_bias"].value,
                               name = "focal_bias")
      context_bias = tf.Variable(self.model_cache["context_bias"].value,
                                 name = "context_bias")
      
    else:
      focal_embeddings = tf.Variable(tf.random_uniform([n_prod, n_emb], 1.0, -1.0),
                                     name = "focal_embeddings")
      context_embeddings = tf.Variable(tf.random_uniform([n_prod, n_emb], 1.0, -1.0),
                                       name = "context_embeddings")
      focal_bias = tf.Variable(initial_value = np.zeros(shape = (n_prod)), dtype = tf.float32,
                               name = "focal_bias")
      context_bias = tf.Variable(initial_value = np.zeros(shape = (n_prod)), dtype = tf.float32,
                                 name = "context_bias")

    return focal_embeddings, context_embeddings, focal_bias, context_bias


  def NormalizeEmbeddings(self):
    from sklearn.preprocessing import normalize
    self._log("Precomputing L2-norms of prod feature vectors ...")
    self.NormEmbeddings = normalize(self.Embeddings) #l2 normalization
    self._log("Finished precomputing L2-norms of prod feature vectors.", show_time = True)

  def GetKMeansClusters(self, n_clusters = 25, norm_embeddings = False):
    from sklearn.cluster import KMeans
    if norm_embeddings and (self.NormEmbeddings is None):
      err_msg = "ERROR! Please compute normalized prod feature vectors before."
      self._log(err_msg)
      raise Exception(err_msg)
    
    self._log("Creating {} KMeans clusters for computed embeddings ...".format(n_clusters))
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
    if norm_embeddings:
      y_kmeans = kmeans.fit_predict(self.NormEmbeddings[1:, :])
    else:
      y_kmeans = kmeans.fit_predict(self.Embeddings[1:, :])
    self._log("Finished computing KMeans clusters.", show_time = True)
    return y_kmeans
  
  def GetEmbeddings(self, norm_embeddings = False):
    if norm_embeddings:
      return self.NormEmbeddings[1:, :]
    
    return self.Embeddings[1:, :]


####################################################################
####################################################################
####################################################################




if __name__ == '__main__':
  ### Data fetch and exploration
  base_folder = "D:/Google Drive/_hyperloop_data/recom_compl_2014_2017/_data"
  tran_folder = "summer"
  app_folder = os.path.join(base_folder, tran_folder)
  
  prods_filename = os.path.join(app_folder, 'ITEMS.csv')
  print('Loading products dataset: {} ... '.format(prods_filename[-30:]))
  start = time()
  df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))
  
  newids = np.array(df_prods['IDE'].tolist())
  newids = list(newids)
  ids = df_prods['ITEM_ID'].tolist()
  names = df_prods['ITEM_NAME'].tolist()
  id2new_id = dict(zip(ids, newids))
  new_id2prod = dict(zip(newids, names))


  r = RecomP2VGlove(nr_products = df_prods.shape[0], cooccurrence_cutoff = 1000)
  r.Fit(epochs = 150, batch_size = 256)
  
  from recom_maps_utils import ProcessModel
  lowest_id = min(newids)

  dict_model_results1 = ProcessModel(r, new_id2prod,
                                     tsne_nr_products = None,
                                     compute_norm_embeddings = True,
                                     do_tsne_3D = False,
                                     lowest_id = lowest_id)

  dict_model_results2 = ProcessModel(r, new_id2prod,
                                     tsne_nr_products = None,
                                     compute_norm_embeddings = False,
                                     do_tsne_3D = False,
                                     lowest_id = lowest_id)
