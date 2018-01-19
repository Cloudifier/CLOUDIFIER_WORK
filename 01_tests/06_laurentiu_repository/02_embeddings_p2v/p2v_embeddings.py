"""
Created on Fri Nov  17 16:00:32 2017

@author: Laurentiu Piciu

@description: Recommender System using Embeddings Class

@modifid:
  2017-11-17 Created
  2017-11-19 Added functionality to use both Tensorflow and Keras
  2017-11-20 KMeans function
  2017-12-21 Skip-gram functionality
"""

import os
import tensorflow as tf
import math
import h5py
import numpy as np
from time import time

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
valid_embed_architectures = ["CBOW", "SKIP-GRAM"]
valid_model_weights = ["embeddings", "sm_weights", "sm_biases"]

class RecomP2VEmbeddings:
  def __init__(self, nr_products, config_file = 'tf_config.txt',
               nr_embeddings = 64, context_window = 3, architecture = 'CBOW'):
    """
    Constructor - initialize all props of Recommender System using Embeddings
    
    'quick' methods:
      Fit(X_train, y_train, epochs, batch_size, learning_rate)
      
      NormalizeEmbeddings()
      GetNormEmbeddings()
      GetEmbeddings()
      
      CreateKMeansClusters(n_clusters)
      GetNormKMeansClusters()
      GetKMeansClusters()
    """
    self.CONFIG = None
    logger_module = load_module('logger','logger.py')
    self.logger = logger_module.Logger(lib_name = "RECOMv3",
                                config_file = config_file,
                                TF_KERAS = True)
    self._log("Initializing RecomP2VEmbeddings model ...")
    
    self.CONFIG = self.logger.config_data
    self.ARCHITECTURE = architecture
    self.nr_products = nr_products
    self.nr_embeddings = nr_embeddings
    self.context_window = context_window
    self.file_prefix = self.logger.file_prefix
    self.model_name = None
    self._base_folder = self.logger.GetBaseFolder()
    self.FRAMEWORK = self.CONFIG["FRAMEWORK"]
    self.Embeddings = None
    self.NormEmbeddings = None
    self._check_config_errors()
    
    if self.FRAMEWORK["NAME"].upper() == "TENSORFLOW":
      self.tf_graph = None
      self.model_cache = None
      self._cfg_embeddings_context = self.FRAMEWORK["EMBEDDINGS_CONTEXT"]
      self._model_weights = dict()

      self._init_tensorflow_model()
      # self._describe_tensorflow_model() # Not implemented
    elif self.FRAMEWORK["NAME"].upper() == "KERAS":
      self.keras_model = None
      
      self._init_keras_model()
      self.logger.LogKerasModel(self.keras_model)
    
    self._log("Initialized recommender model: nr_prod={}, nr_embeddings={}, "\
              "context_window={}, architecture={}".format(nr_products,
              nr_embeddings,
              context_window,
              self.ARCHITECTURE.upper()))


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
      
      if self.ARCHITECTURE.upper() not in self.CONFIG["LOAD_MODEL"].upper():
        err_msg = 'ERROR! architecture does not match with loaded model'
        self._log(err_msg)
        raise Exception(err_msg)

    if (self.FRAMEWORK["NAME"].upper() == "TENSORFLOW") and (self.ARCHITECTURE.upper() == "CBOW"):
      if self.FRAMEWORK["EMBEDDINGS_CONTEXT"].upper() not in valid_embed_contexts:
        err_msg = 'ERROR! Unknown config_data["EMBEDDINGS_CONTEXT"]'
        self._log(err_msg)
        raise Exception(err_msg)
  
      if self.CONFIG["LOAD_MODEL"] != "":
        if self.FRAMEWORK["EMBEDDINGS_CONTEXT"].upper() not in self.CONFIG["LOAD_MODEL"].upper():
          err_msg = 'ERROR! embeddings_context does not match with loaded model'
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
      self.Embeddings = self.model_cache['embeddings'].value

    self._create_tensorflow_graph()


  def _init_keras_model(self):
    """
    Processes keras_config and creates the model
    """

    self._log("Using Keras ...")
    
    if self.CONFIG["LOAD_MODEL"] == "":
      # Parses the architecture given in the configuration file
      LAYERS = self.FRAMEWORK["LAYERS"]
      inputs = None
      current_layer = None
      outputs = None
      
      for layer in LAYERS:
        layer_type = layer[0].upper()
        
        if layer_type == "INPUT":
          inputs = tf.keras.layers.Input(shape = (2 * self.context_window, ))
          current_layer = inputs
        
        if layer_type == "CONV1D":
          params = dotdict(layer[1])
          act = params.activation
          nrf = params.filters
          ks  = params.kernel_size
          current_layer = tf.keras.layers.Conv1D(
                  filters = nrf if nrf != -1 else self.nr_products,
                  kernel_size = ks if ks != -1 else 2 * self.context_window,
                  strides = params.strides,
                  activation = act if act != '' else None)(current_layer)
        
        if layer_type == "FLATTEN":
          current_layer = tf.keras.layers.Flatten()(current_layer)
        
        if layer_type == "DENSE":
          params = dotdict(layer[1])
          act = params.activation
          nru = params.nr_units
          current_layer = tf.keras.layers.Dense(
                  units = nru if nru != -1 else self.nr_products,
                  activation = act if act != '' else None)(current_layer)
        
        if layer_type == "EMBEDDING":
          current_layer = tf.keras.layers.Embedding(input_dim = self.nr_products,
                                    output_dim = self.nr_embeddings)(current_layer)
      #endfor
      
      outputs = current_layer
      self.keras_model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
      self.keras_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam')
    else:
      # Restores the model from the '.h5' file specified in CONFIG["LOAD_MODEL"]
      restore_point = os.path.join(self._base_folder, self.CONFIG["LOAD_MODEL"])
      self.keras_model = self._load_keras_model(restore_point)
      self.Embeddings = self.keras_model.get_weights()[0]
  

  def __check_loaded_model_errors(self, model_cache):
    keys = list(model_cache.keys())

    for w in valid_model_weights:
      if w not in keys:
        err_msg = "ERROR! Weights [{}] not found in loaded model."
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
  
  def _load_keras_model(self, restore_point):
    """
    TODO
    """
    self._log('  Restoring keras model: {}'.format(restore_point))
    model_cache = tf.keras.models.load_model(restore_point)
    self._log('  Done restoring keras model.', show_time = True)
    return model_cache
  
  def _save_keras_model(self, saved_model_filename):
    """
    TODO
    """
    self._log("  Saving keras model ...")
    self.keras_model.save(saved_model_filename)
    self._log('  Model {} saved.'.format(saved_model_filename), show_time = True)
    return


  def _create_tensorflow_graph(self):
    """
    TODO
    """

    self._log("  Initializing Tensorflow graph ...")
    self.tf_graph = tf.Graph()    
    with self.tf_graph.as_default():
      self.tf_train_inputs, self.tf_train_labels = self.__create_tf_graph_placeholders()
      self.tf_embeddings, self.tf_embed_context, prev_neurons_softmax = self.__create_tf_graph_embeddings_layer()
      self.tf_sm_weights, self.tf_sm_biases = self.__create_tf_graph_softmax_layer(prev_neurons_softmax)

      self.tf_loss = tf.reduce_mean(
          tf.nn.sampled_softmax_loss(weights = self.tf_sm_weights,
                         biases = self.tf_sm_biases,
                         labels = self.tf_train_labels,
                         inputs = self.tf_embed_context,
                         num_true = self.num_true,
                         num_sampled = self.FRAMEWORK["NUM_SAMPLED"],
                         num_classes = self.nr_products), name = 'loss')
      
      self.learning_rate = tf.placeholder(tf.float32, shape = [])
      optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.9)
      self.train_step = optimizer.minimize(self.tf_loss)
      self.init = tf.global_variables_initializer()

    self._log("  Done initializing Tensorflow graph.", show_time = True)
    return
  
  def Fit(self, X_train, y_train, epochs = 5, batch_size = 128, learning_rate = 0.1):
    self.X_train = X_train
    self.y_train = y_train
    
    self._log("Start training the model during {} epochs. "\
              "Batch_size={}, learning_rate={} ...".format(epochs,
              batch_size, learning_rate))
    
    if self.FRAMEWORK["NAME"].upper() == "TENSORFLOW":
      self._fit_tf_model(epochs, batch_size, learning_rate)
    elif self.FRAMEWORK["NAME"].upper() == "KERAS":
      self._fit_keras_model(epochs, batch_size)
    return
  
  def _fit_tf_model(self, epochs, batch_size, learning_rate):
    self.model_name = self.file_prefix + '_' + self.FRAMEWORK["NAME"] +\
      '_' + self.ARCHITECTURE.upper() + '_Sampled_' + str(self.FRAMEWORK["NUM_SAMPLED"]) +\
      '_Emb_' + str(self.nr_embeddings) + '_Window_' + str(self.context_window * 2) + '_Batch_' +\
      str(batch_size)  + '_' + self._cfg_embeddings_context + self.CONFIG["AUXILIARY_NAME"]
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
      for step in range(nr_batches):
        if step == nr_batches - 1:
          batch_inputs = self.X_train[(step * batch_size) : self.X_train.shape[0], :]
          batch_labels = self.y_train[(step * batch_size) : self.X_train.shape[0]]
        else:
          batch_inputs = self.X_train[(step * batch_size) : ((step + 1) * batch_size), :]
          batch_labels = self.y_train[(step * batch_size) : ((step + 1) * batch_size)]
          
        feed_dict = {self.tf_train_inputs: batch_inputs,
                     self.tf_train_labels: batch_labels,
                     self.learning_rate: learning_rate}
        _, loss_val = session.run([self.train_step, self.tf_loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 20000 == 0:
          if step > 0:
            average_loss /= 20000
          # The average loss is an estimate of the loss over the last 20000 batches.
          self._log('    Computed cost at step {}: {:.2f}'.format(step, average_loss))
          average_loss = 0
        ### endif
      ### endfor - step
      
      self._model_weights["embeddings"] = self.tf_embeddings.eval(session = session)
      self._model_weights["sm_biases"] = self.tf_sm_biases.eval(session = session)
      self._model_weights["sm_weights"] = self.tf_sm_weights.eval(session = session)
      
      if (epoch + 1) % 5 == 0:
        saved_model_filename = self.model_name + '_Ep_' + str(epoch+1).zfill(2)
        self._save_tensorflow_model(saved_model_filename)
      
      self.Embeddings = self._model_weights["embeddings"]
      
      end_epoch_min, end_epoch_seconds = divmod(time() - start_epoch, 60)
      self._log('  Epoch {} finished in {}m{:.2f}s.'.format(epoch + 1,
                int(end_epoch_min), end_epoch_seconds))
    ### endfor - epoch
    
    end_train_min, end_train_seconds = divmod(time() - start_train, 60)
    end_train_h, end_train_min = divmod(end_train_min, 60)
    self._log('Training finished in {}h{}m{:.2f}s.'.format(int(end_train_h),
              int(end_train_min), end_train_seconds))
  
  
  def _fit_keras_model(self, epochs, batch_size):
    self.model_name = self.file_prefix + '_' + self.FRAMEWORK["NAME"] + '_Emb_' +\
      str(self.nr_embeddings) + '_Window_' + str(self.context_window * 2) + '_Batch_' +\
      str(batch_size)  + self.CONFIG["AUXILIARY_NAME"]
    self._log("  Model Name: {}".format(self.model_name))
    
    if self.keras_model.get_config()['layers'][-1]['class_name'] == 'Conv1D':
      self.y_train = self.y_train.reshape(-1, 1, 1)
    
    self.keras_model.fit(self.X_train,
                         self.y_train,
                         batch_size = batch_size,
                         epochs = epochs,
                         callbacks = [self.logger.GetKerasEpochCallback()])
    
    saved_model_filename = self.model_name + '_Ep_' + str(epochs).zfill(2)
    self._save_keras_model(saved_model_filename)
    self.Embeddings = self.keras_model.get_weights()[0]
    
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
      y_kmeans = kmeans.fit_predict(self.NormEmbeddings)
    else:
      y_kmeans = kmeans.fit_predict(self.Embeddings)
    self._log("Finished computing KMeans clusters.", show_time = True)
    return y_kmeans
  
  def GetEmbeddings(self, norm_embeddings = False):
    if norm_embeddings:
      if self.NormEmbeddings is None:
        err_msg = "ERROR! Please compute normalized prod feature vectors before."
        self._log(err_msg)
        raise Exception(err_msg)
      else:
        return self.NormEmbeddings

    return self.Embeddings

  def __create_tf_graph_placeholders(self):
    shape_tf_train_inputs = [None, self.context_window * 2]
    shape_tf_train_labels = [None, 1]
    self.num_true = 1
    
    if self.ARCHITECTURE.upper() == "SKIP-GRAM":
      shape_tf_train_inputs = [None, 1]
      shape_tf_train_labels = [None, self.context_window * 2]
      self.num_true = self.context_window * 2
    
    tf_train_inputs = tf.placeholder(tf.int32, shape = shape_tf_train_inputs, name = 'train_inputs')
    tf_train_labels = tf.placeholder(tf.int32, shape = shape_tf_train_labels, name = 'train_labels')

    return tf_train_inputs, tf_train_labels


  def __create_tf_graph_embeddings_layer(self):
    tf_embeddings = None
    tf_embed = None
    tf_embed_context = None
    prev_neurons_softmax = 0
    n_prod = self.nr_products
    n_emb  = self.nr_embeddings
    
    if self.model_cache != None:      
      tf_embeddings = tf.Variable(self.model_cache["embeddings"].value, name='embeddings')
    else:
      tf_embeddings = tf.Variable(tf.random_uniform([n_prod, n_emb], -1.0, 1.0), name='embeddings')

    tf_embed = tf.nn.embedding_lookup(tf_embeddings, self.tf_train_inputs, name = 'embed_lookup')

    if self.ARCHITECTURE == "CBOW":
      if self._cfg_embeddings_context.upper() == valid_embed_contexts[0]: # FLATTEN
        prev_neurons_softmax = self.nr_embeddings * self.context_window * 2
        tf_embed_context = tf.reshape(tf_embed, [-1, prev_neurons_softmax], name = 'embed_context')
      elif self._cfg_embeddings_context.upper() == valid_embed_contexts[1]: # MEAN
        prev_neurons_softmax = self.nr_embeddings
        tf_embed_context = tf.reduce_mean(tf_embed, 1, name = 'embed_context')
      elif self._cfg_embeddings_context.upper() == valid_embed_contexts[2]: # SUM
        prev_neurons_softmax = self.nr_embeddings
        tf_embed_context = tf.reduce_sum(tf_embed, 1, name = 'embed_context')
    
    if self.ARCHITECTURE == "SKIP-GRAM":
      prev_neurons_softmax = self.nr_embeddings
      tf_embed_context = tf.reshape(tf_embed, [-1, prev_neurons_softmax], name = 'embed_context')

    return tf_embeddings, tf_embed_context, prev_neurons_softmax
  
  
  def __create_tf_graph_softmax_layer(self, prev_neurons_softmax):
    tf_sm_weights = None
    tf_sm_biases  = None
    if self.model_cache != None:
      tf_sm_weights = tf.Variable(self.model_cache["sm_weights"].value)
      tf_sm_biases  = tf.Variable(self.model_cache["sm_biases"].value)
    else:
      tf_sm_weights = tf.Variable(tf.truncated_normal([self.nr_products, prev_neurons_softmax], 
                                  stddev = 1.0 / math.sqrt(prev_neurons_softmax)),
                                  name = 'sm_weights')
      tf_sm_biases  = tf.Variable(tf.zeros([self.nr_products]), name = 'sm_biases')

    return tf_sm_weights, tf_sm_biases

  
  def ComputeMTC(self): # MTC = Matrice Tranzitii intre Clustere
    pass


  def ComputeBestClustering(self):
    pass
















class ProdSym:
  """
  Class to contain products embeddings and 'vocabulary' after P2V training class and other methods
  which compute symilarities.
  """
  
  def __init__(self, id2new_id, new_id2prod, norm_embeddings, logger):
    self.id2new_id = id2new_id
    self.new_id2prod = new_id2prod
    self.norm_embeddings = norm_embeddings
    self.logger = logger

  def _prod_vec(self, prod):
    if prod in self.id2new_id:
      _result = self.norm_embeddings[self.id2new_id[prod]]
      return _result
    else:
      raise KeyError("Product {} not found".format(prod))

  def _normalize(self, vec, norm = 'l2'):
    if norm not in ('l1', 'l2'):
      raise ValueError("{} is not a supported norm. "\
                       "Currently supported norms are 'l1' and 'l2'.".format(norm))

    if norm == 'l1':
      _magnitude = np.sum(np.abs(vec))
    elif norm == 'l2':
      _magnitude = np.sqrt(np.sum(vec ** 2))
      
    if _magnitude > 0:
      return vec / _magnitude
    else:
      return vec

  def _efficient_argsort(self, x, topk = None, reverse = False):
    if topk is None:
      topk = x.size

    if reverse:
      x = -x

    if topk <= 0:
      return []

    if topk >= x.size:
      return np.argsort(x)
    
    most_extreme = np.argpartition(x, topk)[:topk]
    _result = most_extreme.take(np.argsort(x.take(most_extreme)))
    return _result

  def FindMostSimilar(self, positive = None, negative = None, topk = None):
    
    if (not positive) | (not negative):
      raise ValueError("Cannot compute similarity with no input!")
    
    positive = [(prod, 1.0) for prod in positive]
    negative = [(prod, -1.0) for prod in negative]
    prods_query, mean = set(), []
    for prod, weight in positive + negative:
      mean.append(weight * self._prod_vec(prod))
      prods_query.add(self.id2new_id[prod])

    mean = self._normalize(np.array(mean).mean(axis = 0), norm = 'l2')
    distances = np.dot(self.norm_embeddings, mean)
    if not topk:
      return distances
    
    best = self._efficient_argsort(distances, topk = topk + len(prods_query), reverse = True)
    _result = [(self.new_id2prod[i], float(distances[i])) for i in best if i not in prods_query]
     
    return _result

if __name__ == '__main__':
  print('Cannot run library module!')