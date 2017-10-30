import numpy as np
from sklearn.model_selection import train_test_split
import os
import platform
from importlib.machinery import SourceFileLoader
from tf_engine_cnn import TFConvolutionalNN
import pandas as pd

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "Google Drive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "base.py")).load_module()
  logger = logger_lib.Logger(lib='CNN')

  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)

  X = mnist.data
  y = mnist.target

  X = min_max_scaler(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)
  X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                test_size=0.5,
                                                                random_state=42)

  return dotdict({'train': (X_train, y_train),
                  'test': (X_test, y_test),
                  'validation': (X_validation, y_validation)}), logger

if __name__ == "__main__":
  data_sets, logger = fetch_data()
  hyper_parameters = {'learning_rate': 0.001, 'momentum_speed': 0.9, 'epochs': 5, 'batch_size': 10, 'drop_keep': .7}
  hyper_parameters = dotdict(hyper_parameters)

  X_train, y_train = data_sets.train
  X_validation, y_validation = data_sets.validation
  X_test, y_test = data_sets.test
  """
  X_test_84 = np.load('test_84x84.npy') # sunt si normalizate
  X_test_100 = np.load('test_100x100.npy')  # sunt si normalizate
  X_test_50 = np.load('test_50x50.npy')  # sunt si normalizate
  X_test_14 = np.load('test_14x14.npy')  # sunt si normalizate
  
  TEST = dotdict({'X': [(X_test, 28, 28), (X_test_84, 84, 84), 
                        (X_test_100, 100, 100), (X_test_50, 50, 50),
                        (X_test_14, 14, 14)],
                  'y': y_test})
  """
    
  
  architectures = []  
    
  architecture = dotdict({
    'num_layers': 2,
    'num_filters': [64,256],
    'H': 28,
    'W': 28,
    'C': 1,
    'filter_size': 3,
    'pool_size': 2,
    'fc_units': [],
    'fc_act': [],
    'classes': 10
  })
  architectures.append(architecture)
    
  architecture = dotdict({
    'num_layers': 3,
    'num_filters': [32,64,512],
    'H': 28,
    'W': 28,
    'C': 1,
    'filter_size': 3,
    'pool_size': 2,
    'fc_units': [],
    'fc_act': [],
    'classes': 10
  })
  architectures.append(architecture)
    
  
  for cnn in architectures:
    nn = TFConvolutionalNN(cnn, 'models/test', logger)
    
    nn.train(X_train=X_train,
             y_train=y_train,
             X_validation=X_validation,
             y_validation=y_validation,
             epochs=hyper_parameters.epochs,
             batch_size=hyper_parameters.batch_size,
             learning_rate=hyper_parameters.learning_rate,
             drop_keep=hyper_parameters.drop_keep)
  
      

  #y_pred_mnist = nn.predict(X_test, y_test, 28, 28)
  #y_pred_mnist = np.array(y_pred_mnist).flatten()