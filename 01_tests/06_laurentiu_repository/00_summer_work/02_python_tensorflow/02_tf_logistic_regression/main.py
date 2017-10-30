import os
import numpy as np
import platform
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from tf_logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
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
  logger = logger_lib.Logger(lib='LOGREG')

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
  VERBOSITY = 10
  model_name = 'model_cache'


  lr = LogisticRegression(model_name, logger, VERBOSITY)
  lr.train(data_sets.train[0], data_sets.train[1],
           data_sets.validation[0], data_sets.validation[1],
           epochs=10,
           batch_size=10,
           learning_rate=0.001,
           beta=0.0005,
           momentum_speed=0.85,
           decay_factor=0.65)

  #X_test, y_test = data_sets.test
  #yhat = lr.predict(X_test,y_test)

