import os
import platform
from importlib.machinery import SourceFileLoader
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tf_linear_regression import LinearRegression

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
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

def read_boston_data():
  boston = load_boston()
  features = np.array(boston.data)
  labels = np.array(boston.target)
  return features, labels

def min_max_normalization(data):
  min = np.min(data, axis=0)
  max = np.max(data, axis=0)
  return (data - min) / (max - min)

if __name__ == "__main__":
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")

  logger_lib = SourceFileLoader('logger', os.path.join(utils_path, 'base.py')).load_module()
  logger = logger_lib.Logger(lib='TFLINREG')
  VERBOSITY = 10

  features,labels = read_boston_data()
  labels = labels.reshape(-1, 1)
  m = labels.shape[0]
  normalized_features = min_max_normalization(features)

  if VERBOSITY >= 10:
    logger._log('Finished fetching and normalizing the dataset!')

  X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.20)
  data_sets = {'train': (X_train, y_train), 'test': (X_test, y_test)}
  data_sets = dotdict(data_sets)

  lr = LinearRegression(data_sets, logger, VERBOSITY)

  lr.train_and_predict()
