from engine_dnn import DNNLayer, NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
import os
from time import time as tm
import platform
from importlib.machinery import SourceFileLoader

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
  logger = logger_lib.Logger(lib='DNN')

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
  hyper_parameters = {'learning_rate': 0.01, 'momentum_speed': 0.9, 'epochs': 15, 'batch_size': 10, 'beta': 0,
                      'decay_factor': 1}
  hyper_parameters = dotdict(hyper_parameters)

  nn = NeuralNetwork(logger, hyper_parameters)
  nn.AddLayer(DNNLayer(nr_units=784, layer_name='input_layer', layer_type='input'))
  nn.AddLayer(DNNLayer(nr_units=256, layer_name='hidden_layer', activation='relu', layer_type='hidden'))
  nn.AddLayer(DNNLayer(nr_units=10, layer_name='output_layer', activation='softmax', layer_type='output'))
  nn.PrepareModel()

  X_train, y_train = data_sets.train
  X_validation, y_validation = data_sets.validation
  X_test, y_test = data_sets.test

  nn.train(X_train, y_train, X_validation, y_validation)
  y_pred = nn.predict(X_test, y_test)

  """
  nr_examples = X_train.shape[0]
  batch_size = hyper_parameters.batch_size
  nr_batches = nr_examples // hyper_parameters.batch_size
  t0 = tm()
  for epoch in range(hyper_parameters.epochs):
    logger._log("Start epoch {}".format(epoch+1))
    for i in range(nr_batches):
      xi = X_train[(i * batch_size):((i + 1) * batch_size), :]
      yi = y_train[(i * batch_size):((i + 1) * batch_size)]
      nn.Train(xi, yi)
    nn.step = 0
  t1 = tm()
  tdelta = t1 - t0
  logger._log("Training {} epochs finished in {:.2f}s".format(hyper_parameters.training_epochs, tdelta))
  """
