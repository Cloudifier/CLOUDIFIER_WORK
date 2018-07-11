import numpy as np
from sklearn.model_selection import train_test_split
from logger_helper import LoadLogger
from importlib.machinery import SourceFileLoader
from tf_engine_dnn import TFNeuralNetwork

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__



def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home='.')

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
                  'validation': (X_validation, y_validation)})

if __name__ == "__main__":
  data_sets = fetch_data()
  hyper_parameters = {'learning_rate': 0.001, 'momentum_speed': 0, 'epochs': 15, 'batch_size': 256, 'drop_keep': .7}
  hyper_parameters = dotdict(hyper_parameters)
  architecture = dotdict({
    'sizes': [784, 256, 10],
    'names': ['inputlayer', 'hiddenlayer', 'outputlayer'],
    'types': ['input', 'hidden', 'output'],
    'activations': ['', 'relu', 'softmax']
  })

  logger = LoadLogger(lib_name='TEST', config_file='config.txt')
  nn = TFNeuralNetwork(architecture, 'model_cache', logger)

  X_train, y_train = data_sets.train
  X_validation, y_validation = data_sets.validation
  X_test, y_test = data_sets.test

  nn.train(X_train=X_train,
           y_train=y_train,
           X_validation=X_validation,
           y_validation=y_validation,
           epochs=hyper_parameters.epochs,
           batch_size=hyper_parameters.batch_size,
           learning_rate=hyper_parameters.learning_rate,
           momentum_speed=hyper_parameters.momentum_speed,
           drop_keep=hyper_parameters.drop_keep)

  nn.predict(X_test, y_test)