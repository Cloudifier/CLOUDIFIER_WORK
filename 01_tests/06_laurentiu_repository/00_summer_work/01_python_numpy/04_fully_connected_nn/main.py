from engine_dnn import DNNLayer, NeuralNetwork
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from logger_helper import LoadLogger
import os

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__



def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val




if __name__ == "__main__":
  logger = LoadLogger(lib_name='DNN', config_file='config.txt')
  df_train = pd.read_csv(os.path.join(logger._base_folder, logger.config_data['TRAIN_DF']))
  df_test = pd.read_csv(os.path.join(logger._base_folder, logger.config_data['TEST_DF']))
  
  X_train = min_max_scaler(df_train.loc[:, "pixel1":"pixel784"].values)
  y_train = df_train.loc[:, "label"].values
  
  X_test = min_max_scaler(df_test.loc[:, "pixel1":"pixel784"].values)
  y_test = df_test.loc[:, "label"].values
  
  
  hyper_parameters = {'learning_rate': 0.01, 'momentum_speed': 0.9, 'epochs': 15, 'batch_size': 10, 'beta': 0,
                      'decay_factor': 1}
  hyper_parameters = dotdict(hyper_parameters)

  nn = NeuralNetwork(logger, hyper_parameters)
  nn.AddLayer(DNNLayer(nr_units=784, layer_name='input_layer', layer_type='input'))
  #nn.AddLayer(DNNLayer(nr_units=256, layer_name='hidden_layer', activation='relu', layer_type='hidden'))
  nn.AddLayer(DNNLayer(nr_units=10, layer_name='output_layer', activation='softmax', layer_type='output'))
  nn.PrepareModel()

  
  nn.train(X_train, y_train)
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