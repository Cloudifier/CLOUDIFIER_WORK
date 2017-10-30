import tensorflow as tf
import pandas as pd
import numpy as np
import platform
import importlib
import time
import os
from utils import get_paths, load_mnist, Struct, one_hot
from importlib.machinery import SourceFileLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from tqdm import tqdm, trange
import inspect
import sys

def preprocess(df, test_size, random_state, logger):

  data_set = Struct()
  X = np.array(df.iloc[:, 1:].values)
  y = np.array(df.iloc[:, 0].values, dtype = int)

  normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
  X = normalizer.fit_transform(X)

  logger.log("Finished normalizing data")

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size,
    random_state = random_state)

  logger.log("Finished spliting data into train({:.2f}), test({:.2f})".
    format(1 - test_size, test_size))

  data_set.train_X = train_X
  data_set.train_y = train_y
  data_set.test_X = test_X
  data_set.test_y = test_y

  return data_set

def solve(data_set, logger):

  old_print = print
  inspect.builtins.print = tqdm.write

  tf_trainX = tf.placeholder(dtype = "float", shape = [None, 784])
  tf_testX  = tf.placeholder(dtype = "float", shape = [784])

  distance = tf.reduce_sum(tf.abs(tf.add(tf_trainX, tf.negative(tf_testX))), reduction_indices = 1)
  pred_step = tf.nn.top_k(-distance, k = 5, sorted = True)[1]

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  corrects = 0
  wrongs = 0
  t = trange(data_set.test_X.shape[0], desc='Slider', leave=True)
  for i in range(data_set.test_X.shape[0]):

    top_k_idx = sess.run(pred_step, feed_dict = {
                                                tf_trainX : data_set.train_X,
                                                tf_testX : data_set.test_X[i, :]
                                            })
    top_k = data_set.train_y[top_k_idx]
    crt_val, crt_count = mode(top_k, axis=0)
    crt_val = crt_val[0]

    if crt_val == data_set.test_y[i]:
      corrects += 1
    else:
      wrongs += 1

    t.set_description("Real {} -- Pred {} -- corrects = {}, wrongs = {} -- accuracy = {:.2f} %"
      .format(data_set.test_y[i], crt_val, corrects, wrongs, (corrects/(corrects + wrongs)) * 100))
    t.refresh()
    t.update(1)
    sys.stdout.flush()

    if i % 50 == 0:
      logger.log("Corrects/Wrongs: {}/{}".format(corrects, wrongs), show = False)

  accuracy = (corrects*100) / data_set.test_X.shape[0]
  logger.log("Test accuracy: {}".format(accuracy))

  inspect.builtins.print = old_print
  sess.close()


if __name__=='__main__':

  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
  logger = logger_lib.Logger(show = True, alg_type = "KNN")

  mnist_df = load_mnist(utils_path)
  logger.log("Finished fetching MNIST")

  data_set = preprocess(mnist_df, 0.3, 13, logger)

  solve(data_set, logger)






