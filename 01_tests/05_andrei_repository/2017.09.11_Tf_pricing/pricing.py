import os
import platform
import importlib
from importlib.machinery import SourceFileLoader
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class LinearRegression():

  def __init__(self, logger):
    self.logger = logger

  def cost_function(self, linear_model, y):
    return tf.reduce_sum(tf.square((linear_model - y) / (2 * y.shape[0])))

  def train(self, X_train, y_train, initial_theta, bias, learning_rate, n_epochs):
    tf_theta = tf.Variable(initial_value = initial_theta, dtype = tf.float32)
    tf_b = tf.Variable(initial_value = bias, dtype = tf.float32)

    tf_X = tf.placeholder(dtype = tf.float32, shape = (None, X_train.shape[1]))
    linear_model = tf.add(tf.matmul(tf_X, tf_theta), tf_b)
    tf_y = tf.placeholder(dtype = tf.float32)

    loss = self.cost_function(linear_model, tf_y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(n_epochs):
      sess.run(train, feed_dict = { tf_X : X_train, tf_y : y_train })

    curr_W, curr_b, curr_loss = sess.run([tf_theta, tf_b, loss], { tf_X : X_train, tf_y : y_train })
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

  def predict(self, X_test, y_test):
    pass

def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path


if __name__ == "__main__":

   _, utils_path, data_path = get_paths(platform.system(), "_pricing_data")

   logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
   logger = logger_lib.Logger(show = True)

   data_file =  os.path.join(data_path, "prices.csv")
   df = pd.read_csv(data_file)

   print(df[:10])
   X = np.array(df.iloc[:, 1:-1].values)
   y = np.array(df.iloc[:, -1].values)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
   initial_theta = np.zeros((X_train.shape[1], 1))
   bias = [0]

   solver = LinearRegression(logger)
   solver.train(X_train, y_train, initial_theta, bias, 0.01, 50)

