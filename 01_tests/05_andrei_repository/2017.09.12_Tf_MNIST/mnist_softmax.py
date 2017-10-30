import tensorflow as tf
from utils import get_paths, plot
import pandas as pd
import numpy as np
import platform
import importlib
import time
import os
from importlib.machinery import SourceFileLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata


class MnistSolver():

  def __init__(self, df, logger):
    self.logger = logger
    self.df = df
    self.sess = None
    self.model = (None, None)

  def one_hot(self, y):
    zeros = [[0 for i in range(10)] for i in range(y.shape[0])]
    return np.array([zero[:int(i)] + [1] + zero[int(i)+1:] for zero,i in zip(zeros,y)])

  def preprocess(self, random_state, test_size):

    self.random_state = random_state
    self.test_size = test_size

    X = np.array(self.df.iloc[:, 1:].values)
    y = np.array(self.df.iloc[:, 0].values, dtype = int)

    self.normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = self.normalizer.fit_transform(X)

    self.logger.log("Finished normalizing data")

    self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size =
      self.test_size, random_state = self.random_state)

    self.test_X, self.validation_X, self.test_y, self.validation_y = train_test_split(self.test_X, self.test_y, test_size = 0.5, random_state = int(self.random_state / 2))

    self.logger.log("Finished spliting data into train({:.2f}), test({:.2f}), validation({:.2f})".
      format(1 - self.test_size, self.test_size / 2, self.test_size / 2))


  def train(self, num_epochs = 10, batch_size = 10, alpha = 0.5, beta = 0, momentum = 0,
    verbosity = 5):

    start = time.time()

    tf_X = tf.placeholder(dtype = tf.float32, shape = (None, 784))
    tf_y = tf.placeholder(dtype = tf.float32, shape = (None, 10))

    tf_theta = tf.Variable(initial_value = tf.zeros(shape = (784, 10)))
    tf_bias = tf.Variable(initial_value = tf.zeros(shape = (10)))
    tf_logits = tf.matmul(tf_X, tf_theta) + tf_bias
    tf_logits = tf.nn.softmax(tf_logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_y,
      logits = tf_logits))
    regularizer = tf.nn.l2_loss(tf_theta)
    loss =  tf.reduce_mean(loss + beta * regularizer)
    if momentum is None:
      train_step = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    else:
      train_step = tf.train.MomentumOptimizer(alpha, momentum).minimize(loss)

    if momentum == 0:
      self.reg_type = "simple"
    else:
      self.reg_type = "momentum"

    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

    self.logger.log("Start training {} solver: alpha-{:.3f}, beta-{:.3f}, speed-{:.2f}, batch-{}, num_epochs-{}".format(self.reg_type, alpha, beta, momentum, batch_size, num_epochs))

    loss_history = []

    for crt_epoch in range(num_epochs):
      for i in np.arange(0, self.train_X.shape[0], batch_size):
        current_X = self.train_X[i : i + batch_size, :]
        current_y = self.train_y[i : i + batch_size]
        self.sess.run(train_step, feed_dict = {tf_X: current_X, tf_y: self.one_hot(current_y)})

      crt_loss =  loss.eval(session = self.sess, feed_dict = {tf_X: current_X, tf_y: self.one_hot(current_y)})

      loss_history.append(crt_loss)

      if crt_epoch % verbosity == 0:
        self.logger.log("Loss at epoch#{}: {:.2f}".format(str(crt_epoch).
          zfill(len(str(num_epochs))), crt_loss), tabs = 2)
        correct_prediction = tf.equal(tf.argmax(tf_logits, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.logger.log("Accuracy at epoch#{}: {:.2f}".format(str(crt_epoch).zfill(len(str(num_epochs))), self.sess.run(accuracy, feed_dict = {tf_X: self.validation_X, tf_y: self.one_hot(self.validation_y)})), tabs = 2)

    self.model = (tf_theta, tf_bias)

    stop = time.time()

    self.logger.log("Total train time for {} solver: {:.2f}s".format(self.reg_type, stop - start), tabs = 1)

    correct_prediction = tf.equal(tf.argmax(tf_logits, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.logger.log("Train accuracy for {} solver {:.2f}".format(self.reg_type, self.sess.run(accuracy, feed_dict = {tf_X: self.train_X, tf_y: self.one_hot(self.train_y)})), tabs = 1)

    plot(loss_history)


  def predict(self):

    tf_X = tf.placeholder(dtype = tf.float32, shape = (None, 784))
    tf_y = tf.placeholder(dtype = tf.float32, shape = (None, 10))

    tf_theta = self.model[0]
    tf_bias  = self.model[1]
    tf_ypred = tf.matmul(tf_X, tf_theta) + tf_bias
    tf_ypred = tf.nn.softmax(tf_ypred)

    correct_prediction = tf.equal(tf.argmax(tf_ypred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self.logger.log("Test accuracy for {} solver {:.2f}".format(self.reg_type, self.sess.run(accuracy, feed_dict = {tf_X: self.test_X, tf_y: self.one_hot(self.test_y)})), tabs = 1)

  def clean(self):
    self.sess.close()
    self.df = Nonein
    self.train_X = None
    self.test_X = None
    self.train_y = None
    self.test_y = None

if __name__=='__main__':

  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
  logger = logger_lib.Logger(show = True, alg_type = "LOGREG")

  mnist = fetch_mldata('MNIST original', data_home = mnist_path)
  labels = ["pixel_" + str(i) for i in range(784)]
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']], \
    columns = ["Digit_label"] + labels)
  logger.log("Finished fetching MNIST")

  solver = MnistSolver(mnist_df, logger)
  solver.preprocess(13, 0.3)

  solver.train(num_epochs = 20, batch_size = 10, alpha = 0.01, beta = 0.001)
  solver.predict()
  solver.sess.close()

  solver.train(num_epochs = 20, batch_size = 10, alpha = 0.01, beta = 0.001, momentum = 0.9)
  solver.predict()

  solver.clean()








