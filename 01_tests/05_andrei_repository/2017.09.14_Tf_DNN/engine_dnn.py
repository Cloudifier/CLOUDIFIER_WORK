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

def preprocess(df, test_size, random_state, logger):

  data_set = Struct()
  X = np.array(df.iloc[:, 1:].values)
  y = np.array(df.iloc[:, 0].values, dtype = int)

  normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
  X = normalizer.fit_transform(X)

  logger.log("Finished normalizing data")

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size,
    random_state = random_state)

  test_X, validation_X, test_y, validation_y = train_test_split(test_X, test_y, test_size = 0.5, random_state = int(random_state / 2))

  logger.log("Finished spliting data into train({:.2f}), test({:.2f}), validation({:.2f})".
    format(1 - test_size, test_size / 2, test_size / 2))

  data_set.train_X = train_X
  data_set.train_y = train_y
  data_set.test_X = test_X
  data_set.test_y = test_y
  data_set.validation_X = validation_X
  data_set.validation_y = validation_y

  return data_set

def solve(data_set, learning_rate, batch_size, num_epochs, h1_size, drop_keep, logger):

  mnist_graph = tf.Graph()

  with mnist_graph.as_default():

    tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=())

    tf_X_batch = tf.placeholder(dtype = tf.float32, shape = (None, 784))
    tf_y_batch = tf.placeholder(dtype = tf.float32, shape = (None, 10))


    tf_weights_h1 = tf.get_variable(name = "h1Weights", dtype = tf.float32, shape = (784, h1_size),
                                 initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed = None, dtype = tf.float32))
    tf_bias_h1 = tf.Variable(initial_value = np.zeros(shape = (h1_size)), dtype = tf.float32)


    tf_weights_h2 = tf.get_variable(name = "h2Weights", dtype = tf.float32, shape = (h1_size, 10),
                                    initializer = tf.contrib.layers.xavier_initializer(uniform =True, seed = None, dtype = tf.float32))
    tf_bias_h2 = tf.Variable(initial_value = np.zeros(shape=(10)), dtype = tf.float32)

    z1 = tf.matmul(tf_X_batch, tf_weights_h1) + tf_bias_h1
    a1 = tf.nn.relu(z1)

    a1drop = tf.nn.dropout(a1, keep_prob = tf_keep_prob)

    z2 = tf.matmul(a1drop, tf_weights_h2) + tf_bias_h2

    tf_output = tf.nn.softmax(z2)

    J = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tf_y_batch,logits = z2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer_op = optimizer.minimize(J)

    print("Creating variable initializer within graph")
    init_op = tf.global_variables_initializer()

  print("Creating session")
  session = tf.Session(graph = mnist_graph)

  print("Initializing variables")
  init_op.run(session = session)

  for epoch in range(num_epochs):
    minibatches =  int(data_set.train_X.shape[0] / batch_size)
    for i in range(minibatches):
      crt_X = data_set.train_X[i * batch_size : (i + 1) * batch_size, :]
      crt_y = data_set.train_y[i * batch_size : (i + 1) * batch_size]

      _, batch_loss = session.run([optimizer_op, J], feed_dict={
                                                         tf_X_batch : crt_X,
                                                         tf_y_batch : one_hot(crt_y),
                                                         tf_keep_prob : drop_keep
                                                      })

      if i % 1000 == 0:
        print("epoch {} minibatch {} loss: {:.3f}".format(epoch, i, batch_loss))


    test_output = tf_output.eval(session = session, feed_dict = {
                                                      tf_X_batch : data_set.test_X,
                                                      tf_y_batch: one_hot(data_set.test_y),
                                                      tf_keep_prob : 1
                                                    })
    preds_test = np.sum(np.argmax(test_output, axis=1) == np.argmax(one_hot(data_set.test_y), axis=1))
    acc_test = preds_test / data_set.test_y.shape[0]
    print("Epoch {} test accuracy {:.2f}".format(epoch, acc_test * 100))

    train_output = session.run(tf_output, feed_dict = {
                                            tf_X_batch : data_set.train_X,
                                            tf_y_batch : one_hot(data_set.train_y),
                                            tf_keep_prob : 1
                                          })
    preds_train = np.sum(np.argmax(train_output, axis=1) == np.argmax(one_hot(data_set.train_y), axis=1))
    acc_train = preds_train / data_set.train_y.shape[0]
    print("Epoch {} train accuracy {:.2f}".format(epoch, acc_train * 100))

    return session, tf_output


if __name__=='__main__':

  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
  logger = logger_lib.Logger(show = True)

  mnist_df = load_mnist(utils_path)
  logger.log("Finished fetching MNIST")

  data_set = preprocess(mnist_df, 0.3, 13, logger)


  session, tf_output = solve(data_set = data_set, learning_rate = 0.01, batch_size =  16, num_epochs = 2, h1_size = 1024, drop_keep = 0.7, logger = logger)





