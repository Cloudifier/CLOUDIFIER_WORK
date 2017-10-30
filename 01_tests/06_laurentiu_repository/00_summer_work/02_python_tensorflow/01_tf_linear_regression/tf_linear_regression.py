import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import os
import numpy as np
np.set_printoptions(precision=1, suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LinearRegression():
  def __init__(self, data_sets, logger, VERBOSITY=1):
    self.data_sets = data_sets
    self.n_features = data_sets.train[0].shape[1]
    self.train_loss_history = []
    self.test_loss_history = []
    self.logger = logger
    self.VERBOSITY = VERBOSITY


  def do_eval(self, epoch, sess, loss, X_placeholder, y_placeholder, epoch_time):
    X_train, y_train = self.data_sets.train
    X_test, y_test = self.data_sets.test
    train_loss = sess.run(loss, feed_dict={X_placeholder: X_train, y_placeholder: y_train})
    test_loss = sess.run(loss, feed_dict={X_placeholder: X_test, y_placeholder: y_test})
    self.train_loss_history.append(train_loss)
    self.test_loss_history.append(test_loss)

    self.logger._log('{:.2f}s - train_loss: {:.3f} - test_loss: {:.3f}\n'.format(epoch_time, train_loss, test_loss))


  def train_and_predict(self, learning_rate=0.01, epochs=25, batch_size=10):
    self.logger._log("Training linreg model (initialized with 0)... epochs={}, alpha={:.2f}, batch_sz={}."
                     .format(epochs, learning_rate, batch_size))
    learning_rate = learning_rate
    training_epochs = epochs
    batch_size =batch_size
    graph = tf.Graph()

    with graph.as_default():
      tf_X_batch = tf.placeholder(dtype=tf.float32, shape=(None, self.n_features), name="X_batch")
      tf_y_batch = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y_batch")

      tf_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
      tf_weights = tf.get_variable(name="weights", shape=(self.n_features, 1), dtype=tf.float32,
                                   initializer=tf.zeros_initializer(), regularizer=tf_regularizer)
      tf_bias = tf.Variable(initial_value=[0], name="bias", dtype=tf.float32)

      tf_yhat = tf.matmul(tf_X_batch, tf_weights) + tf_bias

      tf_loss = tf.reduce_mean(tf.square(tf_yhat - tf_y_batch))

      tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
      tf_train = tf_optimizer.minimize(tf_loss)

      tf_init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(tf_init)

    total_train_time = 0
    X_train, y_train = self.data_sets.train
    n_batches = X_train.shape[0] // batch_size
    for epoch in range(training_epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()
      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]
        sess.run(tf_train, feed_dict={tf_X_batch: X_batch, tf_y_batch: y_batch})

        if i % 10 == 0:
          J = sess.run(tf_loss, feed_dict={tf_X_batch: X_batch, tf_y_batch: y_batch})
          y_pred = sess.run(tf_yhat, feed_dict={tf_X_batch: X_batch})
          self.logger._log('   [TRAIN Minibatch: {}] loss: {:.2f}'.format(i, J))
          if self.VERBOSITY >= 10:
            d1_slice = y_batch.reshape(y_batch.size)[:3]
            d2_slice = y_pred.reshape(y_pred.size)[:3]
            self.logger._log('        yTrue:{}'.format(d1_slice))
            self.logger._log('        yPred:{}'.format(d2_slice))

      epoch_time = time() - epoch_start_time
      total_train_time += epoch_time

      self.do_eval(epoch, sess, tf_loss, tf_X_batch, tf_y_batch, epoch_time)

    self.logger._log('Total TRAIN time: {:.2f}s'.format(total_train_time))

    fig, ax = plt.subplots()
    y_pred = sess.run(tf_yhat, feed_dict={tf_X_batch: self.data_sets.test[0]})
    y_test = self.data_sets.test[1]

    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    sess.close()