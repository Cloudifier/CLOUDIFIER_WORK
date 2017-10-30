import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

class LogisticRegression:
  def __init__(self, model_name, logger, VERBOSITY=1):
    self.logger = logger
    self.VERBOSITY = VERBOSITY
    self.train_cost_history = list()
    self.validation_cost_history = list()
    self.classes = 10
    self.model_name = model_name

  def OneHotMatrix(self, y, classes=10):
    n_obs = y.shape[0]
    ohm = np.zeros((n_obs, classes))
    ohm[np.arange(n_obs), y.astype(int)] = 1
    return ohm

  def train(self, X_train, y_train, X_validation=None, y_validation=None,
            epochs=15, learning_rate=0.01, batch_size=10, beta=0,
            momentum_speed=0, decay_factor=1):
    total_train_time = 0
    self.classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    self.logger._log("Training logreg model (initialized with 0)... epochs={}, alpha={:.2f}, batch_sz={}, beta={}, momentum={}, decay={}"
                     .format(epochs, learning_rate, batch_size, beta, momentum_speed, decay_factor))

    graph = tf.Graph()

    with graph.as_default():
      tf_X = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name="X_batch")
      tf_y = tf.placeholder(dtype=tf.float32, shape=(None, self.classes), name="y_batch")
      tf_theta = tf.Variable(initial_value=tf.zeros(shape=(784, 10)), dtype=tf.float32, name="theta")
      tf_bias = tf.Variable(initial_value=tf.zeros(shape=(10)), dtype=tf.float32, name="bias")
      tf_logits = tf.matmul(tf_X, tf_theta) + tf_bias
      tf_y_pred = tf.argmax(tf.nn.softmax(tf_logits), 1, name="predictions")
      tf_correct_prediction = tf.equal(tf_y_pred, tf.argmax(tf_y, 1))
      tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, tf.float32), name="accuracy")

      tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y,
                                                                    logits=tf_logits), name="loss")

      regularizer = tf.nn.l2_loss(tf_theta)
      tf_loss = tf.reduce_mean(tf_loss + beta * regularizer)

      if momentum_speed == 0:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
      else:
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum_speed).minimize(tf_loss)

      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    sess = tf.Session(graph=graph)
    sess.run(init)

    n_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()

      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]

        _, loss, y_pred = sess.run([train_step, tf_loss, tf_y_pred],
                                   feed_dict={tf_X: X_batch, tf_y: self.OneHotMatrix(y_batch)})

        if i % 1000 == 0:
          self.logger._log('   [TRAIN Minibatch: {}] loss: {:.2f}'.format(i, loss))
          if self.VERBOSITY >= 10:
            n_to_slice = batch_size
            if n_to_slice > 10:
              n_to_slice = 10
            d1_slice = y_batch[:n_to_slice]
            d2_slice = y_pred[:n_to_slice]
            self.logger._log('        yTrue:{}'.format(d1_slice.astype(int)))
            self.logger._log('        yPred:{}'.format(d2_slice))

      epoch_time = time() - epoch_start_time
      total_train_time += epoch_time

      loss_train, acc_train = sess.run([tf_loss, tf_accuracy], feed_dict={tf_X: X_train, tf_y: self.OneHotMatrix(y_train)})
      self.train_cost_history.append(loss_train)

      if (X_validation is not None) and (y_validation is not None):
        loss_validation, acc_valid = sess.run([tf_loss, tf_accuracy],
                                                   feed_dict={tf_X: X_validation, tf_y: self.OneHotMatrix(y_validation)})
        self.validation_cost_history.append(loss_validation)

        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}% - val_loss: {:.2f} - val_acc: {:.2f}%\n'
                         .format(epoch_time, loss_train, acc_train * 100, loss_validation, acc_valid * 100))
      else:
        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}%\n'.format(epoch_time, loss_train, acc_train * 100))

    saver.save(sess, self.model_name)
    self.logger._log('Total TRAIN time: {:.2f}s'.format(total_train_time))
    sess.close()


  def predict(self, X_test, y_test):
    saver = tf.train.import_meta_graph(self.model_name + '.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    tf_X = graph.get_tensor_by_name('X_batch:0')
    tf_y = graph.get_tensor_by_name('y_batch:0')
    tf_loss = graph.get_tensor_by_name('loss:0')
    tf_accuracy = graph.get_tensor_by_name('accuracy:0')
    tf_y_pred = graph.get_tensor_by_name('predictions:0')

    loss, acc, y_pred = sess.run([tf_loss, tf_accuracy, tf_y_pred], feed_dict={tf_X: X_test, tf_y: self.OneHotMatrix(y_test)})
    self.logger._log("Predicting ... test_loss: {:.2f} - test_acc: {:.2f}%".format(loss, acc*100))
    sess.close()

    return y_pred


  def plot_cost_history(self, cost_history):
    plt.plot(np.arange(0, len(cost_history)), cost_history)
    # plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()

  def plot_thetas(self):
    saver = tf.train.import_meta_graph(self.model_name + '.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    tf_theta = graph.get_tensor_by_name('theta:0')
    theta_no_bias = sess.run(tf_theta)
    sess.close()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Theta-uri')

    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(theta_no_bias[:, 0].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax1.set_title(str(0))

    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(theta_no_bias[:, 1].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax2.set_title(str(1))

    ax3 = fig.add_subplot(2, 5, 3)
    ax3.imshow(theta_no_bias[:, 2].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax3.set_title(str(2))

    ax4 = fig.add_subplot(2, 5, 4)
    ax4.imshow(theta_no_bias[:, 3].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax4.set_title(str(3))

    ax5 = fig.add_subplot(2, 5, 5)
    ax5.imshow(theta_no_bias[:, 4].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax5.set_title(str(4))

    ax6 = fig.add_subplot(2, 5, 6)
    ax6.imshow(theta_no_bias[:, 5].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax6.set_title(str(5))

    ax7 = fig.add_subplot(2, 5, 7)
    ax7.imshow(theta_no_bias[:, 6].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax7.set_title(str(6))

    ax8 = fig.add_subplot(2, 5, 8)
    ax8.imshow(theta_no_bias[:, 7].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax8.set_title(str(7))

    ax9 = fig.add_subplot(2, 5, 9)
    ax9.imshow(theta_no_bias[:, 8].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax9.set_title(str(8))

    ax10 = fig.add_subplot(2, 5, 10)
    ax10.imshow(theta_no_bias[:, 9].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax10.set_title(str(9))

    fig.savefig('thetas.png', dpi=fig.dpi)
    plt.show()