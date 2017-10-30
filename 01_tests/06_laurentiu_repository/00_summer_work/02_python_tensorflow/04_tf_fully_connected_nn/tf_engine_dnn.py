import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

valid_activations = ['','direct','sigmoid','relu','softmax']
valid_layers = ['','hidden','output']

class TFNeuralNetwork:
  def __init__(self, architecture, model_name, logger, VERBOSITY=10):
    self.logger = logger
    self.VERBOSITY = VERBOSITY
    self.train_cost_history = list()
    self.validation_cost_history = list()
    self.model_name = model_name
    self.graph = tf.Graph()
    self.descriptions = list()
    self.nr_weights = 0

    """
    architecture:
      - list of sizes
      - list of names
      - list of types
      - list of activations
      {'sizes': [...], 'names': [...], 'types': [...], 'activations': [...]}
    """
    self.architecture = architecture

    l1 = len(self.architecture.sizes)
    l2 = len(self.architecture.names)
    l3 = len(self.architecture.types)
    l4 = len(self.architecture.activations)
    if (l1 + l2 + l3 + l4) / 4 != l1:
      raise Exception("[TFNeuralNetwork ERROR] Inconsistent architecture !")

    self.nr_layers = len(self.architecture.sizes)
    if self.nr_layers == 0:
      raise Exception("[TFNeuralNetwork ERROR] Zero layers !")
    elif self.nr_layers < 3:
      raise Exception("[TFNeuralNetwork ERROR] Nr. layers < 3")

    for i in range(len(self.architecture.sizes)):
      res = ' Layer:[{}]'.format(i)
      res += ' Name:[{}]'.format(self.architecture.names[i])
      res += ' Type:[{}]'.format(self.architecture.types[i])
      res += ' Act:[{}]'.format(self.architecture.activations[i])
      res += ' Units:[{}]'.format(self.architecture.sizes[i])
      self.descriptions.append(res)

    res = "Layers:"
    for i in range(len(self.descriptions)):
      res += "\n  " + self.descriptions[i]
    self.logger._log(res)

  def OneHotMatrix(self, y, classes=10):
    n_obs = y.shape[0]
    ohm = np.zeros((n_obs, classes))
    ohm[np.arange(n_obs), y.astype(int)] = 1
    return ohm

  def create_new_layer(self, input_data, keep_prob_placeholder, nr_units, prev_layer_neurons,
                       layer_id, layer_type='', activation='', layer_name=''):
    """
    Creates new layer (excepting input)
    """

    if not (layer_type in valid_layers):
      raise Exception("[TFNeuralNetwork:" + str(layer_id) + " ERROR]" +
                      " unknown layer type: " + layer_type)
    if not (activation in valid_activations):
      raise Exception("[TFNeuralNetwork:" + str(layer_id) + " ERROR]" +
                      " unknown activation: " + activation)

    weights_shape = (prev_layer_neurons, nr_units)
    self.nr_weights += (prev_layer_neurons + 1) * (nr_units)
    with self.graph.as_default():
      weights = tf.get_variable(name='W'+str(layer_id), dtype=tf.float32, shape=weights_shape,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
      bias = tf.Variable(name='bias'+str(layer_id), initial_value=np.zeros(shape=(nr_units)), dtype=tf.float32)
      z = tf.add(tf.matmul(input_data, weights), bias, name='z'+str(layer_id))

      # just relu and softmax !!!
      if activation == 'relu':
        a = tf.nn.relu(z, name='a'+str(layer_id))
        a = tf.nn.dropout(a, keep_prob=keep_prob_placeholder)
      elif activation == 'softmax':
        a = tf.nn.softmax(z, name='a'+str(layer_id))

    return a

  def train(self, X_train, y_train, X_validation=None, y_validation=None,
            epochs=10, batch_size=10, learning_rate=0.01, beta=0, momentum_speed=0, drop_keep=1):

    total_train_time = 0
    classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    self.logger._log("Training tf_dnn model (initialized using xavier method)... epochs={}, alpha={:.2f}, batch_sz={}, beta={}, momentum={}, drop={}"
      .format(epochs, learning_rate, batch_size, beta, momentum_speed, drop_keep))

    with self.graph.as_default():
      tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
      tf_X_batch = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='X_batch')
      tf_y_batch = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='y_batch')

      tf_curr_layer_out = tf_X_batch
      for i in range(len(self.architecture.sizes)-1):
        tf_curr_layer_out = self.create_new_layer(input_data=tf_curr_layer_out,
                                                  keep_prob_placeholder=tf_keep_prob,
                                                  nr_units=self.architecture.sizes[i+1],
                                                  prev_layer_neurons=self.architecture.sizes[i],
                                                  layer_id=i,
                                                  layer_type=self.architecture.types[i+1],
                                                  activation=self.architecture.activations[i+1],
                                                  layer_name=self.architecture.names[i+1])

      tf_y_pred = tf.argmax(tf_curr_layer_out, 1, name="predictions")
      tf_correct_prediction = tf.equal(tf_y_pred, tf.argmax(tf_y_batch, 1))
      tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, tf.float32), name="accuracy")

      cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_y_batch * tf.log(tf_curr_layer_out), reduction_indices=[1]), name='loss')

      if momentum_speed == 0:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
      else:
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum_speed).minimize(cross_entropy)

      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    model_size_MB = self.nr_weights * 4 / (1024 * 1024)
    self.logger._log("Model capacity: {:,} weights, {:,.2f}MB"
                     .format(self.nr_weights, model_size_MB))
    if (model_size_MB > 4000):
      self.logger._log("Model requires to much memory, please optimize!")
      return

    sess = tf.Session(graph=self.graph)
    sess.run(init)

    n_batches = X_train.shape[0] // batch_size
    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()

      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]

        _, loss, y_pred = sess.run([train_step, cross_entropy, tf_y_pred],
                                   feed_dict={tf_X_batch: X_batch,
                                              tf_y_batch: self.OneHotMatrix(y_batch),
                                              tf_keep_prob : drop_keep})

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

      loss_train, acc_train = sess.run([cross_entropy, tf_accuracy],
                                       feed_dict={tf_X_batch: X_train,
                                                  tf_y_batch: self.OneHotMatrix(y_train),
                                                  tf_keep_prob : 1})
      self.train_cost_history.append(loss_train)

      if (X_validation is not None) and (y_validation is not None):
        loss_validation, acc_valid = sess.run([cross_entropy, tf_accuracy],
                                              feed_dict={tf_X_batch: X_validation,
                                                         tf_y_batch: self.OneHotMatrix(y_validation),
                                                         tf_keep_prob : 1})
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

    # [n.name for n in tf.get_default_graph().as_graph_def().node]
    tf_keep_prob = graph.get_tensor_by_name('keep_prob:0')
    tf_X = graph.get_tensor_by_name('X_batch:0')
    tf_y = graph.get_tensor_by_name('y_batch:0')
    tf_loss = graph.get_tensor_by_name('loss:0')
    tf_accuracy = graph.get_tensor_by_name('accuracy:0')
    tf_y_pred = graph.get_tensor_by_name('predictions:0')

    loss, acc, y_pred = sess.run([tf_loss, tf_accuracy, tf_y_pred],
                                 feed_dict={tf_X: X_test,
                                            tf_y: self.OneHotMatrix(y_test),
                                            tf_keep_prob: 1})
    self.logger._log("Predicting ... test_loss: {:.2f} - test_acc: {:.2f}%".format(loss, acc * 100))
    sess.close()

    return y_pred

  def plot_cost_history(self, cost_history):
    plt.plot(np.arange(0, len(cost_history)), cost_history)
    # plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()

if __name__ == "__main__":
    print("[engine_dnn] Cannot run library module!")