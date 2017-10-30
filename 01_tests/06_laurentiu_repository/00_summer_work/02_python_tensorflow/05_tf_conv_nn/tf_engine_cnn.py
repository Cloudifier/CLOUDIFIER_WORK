import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os

valid_activations = ['','direct','sigmoid','relu','softmax']

class TFConvolutionalNN:
  def __init__(self, architecture, model_name, logger, TEST=None, VERBOSITY=10):
    self.model_name = model_name
    self.logger = logger
    self.VERBOSITY = VERBOSITY
    self.train_cost_history = list()
    self.validation_cost_history = list()
    self.descriptions = list()
    self.nr_weights = 0
    self.graph = tf.Graph()
    #self.TEST = TEST

    """
    architecture:
      - number of conv_relu_maxpooling layers (num_layers)
      - list of nr_filters for each conv_relu_maxpooling layer (num_filters)
      - H, W, C of an image (H, W, C)
      - filter size (filter_size)
      - pool size (pool_size)
      - nr units fully connected (fc_units)
      - fully connected layer activation (fc_act)
      - nr classes used for prediction (classes)
    """
    self.architecture = architecture

    if len(self.architecture.num_filters) != self.architecture.num_layers:
      raise Exception("[TFConvolutionalNN ERROR] Inconsistent architecture !")

    if len(self.architecture.fc_units) != len(self.architecture.fc_act):
      raise Exception("[TFConvolutionalNN ERROR] Inconsistent architecture (FC)!")

    self.describe_architecture()


  def describe_architecture(self):
    num_layers = self.architecture.num_layers
    for i in range(num_layers):
      res = ' Layer:[{}]'.format(i)
      res += ' Type:[conv_relu]'
      res += ' NumFilters:[{}]'.format(self.architecture.num_filters[i])
      res += ' FilterSize:[{}]'.format(self.architecture.filter_size)
      self.descriptions.append(res)
      
    res = ' Layer:[{}]'.format(num_layers)
    res += ' Type:[GMP layer]'
    res += ' Units:[{}]'.format(self.architecture.num_filters[-1])
    self.descriptions.append(res)
    
    nr_fc_units = len(self.architecture.fc_units)
    for i in range(nr_fc_units):
      res = ' Layer:[{}]'.format(num_layers + i + 1)
      res += ' Type:[fully_connected layer]'
      res += ' Units:[{}]'.format(self.architecture.fc_units[i])
      res += ' Act:[{}]'.format(self.architecture.fc_act[i])
      self.descriptions.append(res)

    res = ' Layer:[{}]'.format(num_layers + nr_fc_units +  1)
    res += ' Type:[softmax layer]'
    res += ' Units:[{}]'.format(self.architecture.classes)
    res += ' Act:[softmax]'
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

  def create_conv_relu_max_layer(self, input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    self.nr_weights += (filter_shape[0] * filter_shape[1] * num_input_channels + 1) * num_filters

    with self.graph.as_default():
      weights = tf.get_variable(name='W'+name, dtype=tf.float32, shape=conv_filter_shape,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                 dtype=tf.float32))
      bias = tf.Variable(name='b'+name, initial_value=np.zeros(shape=(num_filters)), dtype=tf.float32)

      # convolutional layer operation
      out_layer = tf.nn.conv2d(input_data, weights, strides=[1, 1, 1, 1], padding='SAME')
      out_layer += bias

      # apply relu
      out_layer = tf.nn.relu(out_layer)

      # max pooling
      """
      ksize = [1, pool_shape[0], pool_shape[1], 1]
      strides = [1, 2, 2, 1]
      out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
      """

    return out_layer

  def create_new_layer(self, input_data, keep_prob_placeholder, nr_units, prev_layer_neurons,
                       layer_id, activation=''):
    """
    Creates new layer (excepting input)
    """

    if not (activation in valid_activations):
      raise Exception("[TFCNN:" + str(layer_id) + " ERROR]" +
                      " unknown activation: " + activation)

    weights_shape = (prev_layer_neurons, nr_units)
    self.nr_weights += (prev_layer_neurons + 1) * (nr_units)
    with self.graph.as_default():
      weights = tf.get_variable(name='Wd'+str(layer_id), dtype=tf.float32, shape=weights_shape,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
      bias = tf.Variable(name='bd'+str(layer_id), initial_value=np.zeros(shape=(nr_units)), dtype=tf.float32)
      z = tf.add(tf.matmul(input_data, weights), bias, name='z'+str(layer_id))

      # just relu and softmax !!!
      if activation == 'relu':
        a = tf.nn.relu(z, name='a'+str(layer_id))
        a = tf.nn.dropout(a, keep_prob=keep_prob_placeholder)
      else:
        self.logger._log('[ERROR] Please add ReLU as act for dense layer')

    return a


  def train(self, X_train, y_train, X_validation=None, y_validation=None,
            epochs=10, batch_size=10, learning_rate=0.01, drop_keep=1):
    im_width = self.architecture.W
    im_height = self.architecture.H
    im_channels = self.architecture.C
    filter_size = self.architecture.filter_size
    pool_size = self.architecture.pool_size
    fc_units = self.architecture.fc_units
    classes = self.architecture.classes
    num_layers = self.architecture.num_layers
    num_filters = self.architecture.num_filters

    total_train_time = 0

    self.logger._log("Training tf_cnn model... epochs={}, alpha={:.3f}, batch_sz={}, drop={}"
                     .format(epochs, learning_rate, batch_size, drop_keep))

    with self.graph.as_default():
      tf_X_batch = tf.placeholder(dtype=tf.float32, shape=(None, None, None, im_channels), name='X_batch')
      #tf_X_batch_shaped = tf.reshape(tf_X_batch, [-1, im_height, im_width, im_channels], name='X_batch_reshaped')
      tf_y_batch = tf.placeholder(dtype=tf.float32, shape=(None, classes), name='y_batch')
      tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

      tf_curr_layer_out = tf_X_batch
      curr_input_channels = im_channels
      for i in range(num_layers):
        tf_curr_layer_out = self.create_conv_relu_max_layer(input_data=tf_curr_layer_out,
                                                            num_input_channels=curr_input_channels,
                                                            num_filters=num_filters[i],
                                                            filter_shape=[filter_size, filter_size],
                                                            pool_shape=[pool_size, pool_size],
                                                            name=str(i))
        curr_input_channels = num_filters[i]


      tf_gmp = tf.reduce_max(tf_curr_layer_out, axis=[1,2])
      tf_curr_fclayer_out = tf_gmp
      prev_units = curr_input_channels
      for i in range(len(fc_units)):
        tf_curr_fclayer_out = self.create_new_layer(input_data=tf_curr_fclayer_out,
                                                    keep_prob_placeholder=tf_keep_prob,
                                                    nr_units=fc_units[i],
                                                    prev_layer_neurons=prev_units,
                                                    layer_id=i,
                                                    activation=self.architecture.fc_act[i])
        prev_units = fc_units[i]

      self.nr_weights += (prev_units + 1) * classes
      tf_wd2 = tf.get_variable(name='w_softmax', dtype=tf.float32, shape=[prev_units, classes],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None,
                                                                                dtype=tf.float32))
      tf_bd2 = tf.Variable(name='b_softmax', initial_value=np.zeros(shape=(classes)), dtype=tf.float32)
      tf_dense_layer2 = tf.matmul(tf_curr_fclayer_out, tf_wd2) + tf_bd2
      tf_dense_layer2_softmax = tf.nn.softmax(tf_dense_layer2)

      tf_y_pred = tf.argmax(tf_dense_layer2_softmax, 1, name="predictions")
      tf_correct_prediction = tf.equal(tf_y_pred, tf.argmax(tf_y_batch, 1))
      tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, tf.float32), name="accuracy")

      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf_dense_layer2, labels=tf_y_batch),
                                     name='loss')
      train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

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
    #saver.save(sess, os.path.join('./', self.model_name))

    n_batches = X_train.shape[0] // batch_size
    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()

      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        X_batch = X_batch.reshape(-1, im_height, im_width, im_channels)
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]

        _, loss, y_pred = sess.run([train_step, cross_entropy, tf_y_pred],
                                   feed_dict={tf_X_batch: X_batch,
                                              tf_y_batch: self.OneHotMatrix(y_batch),
                                              tf_keep_prob: drop_keep})

        if i % 500 == 0:
          self.logger._log('   [TRAIN Minibatch: {}] loss: {:.3f}'.format(i, loss))
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
      m, s = divmod(epoch_time, 60)

      if self.VERBOSITY > 10:
        loss_train, acc_train = sess.run([cross_entropy, tf_accuracy],
                                         feed_dict={tf_X_batch: X_train,
                                                    tf_y_batch: self.OneHotMatrix(y_train),
                                                    tf_keep_prob: 1})
        self.train_cost_history.append(loss_train)

        if (X_validation is not None) and (y_validation is not None):
          loss_validation, acc_valid = sess.run([cross_entropy, tf_accuracy],
                                                feed_dict={tf_X_batch: X_validation,
                                                           tf_y_batch: self.OneHotMatrix(y_validation),
                                                           tf_keep_prob: 1})
          self.validation_cost_history.append(loss_validation)

          self.logger._log('{}m{:.2f}s - loss: {:.2f} - acc: {:.2f}% - val_loss: {:.2f} - val_acc: {:.2f}%\n'
                           .format(int(m), s, loss_train, acc_train * 100, loss_validation, acc_valid * 100))
        else:
          self.logger._log('{}m{:.2f}s - loss: {:.2f} - acc: {:.2f}%\n'.format(int(m), s, loss_train, acc_train * 100))

    #save_path = saver.save(sess, 'tmp/model.ckpt')
    #self.logger._log('Model saved in {}'.format(save_path))
    m, s = divmod(total_train_time, 60)
    self.logger._log('Total TRAIN time: {}m{:.2f}s'.format(int(m), s))
    
    """
    JUST A WORKAROUND
    """
    
    """
    y_test = self.TEST.y
    for item in self.TEST.X:
        scene_height = item[1]
        scene_width = item[2]
        self.logger._log("Classifying images ({}x{})...".format(scene_height, scene_width))
        batch_size = 100
        X_test = item[0]
        n_batches = X_test.shape[0] // batch_size
        predicted = []
        for i in range(n_batches):
          X_batch = X_test[(i * batch_size):((i + 1) * batch_size), :]
          y_batch = y_test[(i * batch_size):((i + 1) * batch_size)]
          loss,acc,y_pred = sess.run([cross_entropy, tf_accuracy, tf_y_pred],
                                     feed_dict={tf_X_batch: X_batch.reshape(-1, scene_height, scene_width, self.architecture.C),
                                                  tf_y_batch: self.OneHotMatrix(y_batch),
                                                  tf_keep_prob: 1})
          predicted.append(y_pred)
        predicted = np.array(predicted).flatten()
        
        l = predicted.shape[0]
        acc = np.sum(predicted == y_test[:l]) / l
        self.logger._log("test_acc: {:.2f}%".format(acc * 100))
    """
    sess.close()


  def predict(self, X_test, y_test, scene_height, scene_width):
    saver = tf.train.import_meta_graph('tmp/model.ckpt.meta')
    sess = tf.Session()
    saver.restore(sess, 'tmp/model.ckpt')
    graph = tf.get_default_graph()
    self.logger._log('   [PREDICT] restored graph')


    # [n.name for n in tf.get_default_graph().as_graph_def().node]
    tf_keep_prob = graph.get_tensor_by_name('keep_prob:0')
    tf_X = graph.get_tensor_by_name('X_batch:0')
    tf_y = graph.get_tensor_by_name('y_batch:0')
    tf_loss = graph.get_tensor_by_name('loss:0')
    tf_accuracy = graph.get_tensor_by_name('accuracy:0')
    tf_y_pred = graph.get_tensor_by_name('predictions:0')

    self.logger._log("Classifying images ({}x{})...".format(scene_height, scene_width))
    batch_size = 100
    n_batches = X_test.shape[0] // batch_size
    predicted = []
    for i in range(n_batches):
      X_batch = X_test[(i * batch_size):((i + 1) * batch_size), :]
      y_batch = y_test[(i * batch_size):((i + 1) * batch_size)]
      loss, acc, y_pred = sess.run([tf_loss, tf_accuracy, tf_y_pred],
                                   feed_dict={tf_X: X_batch.reshape(-1, scene_height, scene_width, self.architecture.C),
                                              tf_y: self.OneHotMatrix(y_batch),
                                              tf_keep_prob: 1})
      predicted.append(y_pred)

      self.logger._log("Batch {} - test_loss: {:.2f} - test_acc: {:.2f}%".format(i, loss, acc * 100))

    sess.close()
    return predicted


if __name__ == "__main__":
    print("[engine_cnn] Cannot run library module!")