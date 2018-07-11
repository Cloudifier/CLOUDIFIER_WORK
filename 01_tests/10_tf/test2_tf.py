import tensorflow as tf
import numpy as np
from logger_helper import LoadLogger
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import trange


def batch_generator(X, y, batch_size=128):
  nr_batches = X.shape[0] // batch_size
  
  for i in range(nr_batches):
    X_batch = None
    y_batch = None
    if i == nr_batches - 1:
      X_batch = X[(i * batch_size) : X.shape[0], :]
      y_batch = y[(i * batch_size) : y.shape[0]].reshape(-1,1)
    else:
      X_batch = X[(i * batch_size) : (i+1) * batch_size, :]
      y_batch = y[(i * batch_size) : (i+1) * batch_size].reshape(-1,1)
    
    yield X_batch, y_batch


if __name__ == '__main__':
  logger = LoadLogger(lib_name='TFTEST', config_file='config.txt')

  file_df_header = os.path.join(logger._base_folder, '_data/T2016C2017.xlsx')
  file_df = os.path.join(logger._base_folder, '_data/_DS_CHURN_TRAN_2016.csv')
  
  
  df_header = pd.read_excel(file_df_header)
  header = list(df_header.columns)
  
  logger.P("Reading churn dataset..")
  df = pd.read_csv(file_df, names=header)
  logger.P("Dataset in memory.", show_time=True)
  
  logger.P("Filling missing values..")
  df.fillna(0, inplace=True)
  logger.P("Missing values filled.", show_time=True)
  
  X = df.loc[:, "R1":"ZDROVITAL4"].values
  y = df.loc[:, "CHURN"].values
  nr_feats = X.shape[1]

  logger.P("There are {:,} clients. Each client is described by {} attributes.".format(X.shape[0], X.shape[1]))
  
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=33)
  
  tf_graph = tf.Graph()
  
  nr_hidden_1 = 168
  nr_hidden_2 = 84
  
  
  with tf_graph.as_default():
    
    tf_X_batch = tf.placeholder(dtype=tf.float32, shape=[None, nr_feats], name='x_batch')
    tf_y_batch = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_batch')
    
    tf_W1 = tf.Variable(initial_value=tf.truncated_normal([nr_feats, nr_hidden_1], 
                                                          stddev=1.0 / np.sqrt(nr_hidden_1)),
                        dtype=tf.float32,
                        name='W1')
    tf_bias_1 = tf.Variable(initial_value=tf.zeros(shape=[1, nr_hidden_1]), name='bias1')
    
    z1 = tf.add(tf.matmul(tf_X_batch, tf_W1), tf_bias_1, name='z1')
    a1 = tf.nn.relu(z1, name='a1')

    tf_W2 = tf.Variable(initial_value=tf.truncated_normal([nr_hidden_1, nr_hidden_2], 
                                                          stddev=1.0 / np.sqrt(nr_hidden_2)),
                        dtype=tf.float32,
                        name='W2')
    
    tf_bias_2 = tf.Variable(initial_value=tf.zeros(shape=[1, nr_hidden_2]), name='bias2')
    
    z2 = tf.add(tf.matmul(a1, tf_W2), tf_bias_2, name='z2')
    a2 = tf.nn.relu(z2, name='a2')
    
    tf_W3 = tf.Variable(initial_value=tf.truncated_normal([nr_hidden_2, 1]),
                        dtype=tf.float32,
                        name='W3')
    tf_bias_3 = tf.Variable(initial_value=tf.zeros(shape=[1, 1]), name='bias3')
    
    z3 = tf.add(tf.matmul(a2, tf_W3), tf_bias_3, name='z3')
    
    tf_y_hat = tf.nn.sigmoid(z3)
    tf_y_hat = tf_y_hat >= 0.5
    tf_y_hat = tf.cast(tf_y_hat, tf.float32)
    tf_correct_prediction = tf.equal(tf_y_hat, tf_y_batch)
    tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, tf.float32), name='accuracy')

    tf_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf_y_batch,
          logits=z3,
          name='loss'
    ))

    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    
    train_op = optimizer.minimize(tf_loss)
    
    init_op = tf.global_variables_initializer()
  #end-graph-definition
  
  epochs = 25
  batch_size = 128

  session = tf.Session(graph=tf_graph)
  session.run(init_op)
  
  history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
  for epoch in range(epochs):
    logger.P("Epoch {}/{}".format(epoch+1, epochs))
    nr_batches = X.shape[0] // batch_size
    t = trange(nr_batches, desc='', leave=True)
    epoch_loss = 0
    epoch_accuracy = 0
    for i in t:
      X_batch = None
      y_batch = None
      if i == nr_batches - 1:
        X_batch = X[(i * batch_size) : X.shape[0], :]
        y_batch = y[(i * batch_size) : y.shape[0]].reshape(-1,1)
      else:
        X_batch = X[(i * batch_size) : (i+1) * batch_size, :]
        y_batch = y[(i * batch_size) : (i+1) * batch_size].reshape(-1,1)
      
      feed_dict = {tf_X_batch: X_batch, tf_y_batch: y_batch}
      _, good_loss, acc = session.run([train_op, tf_loss, tf_accuracy], feed_dict=feed_dict)

      epoch_loss += good_loss
      epoch_accuracy += acc

      t.set_description("Loss {:.3f}; Acc {:.2f}%".format(good_loss, acc * 100))
      t.refresh()
    #endfor
    history['loss'].append(epoch_loss / nr_batches)
    history['acc'].append(100.0 * epoch_accuracy / nr_batches)
  #endfor
