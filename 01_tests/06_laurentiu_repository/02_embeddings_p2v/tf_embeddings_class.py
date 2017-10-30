__author__     = ''
__copyright__  = 'Copyright 2017, Cloudifier'
__credits__    = []
__version__    = '1.0.0'
__maintainer__ = ''
__email__      = ''
__status__     = ''
__library__    = ''
__created__    = '2017-10-10'
__modified__   = '2017-10-11'
__lib__        = 'RECOM_V2'
__DEBUG__      = False
__DOTSNE__     = False


import os
import collections
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_object(obj_name):
  from importlib.machinery import SourceFileLoader
  import os
  
  home_dir = os.path.expanduser("~")
  valid_paths = [os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "Google Drive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("D:/", "GoogleDrive")]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    raise Exception("Couldn't find google drive folder!")

  utils_path = os.path.join(drive_path, "_pyutils")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, obj_name)).load_module()

  return logger_lib


class RecomSSBEmbeddings:
  def __init__(self, data, prod_dict, batch_size, context_window,
               embedding_size, epochs, model_name, decay=False, 
               optimization=None, graph_path=None, graph_name=None):

    self.save_folder = os.path.join("D:/Google Drive/_hyperloop_data/recom_v2/_MODELS", model_name)
    if not os.path.exists(self.save_folder):
      os.makedirs(self.save_folder)
    else:
      model_index = 1
      while os.path.exists(self.save_folder + '_v' + str(model_index)):
        model_index += 1
      self.save_folder += ('_v' + str(model_index))
      os.makedirs(self.save_folder)
    
    ## Creare foldere pt imagini tsne si pt checkpoint-uri
    self.tsne_folder = os.path.join(self.save_folder, 'tsne')
    self.checkpoints_folder = os.path.join(self.save_folder, 'checkpoints')
    if not os.path.exists(self.tsne_folder):
      os.makedirs(self.tsne_folder)
      
    if not os.path.exists(self.checkpoints_folder):
      os.makedirs(self.checkpoints_folder)
    
    self.logger = load_object("soft_logger.py").Logger(save_folder=self.save_folder)
    self.logger.lib = __lib__
    
    self.data_index_head = 0
    self.data_index_tail = 0
    self.batch_size = batch_size
    self.context_window = context_window
    self.embedding_size = embedding_size
    self.graph = tf.Graph()
    self.epochs = epochs
    self.model_name = model_name
    self.data = data
    self.dictionary = prod_dict
    self.num_products = len(self.dictionary)
    self.optimization = optimization
    self.decay = decay
  
    self.num_sampled = 64    # Number of negative examples to sample.
    
    if graph_path is None and graph_name is None:
      self.create_graph()
    else:
      self.restore_graph()

    self.graph_path = graph_path
    self.graph_name = graph_name
    self.global_step = 0

  def restore_graph(self):
    self.logger._log('Graful este incarcat din fisierul {} ..'
                     .format(os.path.join(self.graph_path, self.graph_name + '.meta')))
    
    self.saver = tf.train.import_meta_graph(os.path.join(self.graph_path, self.graph_name + '.meta'))
    self.session = tf.Session()
    self.saver.restore(self.session, tf.train.latest_checkpoint(self.graph_path))
    self.graph = tf.get_default_graph()
    
    self.tf_train_inputs = self.graph.get_tensor_by_name('train_inputs:0')
    self.tf_train_labels = self.graph.get_tensor_by_name('train_labels:0')
    self.tf_embeddings   = self.graph.get_tensor_by_name('embeddings:0')
    self.tf_loss         = self.graph.get_tensor_by_name('loss:0')
    self.tf_normalized_embeddings = self.graph.get_tensor_by_name('truediv:0')
    
    with self.graph.as_default():
      self.learning_rate = tf.placeholder(tf.float32, shape=[])
      self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.tf_loss)
      self.saver = tf.train.Saver()

  def create_graph(self):
    self.logger._log('Graful este in curs de initializare ..')
    
    with self.graph.as_default():
      self.tf_train_inputs = tf.placeholder(tf.int32,
                                            shape=[self.batch_size, self.context_window * 2],
                                            name='train_inputs')
      self.tf_train_labels = tf.placeholder(tf.int32,
                                            shape=[self.batch_size, 1],
                                            name='train_labels')

      # Look up embeddings for inputs.
      self.tf_embeddings = tf.Variable(tf.random_uniform([self.num_products, self.embedding_size], -1.0, 1.0),
                                       name='embeddings')
      tf_embed = tf.nn.embedding_lookup(self.tf_embeddings, self.tf_train_inputs)
      
      #tf_embed_context = tf.reshape(tf_embed, [self.batch_size, -1])
      tf_embed_context = tf.reduce_mean(tf_embed, 1)
      
      # Construct the variables for the sampled_softmax_loss 
      #tf_sm_weights = tf.Variable(tf.truncated_normal([self.num_products, self.embedding_size * self.context_window * 2],
      #                      stddev=1.0 / math.sqrt(self.embedding_size * self.context_window * 2)))
      tf_sm_weights = tf.Variable(tf.truncated_normal([self.num_products, self.embedding_size],
                            stddev=1.0 / math.sqrt(self.embedding_size)))
      tf_sm_biases = tf.Variable(tf.zeros([self.num_products]))

      self.tf_loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=tf_sm_weights,
                       biases=tf_sm_biases,
                       labels=self.tf_train_labels,
                       inputs=tf_embed_context,
                       num_sampled=self.num_sampled,
                       num_classes=self.num_products), name='loss')

      self.learning_rate = tf.placeholder(tf.float32, shape=[])
      self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).minimize(self.tf_loss)
      #self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.tf_loss)
      
      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(self.tf_embeddings), 1, keep_dims=True))
      self.tf_normalized_embeddings = self.tf_embeddings / norm ## name=truediv:0

      # Add variable initializer.
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
    

  def generate_batch(self):
    self.data_index_tail = self.data_index_head
    context_size = 2 * self.context_window
    batch = np.ndarray(shape=(self.batch_size, context_size), dtype=np.int32)
    labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
    span = 2 * self.context_window + 1  # [ context_window target context_window ]
    current_window = collections.deque(maxlen=span)
    for _ in range(span):
      current_window.append(self.data[self.data_index_tail])
      self.data_index_tail = (self.data_index_tail + 1) % len(self.data)
    for i in range(self.batch_size):
      buffer_list = list(current_window)
      labels[i, 0] = buffer_list.pop(self.context_window)
      batch[i] = buffer_list

      # next batch
      current_window.append(self.data[self.data_index_tail])
      self.data_index_tail = (self.data_index_tail + 1) % len(self.data)
      self.data_index_head = (self.data_index_head + 1) % len(self.data)
    return batch, labels

  def decay_lr(self):
    if self.global_step <= 100000:
      return 1.0
    elif self.global_step <= 250000:
      return 0.5
    elif self.global_step <= 1000000:
      return 0.25
    elif self.global_step <= 2500000:
      return 0.1
    else:
      return 0.05
  
  def train(self):
    self.logger._log('Se antreneaza un nou model, cu urmatorii param: batch_size: {}, window: {}, n_embeddings: {}, epochs: {}, model_name: {}'
                     .format(self.batch_size, self.context_window, self.embedding_size, self.epochs, self.model_name))
    self.logger._log('Datele referitoare la acest model sunt salvate in: {}'.format(self.save_folder))
    
    
    if self.graph_path is None and self.graph_name is None:
      self.session = tf.Session(graph=self.graph)
      self.session.run(self.init)

    nr_batches = (123892563 // self.batch_size) + 1
    average_loss = 0
    
    for epoch in range(self.epochs):
      for step in range(nr_batches):
        self.global_step = epoch * nr_batches + step
        learning_rate = 0.1
        if epoch >= 5:
            learning_rate = 0.05
        if self.decay is True:
          learning_rate = self.decay_lr()
        
        batch_inputs, batch_labels = self.generate_batch()
        feed_dict = {self.tf_train_inputs: batch_inputs,
                     self.tf_train_labels: batch_labels,
                     self.learning_rate: learning_rate}
        _, loss_val = self.session.run([self.optimizer, self.tf_loss], feed_dict=feed_dict)
        average_loss += loss_val
  
        if (step == 0) and (epoch == 0) and __DOTSNE__:
          self.logger._log('Se creeaza hartile TSNE cu produse LA INCEPUTUL ANTRENAMENTULUI .. ')
          norm_embeddings = self.tf_normalized_embeddings.eval(session=self.session)
          self.tsne(norm_embeddings=norm_embeddings, name='tsne_initial')
        ### endif
        
        if step % 20000 == 0:
          if step > 0:
            average_loss /= 20000
          # The average loss is an estimate of the loss over the last 2000 batches.
          self.logger._log('Epoca {}: Costul calculat la pasul {}: {:.2f}'.format(epoch, step, average_loss))
          average_loss = 0
        ### endif
        
        if (step == (nr_batches // 2)) and __DOTSNE__:
          self.logger._log('Se creeaza hartile TSNE la MIJLOCUL epocii {} .. '.format(epoch))
          norm_embeddings = self.tf_normalized_embeddings.eval(session=self.session)
          self.tsne(norm_embeddings=norm_embeddings, name='tsne_' + str(epoch) + '_middle')
        ### endif
      ### endfor - step
      
      norm_embeddings = self.tf_normalized_embeddings.eval(session=self.session)
      if __DOTSNE__:
        self.logger._log('Se creeaza hartile TSNE la FINALUL epocii {} .. '.format(epoch))
        self.tsne(norm_embeddings=norm_embeddings, name='tsne_' + str(epoch) + '_termination')
  
      # Create checkpoint
      # self.saver.save(self.session, os.path.join(self.checkpoints_folder, 'p2v-model'), global_step=epoch)
      np.save(os.path.join(self.checkpoints_folder, 'norm_embeddings_' + str(epoch) + '.npy'), norm_embeddings)

    ### endfor - epoch
    
    self.session.close()


  def plot_kmeans_clusters(self, low_dim_embs, labels, y_kmeans, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(40, 40), dpi=100)
    plt.subplots_adjust(bottom=0.1)
    X = low_dim_embs[:,0]
    Y = low_dim_embs[:,1]
    plt.scatter(X, Y, marker='o', c=y_kmeans,
      cmap=plt.get_cmap('jet'))
  
    for label, x, y in zip(labels, X, Y):
      plt.annotate(
        label,
        xy=(x, y), xytext=(5, 2),
        textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)
  
  def plotly_kmeans_clusters(self, low_dim_embs, labels, y_kmeans, filename='plotly_tsne.html'):
    try:
      import plotly.graph_objs as go
      from plotly.offline import plot
  
      trace = go.Scattergl(
        x=low_dim_embs[:, 0],
        y=low_dim_embs[:, 1],
        mode='markers',
        text=labels,
        marker=dict(
          color=y_kmeans,
          colorscale='Jet',
          showscale=True,
          line=dict(width=1)
        )
      )
      data = [trace]
      plot(data, config=dict(displayModeBar=False, showLink=False), filename=filename, auto_open=False)
    except ImportError:
      print('Please install plotly.')
  
  def tsne(self, norm_embeddings, name):      
    try:
      tsne = TSNE(perplexity=45, n_components=2, init='pca', n_iter=5000, method='exact',
                  random_state=42)
      plot_only = 1000
      low_dim_embs = tsne.fit_transform(norm_embeddings[:plot_only, :])
      labels1 = [self.dictionary[i][:10] for i in range(plot_only)]
      labels2 = [self.dictionary[i] for i in range(plot_only)]
  
      from sklearn.cluster import KMeans
      kmeans = KMeans(n_clusters=16,random_state=42)
      y_kmeans = kmeans.fit_predict(norm_embeddings)[:plot_only]
  
      filename1 = os.path.join(self.tsne_folder, name +'.png')
      filename2 = os.path.join(self.tsne_folder, 'plotly_' + name + '.html')
      self.plot_kmeans_clusters(low_dim_embs, labels1, y_kmeans, filename1)
      self.plotly_kmeans_clusters(low_dim_embs, labels2, y_kmeans, filename2)
      self.logger._log('Hartile sunt salvate in fisierele [...{}], respectiv [...{}]'
                       .format(filename1[49:], filename2[49:]))
    except ImportError:
      print('Please install sklearn, matplotlib, and scipy to show embeddings.')


if __name__ == '__main__':
  print('Cannot run library module!')