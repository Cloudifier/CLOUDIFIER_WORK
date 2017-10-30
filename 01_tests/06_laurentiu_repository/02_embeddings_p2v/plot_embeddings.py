import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from time import time
import os
import tensorflow as tf
from datetime import datetime as dt

'''
def tsne(norm_embeddings, name, tsne_folder='.'):
  try:
    # pylint: disable=g-import-not-at-top
    strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(strnowtime + 'START TSNE + KMEANS + CREATE FIGS')
    tsne = TSNE(perplexity=45, n_components=2, init='pca', n_iter=5000,
                method='exact', random_state=42)
    plot_only = 1000
    low_dim_embs = tsne.fit_transform(norm_embeddings[:plot_only, :])
    strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(strnowtime + "Created low_dim_embs")
    labels1 = [dictionary[i][:10] for i in range(plot_only)]
    labels2 = [dictionary[i] for i in range(plot_only)]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=16, random_state=42)
    y_kmeans = kmeans.fit_predict(norm_embeddings)[:plot_only]
    strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(strnowtime + "Created clusters")

    filename1 = os.path.join(tsne_folder, name + '.png')
    filename2 = os.path.join(tsne_folder, 'plotly_' + name + '.html')
    plot_kmeans_clusters(low_dim_embs, labels1, y_kmeans, filename1)
    strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(strnowtime + "Created figure")
    plotly_kmeans_clusters(low_dim_embs, labels2, y_kmeans, filename2)
    strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    print(strnowtime + "Created plotly")
    
    return low_dim_embs
  except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
'''

'''
title = "[P2V] t-SNE visualization of {:,} products whose {}-d embeddings " \
          "are trained during {} epochs.".format(10000, 64, 5)
'''

###
prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_compl/PROD.csv")
print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
start = time()
df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
end = time()
print('S-a incarcat setul de date in {:.2f}s .. top 5 prods:\n\n{}'.format(end - start, df_prods.head(5)))

ids = np.array(df_prods['NEW_ID'].tolist()) - 1
ids = list(ids)
dictionary = dict(zip(ids, df_prods['PROD_NAME'].tolist())) # id: prod_name
#del df_prods
###

def plot_kmeans_clusters(title, size, low_dim_embs, labels, y_kmeans, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(size, size), dpi=100)
  plt.title(title)
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

def plotly_kmeans_clusters(low_dim_embs, labels, y_kmeans, filename='plotly_tsne.html'):
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

    
def tsne(norm_embeddings, nr_prods=None, method=None):
  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + 'START TSNE ALGORITHM')
  
  if method is None:
    tsne = TSNE(perplexity=45, n_components=2, init='pca', n_iter=5000, 
                random_state=42, verbose=2)
  else:
    try:
      tsne = TSNE(perplexity=45, n_components=2, init='pca', n_iter=5000, 
                  random_state=42, verbose=2, method=method)
    except:
      print('[TSNE] Error: Gradient optimization method not found')

  
  if nr_prods is not None:
    norm_embeddings = norm_embeddings[:nr_prods, :]
  
  low_dim_embs = tsne.fit_transform(norm_embeddings)
  
  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + "Created low_dim_embs")
  
  return low_dim_embs

def create_clusters(norm_embeddings, nr_prods=None):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=16, random_state=42)
  y_kmeans = kmeans.fit_predict(norm_embeddings)
  
  if nr_prods is not None:
    y_kmeans = y_kmeans[:nr_prods]
    
  return y_kmeans

def create_figures(name, title, size, low_dim_embs, y_kmeans, nr_labels=None, tsne_folder='.'):
  filename1 = os.path.join(tsne_folder, name + '.png')
  filename2 = os.path.join(tsne_folder, 'plotly_' + name + '.html')
  
  if nr_labels is None:
    plot_only = len(dictionary)
  else:
    plot_only = nr_labels
  
  labels1 = [dictionary[i][:10] for i in range(plot_only)]
  labels2 = [dictionary[i] for i in range(plot_only)]
  
  plot_kmeans_clusters(title, size, low_dim_embs, labels1, y_kmeans, filename1)
  plotly_kmeans_clusters(low_dim_embs, labels2, y_kmeans, filename2) 


saver = tf.train.import_meta_graph(os.path.join('D:/Google Drive/_hyperloop_data/recom_compl/_MODELS/emb64_epochs15_momentum/checkpoints', 'p2v-model-14' + '.meta'))
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('D:/Google Drive/_hyperloop_data/recom_compl/_MODELS/emb64_epochs15_momentum/checkpoints/'))
graph = tf.get_default_graph()

# [n.name for n in tf.get_default_graph().as_graph_def().node]
tf_norm_embeddings = graph.get_tensor_by_name('truediv:0')
tf_embeddings = graph.get_tensor_by_name('embeddings:0')

with sess.as_default():
  norm_embeddings = tf_norm_embeddings.eval()
  embeddings = tf_embeddings.eval()
