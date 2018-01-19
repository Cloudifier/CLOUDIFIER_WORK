import pandas as pd
from time import time
import os
import numpy as np
from p2v_embeddings import ProdSym, RecomP2VEmbeddings
import sys
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.manifold import TSNE

__VERBOSE__ = False

'''
title = "[P2V] t-SNE visualization of {:,} products whose {}-d embeddings " \
          "are trained during {} epochs.".format(10000, 64, 5)
'''

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def generate_full_batch(data, window, architecture = 'CBOW'):
  span = 2 * window + 1

  shape = (data.size - span + 1, span)
  strides = (data.itemsize, data.itemsize)
  batch = np.lib.stride_tricks.as_strided(data, shape = shape, strides = strides)
  batch = np.delete(batch, window, 1)

  start = window
  end = -1 * window
  labels = (data[start:end]).reshape(-1,1)
  
  if architecture == 'SKIP-GRAM':
    tmp = np.array(batch)
    batch = np.array(labels)
    labels = np.array(tmp)

  return batch, labels
  

def load_transactions(csv_file = 'SORTED_TRAINS.csv', pickle_file = 'ordered_trans.p'):
  base_folder = "D:/Google Drive/_hyperloop_data/recom_compl/_data"

  if __VERBOSE__:
    filename = os.path.join(base_folder, csv_file)
    print('Se incarca in memorie setul de date cu tranzactii: {} ... '.format(filename))
    start = time()
    df = pd.read_csv(filename)
    end = time()
    valid_columns = ['item_id', 'cust_id', 'dt']
    df = df[valid_columns]
    print('S-a incarcat setul de date in {:.2f}s .. df = {}'.format(end - start, list(df.columns)))
    print('Exista {:,} tranzactii in perioada {} - {}'.format(len(df), df.iloc[0]['dt'], df.iloc[-1]['dt']))
  
    num_products = df.item_id.max()
    print('Numarul produselor este {:,}, iar numarul clientilor in aceasta perioada este {:,}'
          .format(num_products, df['cust_id'].unique().shape[0]))
    data = np.array(df['item_id'].tolist()) - 1
  else:
    filename = os.path.join(base_folder, pickle_file)
    print('Se incarca in memorie setul de date cu tranzactii: {} ... '.format(filename))
    start = time()
    with open(filename, "rb") as fp:
      import pickle
      data = pickle.load(fp)
    end = time()
    print('S-a incarcat setul de date in {:.2f}s'.format(end - start))
    print('Exista {:,} tranzactii!'.format(data.shape[0]))

  return data


def plot_kmeans_clusters(title, size, low_dim_embs, labels, y_kmeans, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  dpi = 100
  size /= dpi
  plt.figure(figsize=(size, size), dpi = dpi)
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

def tsne(embeddings, nr_products = None, method = None, verbose = 2):
  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + 'START TSNE ALGORITHM')
  n_iter = 5000
  
  if method is None:
    tsne = TSNE(perplexity = 45,
                n_components = 2,
                init = 'pca',
                n_iter = n_iter, 
                random_state = 33,
                verbose = verbose)
  else:
    try:
      tsne = TSNE(perplexity = 45,
                  n_components = 2,
                  init = 'pca',
                  n_iter = n_iter, 
                  random_state = 33,
                  verbose = verbose,
                  method = method)
    except:
      print('[TSNE] Error: Gradient optimization method not found')

  
  if nr_products is not None:
    embeddings = embeddings[:nr_products, :]
  
  low_dim_embs = tsne.fit_transform(embeddings)
  
  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + "Created low_dim_embs")
  return low_dim_embs


def create_figures(name, title, size, low_dim_embs, y_kmeans, dict_labels, nr_labels = None,
                   tsne_folder = '.'):
  filename = os.path.join(tsne_folder, name + '.png')
  if nr_labels is None:
    plot_only = len(dict_labels)
  else:
    plot_only = nr_labels
  
  labels = [dict_labels[i][:10] for i in range(plot_only)]
  plot_kmeans_clusters(title, size, low_dim_embs, labels, y_kmeans, filename)

if __name__ == '__main__':
  ### Data fetch and exploration
  if len(sys.argv) > 1:
    data = load_transactions(sys.argv[1], sys.argv[2])
  else:
    data = load_transactions()

  prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_compl/_data/ITEMS.csv")
  print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
  start = time()
  df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
  end = time()
  print('S-a incarcat setul de date in {:.2f}s .. top 5 produse:\n\n{}\n'.format(end - start, df_prods.head(5)))

  newids = np.array(df_prods['NEW_ID'].tolist()) - 1
  newids = list(newids)
  ids = df_prods['ITEM_ID'].tolist()
  names = df_prods['ITEM_NAME'].tolist()
  id2new_id = dict(zip(ids, newids))
  new_id2prod = dict(zip(newids, names))
  

  architecture = 'CBOW'
  context_window = 2
  X_train, y_train = generate_full_batch(data = data,
                                         window = context_window,
                                         architecture = architecture)

  r = RecomP2VEmbeddings(nr_products = len(id2new_id),
                         config_file = 'tf_config.txt',
                         context_window = context_window,
                         architecture = architecture)
  #r.Fit(X_train = X_train, y_train = y_train, epochs = 15, batch_size = 512)
  
  r.NormalizeEmbeddings()
  r.CreateKMeansClusters()
  low_dim_embs = tsne(r.GetNormEmbeddings(), 10000)
  title = "[P2V] t-SNE visualization of {:,} products whose {}-d embeddings " \
          "are trained during {} epochs.".format(10000, 64, 15)
  create_figures(name = r.CONFIG["LOAD_MODEL"][8:-9] + '_TSNE',
                 title = title,
                 size = 20000,
                 low_dim_embs = low_dim_embs,
                 y_kmeans = r.GetNormKMeansClusters()[:10000],
                 dict_labels = new_id2prod,
                 nr_labels = 10000,
                 tsne_folder = os.path.join(r._base_folder, '_tsne'))
