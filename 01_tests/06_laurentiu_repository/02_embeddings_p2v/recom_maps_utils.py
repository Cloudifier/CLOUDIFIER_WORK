# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.manifold import TSNE
import os

def plot_kmeans_clusters(title, size, low_dim_embs, labels, y_kmeans, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  dpi = 100
  size /= dpi
  plt.figure(figsize = (size, size), dpi = dpi)
  plt.title(title)
  plt.subplots_adjust(bottom = 0.1)
  X = low_dim_embs[:,0]
  Y = low_dim_embs[:,1]
  plt.scatter(X, Y, marker = 'o', c = y_kmeans, cmap = plt.get_cmap('jet'))

  for label, x, y in zip(labels, X, Y):
    plt.annotate(
      label,
      xy = (x, y), xytext = (5, 2),
      textcoords = 'offset points', ha = 'right', va = 'bottom')
  plt.savefig(filename)
  return

def tsne(embeddings, dim = 2, n_iter = 1500, indexed_from = 1,
         nr_products = None, method = None, verbose = 2):

  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + 'Starting TSNE algorithm ...')
  
  if method is None:
    tsne = TSNE(perplexity = 45,
                n_components = dim,
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
    embeddings = embeddings[indexed_from:nr_products + indexed_from, :]
  else:
    embeddings = embeddings[indexed_from:, :]

  low_dim_embs = tsne.fit_transform(embeddings)

  strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
  print(strnowtime + "Finished TSNE. Created low_dim_embs.")
  return low_dim_embs


def create_figures(name, title, size, low_dim_embs, y_kmeans, dict_labels, indexed_from = 1,
                   nr_labels = None, tsne_folder = '.'):

  filename = os.path.join(tsne_folder, name + '.png')
  if nr_labels is None:
    plot_only = len(dict_labels)
  else:
    plot_only = nr_labels

  labels = [dict_labels[i][:10] for i in range(indexed_from, plot_only + indexed_from)]
  plot_kmeans_clusters(title, size, low_dim_embs, labels, y_kmeans, filename)
  return

