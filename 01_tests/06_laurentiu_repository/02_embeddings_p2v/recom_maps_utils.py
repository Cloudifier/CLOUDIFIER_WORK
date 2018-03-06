# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.manifold import TSNE
import os
import numpy as np

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


def ProcessModel(model, dict_labels,
                 tsne_nr_products = None, compute_norm_embeddings = True,
                 do_tsne_3D = False, lowest_id = 0, tsne_iter = 2000):
  
  additional_name_figure = '_TSNE'
  if compute_norm_embeddings:
    model.NormalizeEmbeddings()
    additional_name_figure = '_NORMEMB_TSNE'
  
  embeddings = model.GetEmbeddings(norm_embeddings = compute_norm_embeddings)
  y_kmeans = model.GetKMeansClusters(norm_embeddings = compute_norm_embeddings)

  dict_model_results = {}
  dict_model_results['embeddings'] = embeddings
  dict_model_results['y_kmeans'] = y_kmeans
  low_dim_embs2D = tsne(embeddings,
                        dim = 2,
                        n_iter = tsne_iter,
                        indexed_from = lowest_id,
                        nr_products = tsne_nr_products)
  dict_model_results['tsne2d'] = low_dim_embs2D
  if do_tsne_3D:
    low_dim_embs3D = tsne(embeddings,
                          dim = 3,
                          n_iter = tsne_iter,
                          indexed_from = lowest_id,
                          nr_products = tsne_nr_products)
    dict_model_results['tsne3d'] = low_dim_embs3D


  title = "[P2V] t-SNE visualization of hyp products trained using {}-d embeddings.".format(128)
  if model.CONFIG["LOAD_MODEL"] == "":
    fig_name = model.model_name + additional_name_figure
  else:
    fig_name = model.CONFIG["LOAD_MODEL"][8:-3] + additional_name_figure
  tsne_folder = os.path.join(model._base_folder, '_tsne')
  create_figures(name = fig_name,
                 title = title,
                 size = 20000,
                 low_dim_embs = low_dim_embs2D,
                 y_kmeans = y_kmeans[:tsne_nr_products],
                 dict_labels = dict_labels,
                 indexed_from = lowest_id,
                 nr_labels = tsne_nr_products,
                 tsne_folder = tsne_folder)

  print("Saving low_dim_embs ...")
  np.save(os.path.join(tsne_folder, fig_name + '_Coords2D.npy'), low_dim_embs2D)
  if do_tsne_3D:
    np.save(os.path.join(tsne_folder, fig_name + '_Coords3D.npy'), low_dim_embs3D)
  print("low_dim_embs saved.")
  return dict_model_results