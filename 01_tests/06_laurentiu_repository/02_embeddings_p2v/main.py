import pandas as pd
from time import time
import os
import numpy as np
from p2v_embeddings import RecomP2VEmbeddings



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
  

def load_transactions(base_folder, pickle_file):

  filename = os.path.join(base_folder, pickle_file)
  print('Loading all transactions dataset: {} ... '.format(filename))
  start = time()
  with open(filename, "rb") as fp:
    import pickle
    data = pickle.load(fp)
  print('Loaded {:,} transactions in {:.2f}s'.format(data.shape[0], time() - start))

  return data



if __name__ == '__main__':
  ### Data fetch and exploration
  base_folder = "D:/Google Drive/_hyperloop_data/recom_compl_2014_2017/_data"
  tran_folder = "winter"
  app_folder = os.path.join(base_folder, tran_folder)
  
  data = load_transactions(base_folder = app_folder,
                           pickle_file = 'ordered_trans.p')
  data = data - 1 # Index newids from 0

  prods_filename = os.path.join(app_folder, 'ITEMS.csv')
  print('Loading products dataset: {} ... '.format(prods_filename[-30:]))
  start = time()
  df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))

  newids = np.array(df_prods['IDE'].tolist()) - 1
  newids = list(newids)
  ids = df_prods['ITEM_ID'].tolist()
  names = df_prods['ITEM_NAME'].tolist()
  id2new_id = dict(zip(ids, newids))
  new_id2prod = dict(zip(newids, names))

  architecture = 'CBOW'
  context_window = 3
  X_train, y_train = generate_full_batch(data = data,
                                         window = context_window,
                                         architecture = architecture)

  r = RecomP2VEmbeddings(nr_products = len(id2new_id),
                         config_file = 'tf_config.txt',
                         context_window = context_window,
                         architecture = architecture)
  #r.Fit(X_train = X_train, y_train = y_train, epochs = 20, batch_size = 64)
  

  from recom_maps_utils import tsne, create_figures
  tsne_nr_products = None   # None means 'all' :D
  norm_embeddings = True
  do_3D = False

  lowest_newid = min(newids)
  additional_name_figure = '_TSNE'
  if norm_embeddings:
    r.NormalizeEmbeddings()
    additional_name_figure = '_NORMEMB_TSNE'
  
  embeddings = r.GetEmbeddings(norm_embeddings = norm_embeddings)
  y_kmeans = r.GetKMeansClusters(norm_embeddings = norm_embeddings)
  
  """
  low_dim_embs2D = tsne(embeddings,
                        dim = 2,
                        n_iter = 2000,
                        indexed_from = lowest_newid,
                        nr_products = tsne_nr_products)
  if do_3D:
    low_dim_embs3D = tsne(embeddings,
                          dim = 3,
                          n_iter = 2000,
                          indexed_from = lowest_newid,
                          nr_products = tsne_nr_products)


  title = "[P2V] t-SNE visualization of {:,} products whose {}-d embeddings " \
          "are trained during {} epochs.".format(28377, 64, 15)
  if r.CONFIG["LOAD_MODEL"] == "":
    fig_name = r.model_name + additional_name_figure
  else:
    fig_name = r.CONFIG["LOAD_MODEL"][8:-3] + additional_name_figure
  tsne_folder = os.path.join(r._base_folder, '_tsne')
  create_figures(name = fig_name,
                 title = title,
                 size = 20000,
                 low_dim_embs = low_dim_embs2D,
                 y_kmeans = y_kmeans[:tsne_nr_products],
                 dict_labels = new_id2prod,
                 indexed_from = lowest_newid,
                 nr_labels = tsne_nr_products,
                 tsne_folder = tsne_folder)
  
  print("Saving low_dim_embs ...")
  np.save(os.path.join(tsne_folder, fig_name + '_Coords2D.npy'), low_dim_embs2D)
  if do_3D:
    np.save(os.path.join(tsne_folder, fig_name + '_Coords3D.npy'), low_dim_embs3D)
  print("low_dim_embs saved.")
  """