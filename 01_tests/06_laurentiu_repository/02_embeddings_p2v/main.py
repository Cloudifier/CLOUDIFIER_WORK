import pandas as pd
from time import time
import os
import numpy as np
from p2v_embeddings import RecomP2VEmbeddings
from recom_maps_utils import ProcessModel


if __name__ == '__main__':
  ### Data fetch and exploration
  base_folder = "D:/Google Drive/_hyperloop_data/recom_compl_2014_2017/_data"
  tran_folder = "summer"
  app_folder = os.path.join(base_folder, tran_folder)
  
  
  trans_dataset_skip = np.load(os.path.join(app_folder, 'trans_skip.npz'))
  print('Loading training dataset ...')
  start = time()
  train = trans_dataset_skip['train']
  X = train[:, 0].reshape(-1, 1)
  y = train[:, 1].reshape(-1, 1)
  end = time()
  print('Dataset loaded in {:.2f}s.'.format(end - start))
  
  
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


  architecture = 'SKIP-GRAM'
  context_window = 2
  r = RecomP2VEmbeddings(nr_products = len(id2new_id),
                         config_file = 'tf_config.txt',
                         nr_embeddings = 128,
                         context_window = context_window,
                         architecture = architecture)
  #r.Fit(X_train = X, y_train = y, epochs = 30, batch_size = 128)
  
  lowest_id = min(newids)
  dict_model_results1 = ProcessModel(r, new_id2prod,
                                     tsne_nr_products = None,
                                     compute_norm_embeddings = True,
                                     do_tsne_3D = False,
                                     lowest_id = lowest_id)

  dict_model_results2 = ProcessModel(r, new_id2prod,
                                     tsne_nr_products = None,
                                     compute_norm_embeddings = False,
                                     do_tsne_3D = False,
                                     lowest_id = lowest_id)
  
  