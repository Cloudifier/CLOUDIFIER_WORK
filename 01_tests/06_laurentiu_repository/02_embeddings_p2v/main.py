import pandas as pd
from time import time
import os
import numpy as np
from tf_embeddings_class import RecomSSBEmbeddings

__VERBOSE__ = False

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__



if __name__ == '__main__':
  ### Data fetch and exploration
  if __VERBOSE__:
    filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_v2/SORTED_TRANS.csv")
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
    del df
  else:
    filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_v2/CLEANED_DATA_TRANS.p")
    print('Se incarca in memorie setul de date cu tranzactii: {} ... '.format(filename))
    start = time()
    with open(filename, "rb") as fp:
      import pickle
      data = pickle.load(fp)
    end = time()
    print('S-a incarcat setul de date in {:.2f}s'.format(end - start))
    print('Exista {:,} tranzactii!'.format(data.shape[0]))
  
  data = list(data)

  prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_v2/PROD.csv")
  print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
  start = time()
  df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
  end = time()
  print('S-a incarcat setul de date in {:.2f}s .. top 5 produse:\n\n{}\n'.format(end - start, df_prods.head(5)))

  ids = np.array(df_prods['NEW_ID'].tolist()) - 1
  ids = list(ids)
  dictionary = dict(zip(ids, df_prods['PROD_NAME'].tolist())) # id: prod_name
  del df_prods
  ###

  architectures = dotdict({
    'context_window': [3],
    'embedding_size': [64],
    'batch_size': [128],
    'model_name': ['emb64_epochs4_momentum'],
    'epochs': [4]
  })

  for i in range(len(architectures.context_window)):
    r = RecomSSBEmbeddings(
      data = data,
      prod_dict=dictionary,
      batch_size=architectures.batch_size[i],
      context_window=architectures.context_window[i],
      embedding_size=architectures.embedding_size[i],
      epochs=architectures.epochs[i],
      model_name=architectures.model_name[i])

    r.train()

    del r