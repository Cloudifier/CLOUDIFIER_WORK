import numpy as np
import tensorflow as tf
import os
from time import time
import pandas as pd

###
prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_v2/PROD.csv")
print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
start = time()
df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
end = time()
print('S-a incarcat setul de date in {:.2f}s .. top 5 prods:\n\n{}'.format(end - start, df_prods.head(5)))

ids = np.array(df_prods['NEW_ID'].tolist()) - 1
ids = list(ids)
dictionary = dict(zip(ids, df_prods['PROD_NAME'].tolist())) # id: prod_name
del df_prods
###

norm_embeddings = np.load('D:\\Google Drive\\_hyperloop_data\\recom_v2\\_MODELS\\emb64_epochs15_momentum\\norm_embeddings.npy')

x = tf.constant(norm_embeddings[:1000], dtype=tf.float32)
y = tf.constant(norm_embeddings[:1000], dtype=tf.float32)

x_ = tf.expand_dims(x, 0)
y_ = tf.expand_dims(y, 1)

suma = tf.add(x_, y_)

sess = tf.Session()
np_suma = sess.run(suma)

import itertools
a = list(itertools.permutations(enumerate(range(1000)), 2))
possible_indices = list((i,j) for ((i,_),(j,_)) in a)

nr_valid_operations = len(possible_indices)
valid_operations = np.zeros((nr_valid_operations, 64))
for i in range(nr_valid_operations):
  valid_operations[i] = np_suma[possible_indices[i]]
  

from sklearn.metrics import pairwise_distances
import multiprocessing
"""
distances = pairwise_distances(valid_operations, norm_embeddings[:1000], metric='cosine',n_jobs=multiprocessing.cpu_count())  

k = 5
top_k_indexes = np.argsort(distances, axis=1)[:,:k]
top_k_distances=  np.sort(distances,axis=1)[:,:k]

A = np.sort(top_k_distances.flatten())[::2]
A = A[:2000]

def find_op(small_dist, allow_dup=False):
  idx = np.where(top_k_distances == small_dist)
  results = top_k_indexes[idx]
  
  for i in range(idx[0].shape[0]):
    if results[i] in possible_indices[idx[0][i]]:
      if not allow_dup:
        break
      print("rezultat intre operanzi")
    
    t1,t2 = possible_indices[idx[0][i]]
    
    print("{} = {} + {}; dist={:.5f}"
          .format(dictionary[results[i]][:20], dictionary[t1][:20], dictionary[t2][:20], small_dist))
    break
"""