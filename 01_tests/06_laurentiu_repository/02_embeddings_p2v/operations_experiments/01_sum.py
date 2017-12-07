import numpy as np
import tensorflow as tf
import os
from time import time
import pandas as pd

###
prods_filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_compl/_data/PROD.csv")
print('Se incarca in memorie setul de date cu produse: {} ... '.format(prods_filename))
start = time()
df_prods = pd.read_csv(prods_filename, encoding='ISO-8859-1')
end = time()
print('S-a incarcat setul de date in {:.2f}s.'.format(end - start))

ids = np.array(df_prods['NEW_ID'].tolist()) - 1
ids = list(ids)
dictionary = dict(zip(ids, df_prods['PROD_NAME'].tolist())) # id: prod_name
del df_prods
###

embeddings = np.load('D:/Google Drive/_hyperloop_data/recom_compl/models_comparison/04.full_softmax/embeddings.npy')

x = tf.constant(embeddings[:2000], dtype=tf.float32)
y = tf.constant(embeddings[:2000], dtype=tf.float32)

x_ = tf.expand_dims(x, 1)
y_ = tf.expand_dims(y, 0)

suma = tf.add(x_, y_)

sess = tf.Session()
np_suma = sess.run(suma)

import itertools
a = list(itertools.permutations(enumerate(range(2000)), 2))
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

def similar_prod(str1, str2):
  words1 = str1.split(' ')
  words2 = str2.split(' ')
  
  intersection = list(set(words1) & set(words2))
  
  if len(intersection) >= 2:
    return True
  
  return False

def find_op(small_dist):
  idx = np.where(top_k_distances == small_dist)
  results = top_k_indexes[idx]
  
  sum_idx = idx[0][0]
  identic_names = False
  
  t1, t2 = possible_indices[sum_idx]
  prod_result = dictionary[results[0]]
  prod_t1 = dictionary[t1]
  prod_t2 = dictionary[t2]
  
  identic_names = similar_prod(prod_result, prod_t1) | similar_prod(prod_result, prod_t2) | similar_prod(prod_t1, prod_t2)
  
  if not identic_names:
    print("{} = {} + {}; dist={:.5f}".format(prod_result,
                prod_t1,
                prod_t2,
                small_dist))
"""