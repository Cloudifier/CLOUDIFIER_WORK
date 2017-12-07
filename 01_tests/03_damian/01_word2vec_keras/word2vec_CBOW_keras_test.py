# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:20:50 2017

@author: Andrei
"""

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import sys

import pickle

from keras.layers import Embedding, Dense, Flatten
from keras.models import Sequential




sfolder = 'D:\\_DatasetsRep\\Text8\\'
s_pickle = sfolder + "dataset.pkl"

# Download the data from the source website if necessary.

url = 'http://mattmahoney.net/dc/'

def maybe_download(folder, filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  target_file = folder + filename
  if not os.path.exists(target_file):
    filename, _ = urlretrieve(url + filename, target_file)
  else:
    filename = target_file
    
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_dataset(words):
  count = [['UNK', -1]]  
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  i = 0
  all_count = len(words)
  unk_count = 0
  print("Building {:,} words dataset".format(all_count), flush= True)
  for word in words:
    i +=1
    if((i%100000)==0):
      print("\rBuilding dataset {:.2f}%".format(i/all_count*100), end="", flush=True)
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index) # append token if known word or 0 (UNK) otherwise
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  print("\nBuilding dataset done.")
  return data, count, dictionary, reverse_dictionary
  
vocabulary_size = 50000

if os.path.exists(s_pickle):
  print("Trying to load pickle...", flush = True)
  data_touple = pickle.load(open(s_pickle, "rb"))
  data, count, dictionary, reverse_dictionary  = data_touple
  print("Data loaded.", flush = True)
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
else:   
  filename = maybe_download(sfolder, 'text8.zip', 31344016)
  words = read_data(filename)
  print('Data size %d' % len(words))
  
  #  Build the dictionary and replace rare words with UNK token.
  print("---------------------------------------------------")
  print(" CBOW example based on {} words dictionary".format(vocabulary_size))
  print("---------------------------------------------------")
  
  
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
  del words  # Hint to reduce memory.
  data_touple_w = (data, count, dictionary, reverse_dictionary)
  print("Saving pickle...", flush = True)
  pickle.dump(data_touple_w, open(s_pickle,"wb"))
  print("Done saving pickle.", flush = True)
 

cbow_data_index = 0
# Function to generate a training batch for the skip-gram model.
def generate_batch_CBOW(batch_size, num_words_to_consider, words_window):
    """
    batch_size is the actual batch
    num_words_to_consider - how many words to consider in each observation
    words_window - the window (left & right) where to draw the words from
    if num_words_to_consider == words_window*2 then each target will have 1 observation    
    """    
    global cbow_data_index
    assert num_words_to_consider % 2 == 0 # num must be left + right
    assert words_window <= num_words_to_consider
    assert (2 * num_words_to_consider) % words_window == 0 
    # calculate number of observations per target
    obs_per_target = (words_window*2) // (num_words_to_consider)
    assert batch_size % obs_per_target == 0
    
    batch = np.ndarray(shape=(batch_size,num_words_to_consider), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)    
    span = 2 * words_window + 1 # [ words_window target words_window ]
    buffer = collections.deque(maxlen=span)
    # first get all needed words
    for _ in range(span):
        buffer.append(data[cbow_data_index])
        cbow_data_index = (cbow_data_index + 1) % len(data)
    # now for all batches
    for i in range(batch_size // obs_per_target):
        observation = words_window  # target at the center of the buffer so skip this observation
        observations_to_avoid = [ observation ]
        for n_o in range(obs_per_target):
            # for each num_skip of current sub batch (current label/target)
            for j in range(num_words_to_consider):
                while observation in observations_to_avoid:
                    observation = random.randint(0, span - 1)
                batch[i*obs_per_target+n_o,j] = buffer[observation]
                observations_to_avoid.append(observation) # allready seen the observation
            labels[i*obs_per_target+n_o,0] = buffer[words_window]            
        # done observations
        buffer.append(data[cbow_data_index]) # append new char at the end (move the window)
        cbow_data_index = (cbow_data_index + 1) % len(data)
        
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_words, words_window in [(2, 1), (4, 2), (2,2)]:
    cbow_data_index = 0
    batch, labels = generate_batch_CBOW(batch_size=8, 
                                        num_words_to_consider=num_words, 
                                        words_window=words_window)
    print('\nwith num_words = %d and word_window = %d:' % (num_words, words_window))
    
    print('    batch:  ', [[reverse_dictionary[bi] for bi in batch[batch_no,:]] for batch_no in range(batch.shape[0])])
    print('    labels: ', [reverse_dictionary[li] for li in labels.reshape(8)])
    
def generate_dataset(num_samples, num_words, words_window):
  global cbow_data_index
  cbow_data_index = 0
  batch_data, batch_labels = generate_batch_CBOW(num_samples,num_words, words_window)
  return batch_data, batch_labels  

def test_dataset(X,y):
  global reverse_dictionary
  print("")
  for i in range(5):
    print("y[{}]='{}' X[{}]={} ".format(i,reverse_dictionary[y[i]],
                                        i,[reverse_dictionary[bi] for bi in X[i]]))
    
if __name__=="__main__":
  
  vocabulary_size = 50000
  batch_size = 128
  embedding_size = 128
  word_window = 2 
  num_words = word_window * 2
  
  X_train, y_train = generate_dataset(50000,num_words,word_window)
  y_train = y_train.ravel()
  
  test_dataset(X_train,y_train)
  
  resp = input("Continue to prepare model?")
  if resp.upper() != "Q":
  
    model = Sequential()
    
    embed_layer = Embedding(vocabulary_size,embedding_size, input_length=num_words)
    
    model.add(embed_layer)
    model.add(Dense(512,activation = "relu"))
    model.add(Dense(1,activation=None))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1)
    
    word_vects = embed_layer.get_weights()[0]
  
    num_points = 400
    
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    tsne_embeddings = tsne.fit_transform(word_vects[1:num_points+1, :])
    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
  
    pylab.figure(figsize=(15,15))  
    for i, label in enumerate(words):
      x, y = tsne_embeddings[i,:]
      pylab.scatter(x, y)
      pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom')
    pylab.show()
  
  
