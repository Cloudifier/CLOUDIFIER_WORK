import pandas as pd
from time import time
import os
import numpy as np

filename = os.path.join("D:/", "Google Drive/_hyperloop_data/recom_compl/CLEANED_DATA_TRANS.p")
print('Se incarca in memorie setul de date cu tranzactii: {} ... '.format(filename))
start = time()
with open(filename, "rb") as fp:
  import pickle
  data = pickle.load(fp)
end = time()
print('S-a incarcat setul de date in {:.2f}s'.format(end - start))
print('Exista {:,} tranzactii!'.format(data.shape[0]))

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

def generate_full_batch(data):
  def rolling(window):
    shape = (data.size - window + 1, window)
    strides = (data.itemsize, data.itemsize)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
  
  batch = rolling(5)
  batch = np.delete(batch, 2, 1)
  labels = (data[2:-2]).reshape(-1,1)
  
  return batch, labels

X,y = generate_full_batch(data)

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

emb_layer = Embedding(len(dictionary), 64, input_length=4)

model = Sequential()
model.add(emb_layer)
model.add(Flatten())
model.add(Dense(len(dictionary), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X,y,batch_size=16384, epochs=4)