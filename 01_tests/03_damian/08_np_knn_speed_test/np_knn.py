# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 08:18:02 2018

@author: Andrei Ionut Damian
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import threading
from time import time

np.__config__.show()


class ThreadedKNN(threading.Thread):
  def __init__(self, X_train, y_train, X_test, y_test,
              threadID = -1, name = 'KNNThread',
              threadLock = None
              ):
    self.threadLock = threadLock
    threading.Thread.__init__(self)
    if threadID == -1:
      threadID = str(time()).replace(".","")
      
    self.threadID = str(threadID)
    self.name = name + ' ' + self.threadID
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    return
  
  def log(self, _str):
    if self.threadLock is not None:
      self.threadLock.acquire()
    print(_str, flush=True)
    if self.threadLock is not None:
      self.threadLock.release()
    return
    
    
  def run(self):
    self.log("Starting " + self.name)
    y_hat_list = []
    acc_list = []
    test_size = self.X_test.shape[0]
    for obs in range(test_size):
      x = self.X_test[obs]
      dists = norm(self.X_train - x, axis = 1, ord = 2)
      y_hat = y_train[np.argmin(dists)]
      y_hat_list.append(y_hat)
      acc_list.append(y_hat == self.y_test[obs])
      acc = sum(acc_list) / float(len(acc_list))
      self.log(" {} completed {:.2f}% with acc {:.2f}%".format(
          self.name, obs/test_size*100, acc*100))
    self.log("Exiting " + self.name)
    self.y_preds = y_hat_list
    self.acc = acc
    return self.y_preds


def compute_knn(x_train, x_test, y_train, y_test):
  assert len(x_train.shape) == 2
  assert len(x_test.shape) == 2
  assert len(y_test.shape) == 1
  assert len(y_train.shape) == 1
  nr_obs = x_test.shape[0]
  y_res = []
  tbar = tqdm(range(nr_obs))
  for obs in tbar:
    x = x_test[obs]
    dists = norm(x_train - x, axis = 1, ord = 2)
    y_hat = y_train[np.argmin(dists)]
    y_res.append(y_hat == y_test[obs])
    acc = sum(y_res) / float(len(y_res))
    tbar.set_description("Acc: {:.2f}".format(acc*100))
  return y_res

def compute_knn_mp(x_train, x_test, y_train, y_test, 
                   nr_threads = 4,
                   limit_batch = None):
  assert len(x_train.shape) == 2
  assert len(x_test.shape) == 2
  assert len(y_test.shape) == 1
  assert len(y_train.shape) == 1
  print('Start MP KNN')
  nr_obs = x_test.shape[0]
  if limit_batch is None:
    thread_data_size = nr_obs // nr_threads
  else:
    thread_data_size = limit_batch // nr_threads
  thrLock = threading.Lock()
  threads = []
  for ithr in range(nr_threads):
    X_tr = x_train
    y_tr = y_train
    X_ts = x_test[ithr*thread_data_size:ithr*thread_data_size + thread_data_size]
    y_ts = y_test[ithr*thread_data_size:ithr*thread_data_size + thread_data_size]
    c_thread = ThreadedKNN(X_tr, y_tr, X_ts, y_ts, threadID=ithr,
                           threadLock=thrLock)
    c_thread.start()
    threads.append(c_thread)
  for thr in threads:
    thr.join()
  y_preds = []
  print('\nAggregating results...')  
  for thr in threads:
    y_preds = y_preds + thr.y_preds
  print('Done MP KNN')  
  return y_preds  
  
print("loading mnist")
mnist = fetch_mldata('mnist-original')
data = mnist.data / 255
print("Split")
X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size=0.1, random_state=1234)

print("compute knn")
limit_size = 100
y_pred = compute_knn_mp(X_train, X_test, y_train, y_test,
                        limit_batch = limit_size)
y_test = y_test[:limit_size]
acc = np.sum(y_pred == y_test) / y_test.shape[0]

print(acc)
