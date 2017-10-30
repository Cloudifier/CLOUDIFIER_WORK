import numpy as np
from sklearn.model_selection import train_test_split
import os
from time import time as tm
import platform
from importlib.machinery import SourceFileLoader
from random import randint
import matplotlib.pyplot as plt
import pandas as pd

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "base.py")).load_module()
  logger = logger_lib.Logger(lib='DNN')

  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)

  X = mnist.data
  y = mnist.target

  #X = min_max_scaler(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)
  X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                test_size=0.5,
                                                                random_state=42)

  return dotdict({'train': (X_train, y_train),
                  'test': (X_test, y_test),
                  'validation': (X_validation, y_validation)})

def create_scene(big_dim, small_dim, small_img):
  small_img = small_img.reshape(small_dim[0], small_dim[1])
  pos_X = randint(0, big_dim[1] - small_dim[1])
  pos_Y = randint(0, big_dim[0] - small_dim[0])
  position = (pos_X, pos_Y)

  big_img = np.zeros((big_dim[0], big_dim[1]))
  big_img[pos_Y:pos_Y + small_dim[1], pos_X:pos_X + small_dim[0]] = small_img
  return big_img.ravel(), position

def create_new_test_dataframe(X_test, y_test, big_dim, path_to_save):
  H = big_dim[0]
  W = big_dim[1]

  X_test_reshaped = np.zeros((X_test.shape[0], H*W), dtype=int)
  positions = list()

  for i in range(X_test.shape[0]):
    if i % 500 == 0:
      print("Iteration #{}".format(i))
    big_img, position = create_scene(big_dim, (28, 28), X_test[i])
    X_test_reshaped[i] = big_img
    positions.append(position)
  print("Done creating {}x{} scenes".format(H,W))

  labels = ["pixel_" + str(i) for i in range(H*W)]
  df = pd.DataFrame(X_test_reshaped, columns=labels)
  df['Digit_label'] = y_test
  list1, list2 = zip(*positions)
  df['Position_x'] = np.array(list1)
  df['Position_y'] = np.array(list2)
  cols = ['Digit_label', 'Position_x', 'Position_y'] + \
         [col for col in df if (col != 'Digit_label') and (col != 'Position_x') and (col != 'Position_y')]
  df = df[cols]

  df.to_pickle(path_to_save + '/{}x{}.p'.format(H,W))

  #return X_test_reshaped, df

if __name__=='__main__':
  data = fetch_data()
  X_test, y_test = data.test

  scenes = [(40, 80), (100, 50), (50, 40)]
  _, _, mnist_path = get_paths(platform.system(), "_MNIST_scenes")

  #for i in range(len(scenes)):
    #create_new_test_dataframe(X_test, y_test, scenes[i], mnist_path)
