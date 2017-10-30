import os
from sklearn.datasets import fetch_mldata
import pandas as pd
import numpy as np

class Struct():

  def __init__(self, **kwds):
    self.__dict__.update(kwds)


def one_hot(y):
  zeros = [[0 for i in range(10)] for i in range(y.shape[0])]
  return np.array([zero[:int(i)] + [1] + zero[int(i)+1:] for zero,i in zip(zeros,y)])

def load_mnist(path):

  mnist = fetch_mldata('MNIST original', data_home = path)
  labels = ["pixel_" + str(i) for i in range(784)]
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']], columns = ["Digit_label"] +
    labels)

  return mnist_df


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path
