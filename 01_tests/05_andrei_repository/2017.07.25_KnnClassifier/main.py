from importlib.machinery import SourceFileLoader
from sklearn.datasets import fetch_mldata
from classifier import KnnClassifier
import platform
import pandas as pd
import numpy as np
import os

def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

if __name__=='__main__':

  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
  logger = logger_lib.Logger(show = True)

  mnist = fetch_mldata('MNIST original', data_home = mnist_path)
  labels = ["pixel_" + str(i) for i in range(784)]
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']], \
    columns = ["Digit_label"] + labels)
  logger.log("Finished fetching MNIST")

  solver = KnnClassifier(mnist_df, 0.14, logger)
  solver.preprocess()
  vals, counts = solver.predict(k = 5)
  solver.compute_accuracy(vals, counts)