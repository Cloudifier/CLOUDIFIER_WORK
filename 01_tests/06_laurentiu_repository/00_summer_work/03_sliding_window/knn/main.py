from sklearn.datasets import fetch_mldata
from importlib.machinery import SourceFileLoader
import numpy as np
import pandas as pd
import platform
import os

from slider import Slider
from classifier import KnnClassifier

def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path


if __name__=="__main__":

  base_dir, utils_path, data_path = get_paths(platform.system(), "_simple_sliding_window_mnist")
  _, _, mnist_path = get_paths(platform.system(), "_MNIST_data")
  file_names = ['40x80.p', '50x40.p', '100x50.p', '200x350.p']

  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()

  labels = ["pixel_" + str(i) for i in range(784)]
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']],
    columns = ["Digit_label"] + labels)

  df_files = [os.path.join(data_path, file_names[i]) for i in range(len(file_names))]
  sizes = [(40, 80), (50, 40), (100, 50), (200, 350)]
  window_size = (28, 28)
  split_params = (0.14, 24)
  step_size = 2
  epsilon = 3

  logger = logger_lib.Logger(show = True, verbosity_level = 2)
  classifier = KnnClassifier(mnist_df, split_params, logger)
  classifier.preprocess()
  slider = Slider(df_files, sizes, window_size, step_size, classifier, epsilon, logger)
  slider.slide()
