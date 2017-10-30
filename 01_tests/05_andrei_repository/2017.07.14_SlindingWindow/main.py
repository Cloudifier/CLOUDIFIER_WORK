from sklearn.datasets import fetch_mldata
from importlib.machinery import SourceFileLoader
import numpy as np
import pandas as pd
import platform
import os

from slider import Slider

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
  theta_file = os.path.join(data_path, 'sigmoid_clf_model.npy')
  sizes = [(40, 80), (50, 40), (100, 50), (200, 350)]
  step_size = 1

  logger = logger_lib.Logger(show = True, verbosity_level = 2)
  slider = Slider(df_files, theta_file, sizes, step_size, logger)
  slider.slide()

