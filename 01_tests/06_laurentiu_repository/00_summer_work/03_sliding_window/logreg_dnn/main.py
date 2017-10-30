from sklearn.datasets import fetch_mldata
from importlib.machinery import SourceFileLoader
import numpy as np
import pandas as pd
import platform
import os

from slider import LogRegSlider, FCNSlider, CNNSlider


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path


def slide_with_logreg(model_path, df_files, sizes, window_size, step_size, logger):

  theta_file = os.path.join(model_path, "_logreg")
  theta_file = os.path.join(theta_file, "weights.npy")
  print(theta_file)

  slider = LogRegSlider(df_files, sizes, theta_file, step_size, window_size, logger)
  slider.slide()


def slide_with_fcn(model_path, df_files, sizes, window_size, step_size, logger):

  model_files = [os.path.join(model_path, "_fully_connected_nn") for i in range(4)]
  files = ["w0", "b0", "w1", "b1"]
  files = [files[i] + ".npy" for i in range(4)]
  model_files = [os.path.join(path, file) for (path, file) in zip (model_files, files)]

  slider = FCNSlider(df_files, sizes, model_files, step_size, window_size, logger)
  slider.slide()

def slide_with_cnn(model_path, df_files, sizes, window_size, step_size, logger):

  #slider = CNNSlider(df_files, sizes, None, step_size, window_size, logger)
  slider.slide()


if __name__=="__main__":
  base_dir, utils_path, model_path = get_paths(platform.system(), "_MNIST_models")
  _, _, scenes_path = get_paths(platform.system(), "_MNIST_scenes")
  _, _, mnist_path = get_paths(platform.system(), "_MNIST_data")
  #file_names = ['40x80.p', '50x40.p', '100x50.p']

  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()

  labels = ["pixel_" + str(i) for i in range(784)]
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']],
    columns = ["Digit_label"] + labels)

  df_files = ['250x250.p']
  #sizes = [(40, 80), (50, 40), (100, 50)]
  sizes=[(250,250)]
  window_size = (28, 28)
  step_size = 2

  logger = logger_lib.Logger(show = True)

  '''
  theta_file = os.path.join(model_path, "_logreg")
  theta_file = os.path.join(theta_file, "weights.npy")
  slider = LogRegSlider(df_files, sizes, theta_file, step_size, window_size, logger)
  '''
  #slide_with_logreg(model_path, df_files, sizes, window_size, step_size, logger)
  #slide_with_fcn(model_path, df_files, sizes, gwindow_size, step_size, logger)

  slider = CNNSlider(df_files, sizes, None, step_size, window_size, logger)
  slide_with_cnn(model_path, df_files, sizes, window_size, step_size, logger)
  slider.process_scene_inference()

