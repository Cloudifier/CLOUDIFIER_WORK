import os
import platform
import importlib
from importlib.machinery import SourceFileLoader

def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path


if __name__ == "__main__":
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")

  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "logger.py")).load_module()
  logger = logger_lib.Logger(show = True)

  logger.log("Check import from utils")
  logger.log(base_dir + " " + utils_path + " " + data_path)

