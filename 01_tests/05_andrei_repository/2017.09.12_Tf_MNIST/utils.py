import matplotlib.pyplot as plt
import os
import numpy as np

def plot(cost_history):
    plt.plot(np.arange(0, len(cost_history)), cost_history)
    plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path  = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path