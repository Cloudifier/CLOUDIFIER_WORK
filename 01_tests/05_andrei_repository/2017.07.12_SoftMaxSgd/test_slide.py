import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sgd_logger.logger import Logger
import platform

np.set_printoptions(precision = 6)

def softmax(z):
  z -= np.max(z)
  sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
  return sm

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def minmax(X):
  return (X - X.min()) / (X.max() - X.min())

if __name__=="__main__":
  os_home = os.path.expanduser("~")
  if platform.system() == "Windows":
    dir_name = os.path.join("D:/", 'GoogleDrive/_cloudifier_data/09_tests/_simple_sliding_window_mnist')
  else:
    dir_name = os.path.join(os.path.expanduser("~"), 'Google Drive/_cloudifier_data/09_tests/_simple_sliding_window_mnist')

  df_file = os.path.join(dir_name, '100x50.p')
  theta_file = os.path.join(dir_name, 'softmax_clf_mnist.npy')

  df = pd.read_pickle(df_file)
  model = np.load(theta_file)
  test_X = np.array(df.iloc[:, 3:].values, dtype=float)
  test_y = np.array(df.iloc[:, 0].values)

  test_X = minmax(test_X)

  # test_X[0]
  poz = 2012
  image = test_X[poz].reshape(100,50)
  plt.matshow(image, cmap = 'gray')
  plt.colorbar()
  plt.show()

  logger = Logger(show = True, verbosity_level = 2)

  logger.log("Sliding window over a scene 100x50.....", verbosity_level=2)
  logger.log("Position of the image in the scene = {}, {}".format(df.iloc[poz][2], df.iloc[poz][1]),  verbosity_level=2)
  maximum_prob = 0
  windows = {}
  above90 = {}
  for y in range(0, image.shape[0]-28, 1):
    for x in range(0, image.shape[1]-28, 1):
      logger.log("\tTaking window at pos = {}, {}...".format(y,x))
      window = image[y:y+28, x:x+28]
      windows[(y,x)] = window

      window = window.flatten()
      window = np.insert(window, 0, 1)
      window = np.reshape(window, (-1, window.shape[0]))
      probabilities = softmax(np.dot(window, model))
      logger.log("\t{}".format(probabilities))
      prob = np.max(probabilities)
      preds = np.argmax(probabilities)
      logger.log("\t...Found target={} with prob={:.2f}".format(preds,prob))
      if prob > maximum_prob:
        maximum_prob = max(maximum_prob, prob)
        max_tuple = (y, x, preds, maximum_prob)
      logger.log("\n")

      if prob > 0.9:
        above90[(y,x)] = (prob, preds)

  logger.log("y={}, x={}, target={}, prob={}"
             .format(max_tuple[0], max_tuple[1], max_tuple[2], max_tuple[3]), verbosity_level=2)