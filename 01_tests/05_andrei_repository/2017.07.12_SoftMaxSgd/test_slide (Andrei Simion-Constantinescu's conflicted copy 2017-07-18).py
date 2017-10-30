import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
from sgd_logger.logger import Logger
np.set_printoptions(precision = 6)

def softmax(z):
  z -= np.max(z)
  sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
  return sm

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

if __name__=="__main__":
  os_home = os.path.expanduser("~")
  dir = os.path.join(os_home, 'Google Drive/_cloudifier_data/09_tests/_simple_sliding_window_mnist')

  df_file = os.path.join(dir, '100x50.p')
  theta_file = os.path.join(dir, 'sklearn_clf_model.npy')

  df = pd.read_pickle(df_file)
  model = np.load(theta_file)
  test_X = np.array(df.iloc[:, 3:].values, dtype = float)
  test_y = np.array(df.iloc[:, 0].values)
  normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
  test_X = normalizer.fit_transform(test_X)


  # test_X[0]
  poz = 26
  image = test_X[poz].reshape(100,50)
  plt.matshow(image, cmap = 'gray')
  plt.colorbar()
  plt.show()

  logger = Logger(show = True, verbosity_level = 2)

  logger.log("Sliding window over a scene 100x50.....", verbosity_level=2)
  logger.log("Position of the image in the scene = {}, {}".format(df.iloc[poz][2], df.iloc[poz][1]),  verbosity_level=2)
  maximum_prob = 0
  for y in range(0, image.shape[0]-28, 1):
    for x in range(0, image.shape[1]-28, 1):
      logger.log("\tTaking window at pos = {}, {}...".format(y,x))
      window = image[y:y+28, x:x+28]

      if x == 0 and y == 0:
      	plt.matshow(window, cmap = 'gray')
        plt.title("%d %d" % (y, x))
        plt.colorbar()
        plt.show()

      if (y == 65) and (x % 5 == 0):
        plt.matshow(window, cmap = 'gray')
        plt.title("%d %d" % (y, x))
        plt.colorbar()
        plt.show()


      if (y == 54) and (x % 5 == 0):
        plt.matshow(window, cmap = 'gray')
        plt.title("%d %d" % (y, x))
        plt.colorbar()
        plt.show()

      window = window.flatten()
      window = np.insert(window, 0, 1)
      #window = np.reshape(window, (-1, window.shape[0]))
      probabilities = sigmoid(np.dot(window, model.T))
      logger.log("\t{}".format(probabilities))
      prob = np.max(probabilities)
      preds = np.argmax(probabilities)
      logger.log("\t...Found target={} with prob={:.2f}".format(preds,prob))
      if prob > maximum_prob:
        maximum_prob = max(maximum_prob, prob)
        max_tuple = (y, x, preds, maximum_prob)
      logger.log("\n")

  logger.log("y={}, x={}, target={}, prob={}"
             .format(max_tuple[0], max_tuple[1], max_tuple[2], max_tuple[3]), verbosity_level=2)