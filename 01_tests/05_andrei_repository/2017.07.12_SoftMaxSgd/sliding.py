import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

class Slider():

  def __init__(self, df_files, theta_file, sizes):
    self.df_files = df_files
    self.theta_file = theta_file
    self.sizes = sizes
    self.num_df = len(df_files)
    self.dfs = []
    self.Xs = []
    self.ys = []
    self.img_xys = []
    self.model = None

  def softmax(self, z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

  def read_data(self):

    for i in range(self.num_df):
      self.dfs.append(pd.read_pickle(df_files[i]))

    self.model = np.load(theta_file)

  def preprocess_data(self):

    for i in range(self.num_df):
      self.Xs.append(np.array(self.dfs[i].iloc[:, 3:].values))
      self.ys.append(np.array(self.dfs[i].iloc[:, 0].values))
      self.img_xys.append(np.array(self.dfs[i].iloc[:, 1:2].values))

    normalizer = preprocessing.MinMaxScaler(feature_range=(-1,1))

    for i in range(self.num_df):
      m = self.ys[i].size
      self.Xs[i] = normalizer.fit_transform(self.Xs[i])
      self.Xs[i] = np.c_[np.ones(m), self.Xs[i]]


  def sliding_window(self, image, step_size, window_size):

      for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


  def slide_over_image(self, image, num_set, win_w = 28, win_h = 28):

      img = image[1:].reshape(self.sizes[num_set][0], self.sizes[num_set][1])

      for (x, y, window) in self.sliding_window(img, step_size=4, window_size=(win_w, win_h)):
        if window.shape[0] != win_h or window.shape[1] != win_w:
            continue

        window = window.flatten()
        window = np.insert(window, 0, 1)
        print(window.shape)
        probabilities = self.softmax(np.dot(window, self.model))
        preds = np.argmax(probabilities, axis=1)
        print(preds)

  def slide(self):

    self.slide_over_image(self.Xs[0][0], 0)

if __name__=='__main__':

  dir = os.path.join(os.path.expanduser("~"), 'Google Drive/_cloudifier_data/09_tests/_sliding_windowMNIST_SimpleTest')
  file_names = ['40x80.p', '50x40.p', '100x50.p', '200x350.p']

  df_files = [os.path.join(dir, file_names[i]) for i in range(len(file_names))]
  theta_file = os.path.join(dir, 'trained_model.npy')
  sizes = [(40, 80), (50, 40), (100, 50), (200, 350)]


  slider = Slider(df_files, theta_file, sizes)

  slider.read_data()
  slider.preprocess_data()
  slider.slide()

