import numpy as np
import pandas as pd
import gc
import time
from utils import sigmoid, min_max_scaler

_DEBUG_ = False

class Slider():

  def __init__(self, df_files, theta_file, sizes, step_size, logger):
    self.df_files = df_files
    self.theta_file = theta_file
    self.sizes = sizes
    self.step_size = step_size
    self.num_df = len(df_files)
    self.crt_df = None
    self.crt_idx = 0
    self.logger = logger
    self.model = np.load(theta_file)
    #self.model = self.model[:,1:]
    self.X = None
    self.y = None
    self.img_pos = None
    self.results = [[0, 0] for i in range(self.num_df)]
    np.set_printoptions(precision = 2, suppress = True)

  def sliding_window(self, image, step_size, window_size = (28, 28)):
    for y in range(0, image.shape[0] - window_size[1], step_size):
      for x in range(0, image.shape[1] - window_size[0], step_size):
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

  def slide_over_image(self, image, i):
    start_time = time.time()

    windows = []
    positions = []
    self.dict_windows = {}

    for (x, y, window) in self.sliding_window(image, step_size = self.step_size):

      if _DEBUG_:
        self.logger.log("\tTaking window at pos = {}, {}...".format(y,x))

      self.dict_windows[(y,x)] = window
      window = window.flatten()
      window = np.insert(window, 0, 1)

      windows.append(window)
      positions.append((x, y))
      window = np.reshape(window, (-1, window.shape[0])) # used only for softmax

    self.windows = np.array(windows)
    self.positions = np.array(positions)

    empty_img_threshold = 35
    positions_empty_im = np.where(self.windows.sum(axis=1) < empty_img_threshold)

    self.probabilities = sigmoid(np.dot(self.windows, self.model.T))
    if _DEBUG_:
      self.logger.log("\tProbabilities={}".format(probabilities))

    self.logger.log("\tScene slided in {:.2f}s".format(time.time()-start_time),
                    verbosity_level=2)


  def read_df(self, idx):

    self.crt_df = pd.read_pickle(self.df_files[self.crt_idx])

    self.X = np.array(self.crt_df.iloc[:, 3:].values, dtype = float)
    self.X = min_max_scaler(self.X)

    self.y = np.array(self.crt_df.iloc[:, 0].values, dtype = int)
    self.img_pos = np.array(self.crt_df.iloc[:, 1:3].values, dtype = int)

  def slide_over_df(self):

    self.read_df(self.crt_idx)

    for i in range(1):
      self.logger.log("Start sliding scene #{}; position of the image with target = {} in the scene = ({}, {})"
                    .format(i, self.y[i], self.img_pos[i][1], self.img_pos[i][0]),
                    verbosity_level=2)
      image = self.X[i].reshape(self.sizes[self.crt_idx][0], self.sizes[self.crt_idx][1])
      self.slide_over_image(image, i)

  def slide(self):

    for i in range(1):
      start_time = time.time()
      self.slide_over_df()
      self.logger.log("Test scenes of sz {}x{} slided in {:.2f}s; corrects={}, wrongs={}"
                      .format(self.sizes[i][0], self.sizes[i][1], time.time()-start_time,
                      self.results[i][0], self.results[i][1]))

      self.crt_idx += 1
      del self.crt_df
      gc.collect()

if __name__=='__main__':
  print("Library module. No main function")