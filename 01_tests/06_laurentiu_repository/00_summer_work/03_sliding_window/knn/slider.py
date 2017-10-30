import numpy as np
import pandas as pd
from utils import sigmoid, softmax, min_max_scaler
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
import multiprocessing
from tqdm import tqdm, trange
import time
import gc
import inspect
import sys

class Slider():

  def __init__(self, df_files, sizes, window_size, step_size, classifier, epsilon, logger):
    self.df_files = df_files
    self.sizes = sizes
    self.window_size = window_size
    self.step_size = step_size
    self.epsilon = epsilon
    self.num_df = len(df_files)
    self.crt_df = None
    self.crt_idx = 0
    self.logger = logger
    self.classifier = classifier
    self.X = None
    self.y = None
    self.img_pos = None
    self.results = [[0, 0] for i in range(self.num_df)]
    np.set_printoptions(precision = 2, suppress = True)

  def sliding_window(self, image, step_size):
    for y in range(0, image.shape[0] - self.window_size[1], step_size):
      for x in range(0, image.shape[1] - self.window_size[0], step_size):
        yield (x, y, image[y:y + self.window_size[1], x:x + self.window_size[0]])

  def slide_over_image(self, image):
    start_time = time.time()

    windows = []
    positions = []
    self.dict_windows = {}
    for (x, y, window) in self.sliding_window(image, step_size = self.step_size):

      self.logger.log("\tTaking window at pos = ({},{})".format(y,x), verbosity_level = 0)

      self.dict_windows[(y,x)] = window
      window = window.flatten()
      windows.append(window)
      positions.append((x, y))

    self.windows = np.array(windows)
    self.positions = np.array(positions)

    predicted_val, predicted_pos, vals, counts, top_k_sums = self.classifier.predict(self.windows, self.positions, k = 5)

    self.predicted_val = predicted_val
    self.predicted_pos = predicted_pos
    self.vals = vals
    self.counts = counts
    self.top_k_sums = top_k_sums

    self.logger.log("\tScene slided in {:.2f}s".format(time.time()-start_time),
                    verbosity_level = 2)

    return predicted_val, predicted_pos

  def read_df(self):

    self.crt_df = pd.read_pickle(self.df_files[self.crt_idx])
    self.X = np.array(self.crt_df.iloc[:, 3:].values, dtype = float)
    self.y = np.array(self.crt_df.iloc[:, 0].values, dtype = int)
    self.img_pos = np.array(self.crt_df.iloc[:, 1:3].values, dtype = int)
    self.X = min_max_scaler(self.X)

  def check_position(self, i, predicted_pos):
    return (abs(self.img_pos[i][1] - predicted_pos[1]) < self.epsilon and
      abs(self.img_pos[i][0] - predicted_pos[0]) < self.epsilon)

  def slide_over_df(self):

    self.read_df()
    self.logger.log("Sliding {} test scenes of size {}x{} with {}x{} windows and step_size={}".format(self.X.shape[0], self.sizes[self.crt_idx][0], self.sizes[self.crt_idx][1],
      self.window_size[0], self.window_size[1], self.step_size), verbosity_level = 2)

    old_print = print
    inspect.builtins.print = tqdm.write

    t = trange(self.X.shape[0], desc='Slider', leave=True)
    for i in range(self.X.shape[0]):
      if self.results[self.crt_idx][0] + self.results[self.crt_idx][1] == 0:
        crt_accuracy = 0
      else:
        crt_accuracy = float(self.results[self.crt_idx][0]) / (self.results[self.crt_idx][0] +
          self.results[self.crt_idx][1])
      t.set_description("Target {} -- Position ({}, {}) -- corrects = {}, wrongs = {} -- accuracy = {:.2f} %".format(self.y[i], self.img_pos[i][1], self.img_pos[i][0],
        self.results[self.crt_idx][0], self.results[self.crt_idx][1], crt_accuracy * 100))
      t.refresh()
      t.update(1)
      sys.stdout.flush()

      self.logger.log("Start sliding scene #{}; position of the image with target = {} in the scene = ({}, {})".format(i, self.y[i], self.img_pos[i][1], self.img_pos[i][0]), verbosity_level = 2)

      image = self.X[i].reshape(self.sizes[self.crt_idx][0], self.sizes[self.crt_idx][1])
      predicted_val, predicted_pos = self.slide_over_image(image)

      if predicted_val == self.y[i]:
        if self.check_position(i, predicted_pos):
          self.results[self.crt_idx][0] += 1

          self.logger.log("\tFound {} at pos ({}, {}) ... correct target, correct position"
                        .format(predicted_val, predicted_pos[0], predicted_pos[1]),
                        verbosity_level = 2)
        else:
          self.logger.log("\tFound {} at pos ({}, {}) ... correct target, wrong position"
                        .format(predicted_val, predicted_pos[0], predicted_pos[1]),
                        verbosity_level = 2)

          self.results[self.crt_idx][1] += 1
      else:
        if predicted_val == -1:
          self.logger.log("\tCould not match a window .. ", verbosity_level = 2)
        else:
          self.logger.log("\tFound {} at pos ({}, {}) ... incorrect target"
                        .format(predicted_val, predicted_pos[0], predicted_pos[1]),
                        verbosity_level = 2)

        self.results[self.crt_idx][1] += 1

      self.logger.log("Finished sliding scene #{}".format(i), verbosity_level = 2)

    inspect.builtins.print = old_print


  def slide(self):

    for i in range(1):
      start_time = time.time()
      self.slide_over_df()
      self.logger.log("Test scenes of size {}x{} slided in {:.2f}s; corrects={}, wrongs={}"
                      .format(self.sizes[i][0], self.sizes[i][1], time.time() - start_time,
                      self.results[i][0], self.results[i][1]))

      self.crt_idx += 1
      del self.crt_df
      gc.collect()

if __name__=='__main__':
  print("Library module. No main function")