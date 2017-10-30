from sklearn.datasets import fetch_mldata
from logger import Logger
import numpy as np
import pandas as pd
import platform
import os

from slider import Slider
from classifier import KnnClassifier

if __name__=="__main__":

  file_names = ['40x80.p', '50x40.p', '100x50.p', '200x350.p']

  labels = ["pixel_" + str(i) for i in range(784)]
  mnist = fetch_mldata('MNIST original')
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']],
    columns = ["Digit_label"] + labels)

  sizes = [(40, 80), (50, 40), (100, 50), (200, 350)]
  window_size = (28, 28)
  split_params = (0.14, 24)
  step_size = 2
  epsilon = 3

  logger = Logger(show = True, verbosity_level = 2)
  classifier = KnnClassifier(mnist_df, split_params, logger)
  classifier.preprocess()
  slider = Slider(file_names, sizes, window_size, step_size, classifier, epsilon, logger)
  slider.slide()
