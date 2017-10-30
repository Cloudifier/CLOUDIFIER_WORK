from sklearn.model_selection import train_test_split
import numpy as np
from sliding_window_preprocessor import SlidingWindowPreprocessor

class DataPreprocessor():

  def __init__(self, df, test_size, logger):
    self.df = df
    self.logger = logger
    self.normalizer = None
    self.test_size = test_size
    self.train_X = None
    self.train_y = None
    self.test_X = None
    self.test_y = None
    self.validation_X = None
    self.validation_y = None
    self.hyperparameters = None
    self.initial_theta = None
    self.logger.log("Initialize data preprocessor")

  def minmax(self, X):
    return (X - X.min()) / (X.max() - X.min())

  def process_data(self):

    self.logger.log("Start preprocessing data")
    X = np.array(self.df.iloc[:, 1:].values)
    y = np.array(self.df.iloc[:, 0].values)

    self.logger.log("Split in train set and test set by {}"
                    .format(self.test_size * 100), tabs = 1)

    self.train_X, self.test_X, self.train_y, self.test_y  = \
        train_test_split(X, y, test_size = self.test_size, random_state=42)
    self.sld = SlidingWindowPreprocessor(self.test_X, self.test_y)

    m1 = self.train_y.size
    m2 = self.test_y.size
    self.logger.log("Normalize data", tabs = 1)
    self.train_X = self.minmax(self.train_X)
    self.test_X = self.minmax(self.test_X)

    self.train_X = np.c_[np.ones(m1), self.train_X]
    self.test_X = np.c_[np.ones(m2), self.test_X]
    self.logger.log("Finished normalizing data", tabs = 1)

    """
    self.validation_X, self.test_X, self.validation_y, self.test_y = \
        train_test_split(self.test_X, self.test_y, test_size = 0.5)
    """
    # We don't use it yet
    self.validation_X = None
    self.validation_y = None
    self.logger.log("Finished splitting data", tabs = 1)

    self.initial_theta = np.zeros(self.train_X.shape[1])
