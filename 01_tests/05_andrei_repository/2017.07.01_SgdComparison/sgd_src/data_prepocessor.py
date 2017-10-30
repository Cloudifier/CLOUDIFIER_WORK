from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

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

  def process_data(self):

    self.logger.log("Start preprocessing data")
    X = np.array(self.df.iloc[:, 1:].values)
    y = np.array(self.df.iloc[:, 0].values)

    m = y.size
    self.logger.log("Normalize data", tabs = 1)
    self.normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = self.normalizer.fit_transform(X)
    X = np.c_[np.ones(m), X]
    self.logger.log("Finished normalizing data", tabs = 1)

    self.logger.log("Split in train set and test set by {}"
                    .format(self.test_size * 100), tabs = 1)

    self.train_X, self.test_X, self.train_y, self.test_y  = \
        train_test_split(X, y, test_size = self.test_size, random_state=42)
    #self.validation_X, self.test_X, self.validation_y, self.test_y = \
        #train_test_split(self.test_X, self.test_y, test_size = 0.5, random_state=42)

    print(self.test_X[0][204:214])

    #we dont use it yet
    self.validation_X = None
    self.validation_y = None

    self.logger.log("Finished splitting data", tabs = 1)

    self.initial_theta = np.zeros(self.train_X.shape[1])
