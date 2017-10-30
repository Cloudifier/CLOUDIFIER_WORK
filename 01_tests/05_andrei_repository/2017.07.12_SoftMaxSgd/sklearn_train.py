import pandas as pd
import numpy as np
from sklearn.datasets import fetch_mldata
from sgd_src.data_prepocessor import DataPreprocessor
from sgd_logger.logger import Logger
from sgd_comparator.comparator import SgdComparator
import os
from sklearn.linear_model import SGDClassifier

if __name__=="__main__":
  logger = Logger(show = True, verbosity_level = 2)
  logger.log("Fetch MNIST Data Set", verbosity_level = 2)
  os_home = os.path.expanduser("~")
  data_home = os.path.join(os_home, 'Google Drive/_cloudifier_data/09_tests/_MNIST_data')
  mnist = fetch_mldata('MNIST original', data_home=data_home)
  labels = ["pixel_" + str(i) for i in range(784)]
  mnist_df = pd.DataFrame(np.c_[mnist['target'], mnist['data']] , \
                          columns = ["Digit_label"] + labels)
  logger.log("Finished fetching MNIST Data Set", verbosity_level = 2)

  data_preprocessor = DataPreprocessor(mnist_df, 0.14, logger)
  data_preprocessor.process_data()

  clf = SGDClassifier(loss="log", alpha=0.001)
  clf.fit(data_preprocessor.train_X, data_preprocessor.train_y)