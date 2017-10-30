import pandas as pd
import numpy as np
from sklearn.datasets import fetch_mldata
from sgd_src.data_prepocessor import DataPreprocessor
from sgd_src.solvers import SimpleSgdSolver
from sgd_logger.logger import Logger
from sgd_comparator.comparator import SgdComparator
import os

def train_and_test(logger, solver, data_preprocessor, sgd_comparator, type, epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):
  sgd_time = 0
  sgd_time += solver.train(data_preprocessor.train_X, data_preprocessor.train_y, data_preprocessor.validation_X, data_preprocessor.validation_y,
    epochs, alpha, batch_size, beta, speed, n_boost)

  logger.log("Total train time for {} {:.3f}s".format(type, sgd_time), verbosity_level = 2)

  print()
  sgd_comparator.show_results(type, "test", solver.model, data_preprocessor.test_X, data_preprocessor.test_y, sgd_time)
  print()
  sgd_comparator.show_results(type, "train", solver.model, data_preprocessor.train_X, data_preprocessor.train_y, sgd_time, solver.cost_history, solver.epochs_to_convergence)
  print()

if __name__=='__main__':

  pd.set_option('display.height', 1000)
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)

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

  sgd_comparator = SgdComparator(logger)


  solver = SimpleSgdSolver(logger, epsilon = pow(10, -4))
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "simple with reg",
    epochs = 15, alpha = 0.01, batch_size = 1, beta = 0.001, speed = 0, n_boost = 0)

  logger.log("Summary of general results:\n\n {}".format(sgd_comparator.df), verbosity_level = 2)