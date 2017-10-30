import pandas as pd
import numpy as np
from sklearn.datasets import fetch_mldata
from sgd_src.data_prepocessor import DataPreprocessor
from sgd_src.solvers import SimpleSgdSolver, MomentunSgdSolver, BoostingSgdSolver
from sgd_logger.logger import Logger
from sgd_comparator.comparator import SgdComparator
from sgd_crossvalid.cross_validator import CrossValidator
import os
#import progressbar

def train_and_test(logger, solver, data_preprocessor, sgd_comparator, type, epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):

  i = 0
  sgd_cost_histories = []
  sgd_epochs_to_converge = []
  sgd_all_theta = np.zeros((10,data_preprocessor.train_X.shape[1]))
  sgd_time = 0
  for target in np.unique(mnist.target):
    logger.log("Computing theta for target = {}".format(int(target)), verbosity_level = 2)
    tmp_y = np.array(data_preprocessor.train_y == target, dtype=int)

    if n_boost != 0:
      tmp_y = np.array([10 if i != 0 else -10 for i in tmp_y])

    sgd_time += solver.train(data_preprocessor.train_X, tmp_y, data_preprocessor.validation_X, data_preprocessor.validation_y, data_preprocessor.initial_theta,
      epochs, alpha, batch_size, beta, speed, n_boost)
    sgd_all_theta[i] = solver.model
    sgd_epochs_to_converge.append(solver.epochs_to_convergence)
    sgd_cost_histories.append(solver.cost_history)
    i += 1

  logger.log("Total train time for {} {:.3f}s".format(type, sgd_time), verbosity_level = 2)

  print()
  sgd_comparator.show_results(type, "test", sgd_all_theta, data_preprocessor.test_X, data_preprocessor.test_y, sgd_time)
  print()
  sgd_comparator.show_results(type, "train", sgd_all_theta, data_preprocessor.train_X, data_preprocessor.train_y, sgd_time, sgd_cost_histories, sgd_epochs_to_converge)
  print()

  #np.save(os.path.join(os.path.expanduser("~"), 'Google Drive/_cloudifier_data/09_tests/_sliding_windowMNIST_SimpleTest/another_model.npy'), sgd_all_theta)

def cross_validation(logger, data_preprocessor, batches, alphas, betas, speeds, boosts, epochs,
  file_to_save = "results.csv", _ignore = True):

  cross_validator = CrossValidator(logger, file_to_save)
  cross_validator.compute_best_hyperparams(mnist.target, data_preprocessor.train_X, data_preprocessor.train_y, data_preprocessor.validation_X, data_preprocessor.validation_y, batches, alphas, betas, speeds, boosts, epochs, ignore = _ignore)

  print(cross_validator.best_hyperparams)

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

  batches = [5, 10, 15, 20]
  alphas = [0.1, 0.01, 0.001]
  betas = [0, 0.001, 0.0001]
  speeds = [0.8, 0.9]
  boosts = [2, 3, 4]
  epochs = [5 + i for i in range(10)]
  """
  cross_validation(logger, data_preprocessor, batches, alphas, betas, speeds, boosts, epochs,
    _ignore = True)
  """
  sgd_comparator = SgdComparator(logger)

  solver = SimpleSgdSolver(logger, epsilon = pow(10, -4))
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "simple without reg",
    epochs = 15, alpha = 0.01, batch_size = 10, beta = 0, speed = 0, n_boost = 0)

  solver = SimpleSgdSolver(logger, epsilon = pow(10, -4))
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "simple with reg",
    epochs = 15, alpha = 0.01, batch_size = 10, beta = 0.001, speed = 0, n_boost = 0)

  solver = MomentunSgdSolver(logger, epsilon = pow(10, -4))
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "momentun without reg",
    epochs = 15, alpha = 0.001, batch_size = 10, beta = 0, speed = 0.9, n_boost = 0)

  solver = MomentunSgdSolver(logger, epsilon = pow(10, -4))
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "momentun with reg",
    epochs = 15, alpha = 0.01, batch_size = 10, beta = 0.001, speed = 0.9, n_boost = 0)

  logger.log("Summary of general results:\n\n {}".format(sgd_comparator.df), verbosity_level = 2)

  '''
  aux_solver = SimpleSgdSolver(logger, epsilon = pow(10, -4))
  solver = BoostingSgdSolver(logger, epsilon = pow(10, -4), solver = aux_solver)
  train_and_test(logger, solver, data_preprocessor, sgd_comparator, "boosting without reg",
    epochs = 15, alpha = 0.0001, batch_size = 10, beta = 0, speed = 0.0, n_boost = 3)
'''
