import numpy as np
from sgd_src.solvers import SimpleSgdSolver, MomentunSgdSolver, BoostingSgdSolver
from tqdm import tqdm
import csv

class CrossValidator():

  def __init__(self, logger, file_to_save):
    self.best_hyperparams = None
    self.logger = logger
    self.file_to_save = file_to_save

  def sigmoid(self, z):
    z -= z.max()
    return 1.0 / (1 + np.exp(-z))

  def compare_results(self, X, y, all_theta):

    prediction = self.sigmoid(X.dot(all_theta.T))
    corrects = np.zeros(10)
    wrongs = np.zeros(10)

    for i in range(X.shape[0]):
      current_digit = int(y[i])
      res = current_digit - np.argmax(prediction[i, :])

      if res == 0:
        corrects[current_digit] += 1
      else:
        wrongs[current_digit] += 1

    t_correct = np.sum(corrects)
    t_wrong = np.sum(wrongs)
    accuracy = 100 * t_correct / float(t_correct + t_wrong)

    return accuracy

  def train_model(self, targets, X, y, solver, epochs, alpha, batch_size, beta, speed, n_boost):

    i = 0
    initial_theta = np.zeros(X.shape[1])
    sgd_all_theta = np.zeros((len(np.unique(targets)), X.shape[1]))
    for target in np.unique(targets):
      tmp_y = np.array(y == target, dtype=int)

      if n_boost != 0:
        tmp_y = np.array([10 if i != 0 else -10 for i in tmp_y])

      solver.train(X, tmp_y, None, None, initial_theta, epochs, alpha, batch_size, beta, speed, n_boost)
      sgd_all_theta[i] = solver.model
      i += 1

    return sgd_all_theta

  def get_type(self, index):

    types = ["Simple", "Momentun", "Boosting"]
    return types[index]

  def save_to_file(self):

    if self.best_hyperparams is None:
      return

    if self.best_hyperparams[0] == 'Simple':
      self.best_hyperparams = tuple(self.best_hyperparams[i] if i < len(self.best_hyperparams) - 2 else "N/A" for i in range(len(self.best_hyperparams)))
    elif self.best_hyperparams[0] == 'Momentun':
      self.best_hyperparams = tuple(self.best_hyperparams[i] if i < len(self.best_hyperparams) - 1 else "N/A" for i in range(len(self.best_hyperparams)))

    with open(self.file_to_save, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(["Alg", "Acc", "Epochs", "Batch", "Alpha", "Beta", "Speed", "Boosters"])
      writer.writerow(self.best_hyperparams)

  def compute_best_hyperparams(self, targets, train_X, train_y, test_X, test_y, batches, alphas, betas, speeds, boosts, epochs, ignore = True):

    if ignore:
      return

    s_solver = SimpleSgdSolver(self.logger, epsilon = pow(10, -4))
    m_solver = MomentunSgdSolver(self.logger, epsilon = pow(10, -4))
    max_acc = 0

    self.logger.change_show(False)
    for n_epochs in tqdm(epochs):
      for batch in batches:
        for alpha in alphas:
          for beta in betas:
            for speed in speeds:
              for boost in boosts:
                accuracies = []
                all_theta = np.zeros((len(np.unique(targets)), train_X.shape[1]))
                all_theta = self.train_model(targets, train_X, train_y, s_solver, n_epochs, alpha, batch, beta, 0, 0)
                accuracies.append(self.compare_results(test_X, test_y, all_theta))

                all_theta = np.zeros((len(np.unique(targets)), train_X.shape[1]))
                all_theta = self.train_model(targets, train_X, train_y, m_solver, n_epochs, alpha, batch, beta, speed, 0)
                accuracies.append(self.compare_results(test_X, test_y, all_theta))

                if max_acc < max(accuracies):
                  max_acc = max(accuracies)
                  index = accuracies.index(max(accuracies))

                  self.best_hyperparams = (self.get_type(index), "{:.2f}".format(max_acc),
                    n_epochs, batch, alpha, beta, speed, boost)

    self.logger.change_show(True)
    self.save_to_file()












