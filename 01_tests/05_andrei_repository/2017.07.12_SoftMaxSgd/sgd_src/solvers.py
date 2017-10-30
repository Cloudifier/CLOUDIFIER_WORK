from abc import ABC, abstractmethod
import numpy as np
import time

class SgdSolver(ABC):

  def __init__(self, logger, epsilon):
    self.logger = logger
    self.epsilon = epsilon
    self.model = None
    self.cost_history = []
    self.epochs_to_convergence = 0
    np.set_printoptions(precision = 6)

  def save_results(self, theta, cost_history, epochs):
    self.model = theta
    self.cost_history = cost_history
    self.epochs_to_convergence = epochs

  def print_statistics(self, epoch, current_cost, diff, gradient, theta):
    self.logger.log("cost_eph#{:>2} = {:.4f}; abs diff between current and last eph = {:.4f}"
                    .format(epoch, current_cost, abs(diff)), tabs = 1, verbosity_level = 2)
    self.logger.log("eph#{:>2}, gradient[380:385] = {}"
                    .format(epoch, gradient[380:385]), tabs = 1, verbosity_level = 1)

  def binTargets(self, y):
    zeros = [[0 for i in range(self.classes)] for i in range(y.shape[0])]
    return np.array([zero[:int(i)] + [1] + zero[int(i)+1:] for zero,i in zip(zeros,y)])

  def softmax(self, z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

  def computeLossAndGradient(self, X, y, theta, beta):
    m = X.shape[0]
    y_mat = self.binTargets(y)
    h = self.softmax(np.dot(X, theta))
    loss = (-1 / m) * np.sum(y_mat * np.log(h)) + (beta / (2 * m) * np.sum(theta[1:] ** 2))

    residual = y_mat - h

    gradient = -X.T.dot(residual) / m
    gradient[1:] += beta * theta[1:] / m

    return loss, gradient

  def dump_model_to_file(self, filename):
    np.save(filename, self.model)

  @abstractmethod
  def train(self, train_X, train_y, validation_X, validation_y,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):
    pass

class SimpleSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)
    self.logger.log("Initialize simple gradient descendent softmax regression solver",\
                    verbosity_level=2)

  def train(self, train_X, train_y, validation_X, validation_y,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,
    is_linear_reg = False):

    self.classes = len(np.unique(train_y))
    start = time.time()

    reg_str = "simple without reg"
    if beta != 0:
      reg_str = "simple with reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}"
      .format(reg_str, alpha, batch_size, beta), verbosity_level = 2)

    theta = np.zeros((train_X.shape[1], self.classes))
    cost_history = []
    last_theta = None
    for epoch in range(epochs):
      for i in np.arange(0, train_X.shape[0], batch_size):

        current_X = train_X[i : i + batch_size, :]
        current_y = train_y[i : i + batch_size]

        loss, gradient = self.computeLossAndGradient(current_X, current_y, theta, beta)

        theta -= alpha * gradient

      cost_history.append(loss)
      last_theta = theta

      if epoch == 0:
        continue

      diff = cost_history[-2] - cost_history[-1]

      if diff < 0:
        alpha /= 2
        theta = last_theta
        self.logger.log("eph#{}, cost increased by {:.4f} ==> decrease alpha to {:.4f}"
          .format(epoch, abs(diff), alpha), tabs = 1, verbosity_level = 1)
      else:
        alpha += alpha * 0.05
        self.logger.log("eph#{}, cost decreased by {:.4f} ==> increasing alpha to {:.4f}"
                        .format(epoch, abs(diff), alpha), tabs = 1, verbosity_level = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("eph#{}, delta(cost) < epsilon"
                      .format(epoch, abs(diff)), tabs = 1, verbosity_level = 2)
        self.save_results(theta, cost_history, epoch + 1)

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for {} training = {:.3f}s".format(reg_str, stop - start))

    return stop - start

class MomentumSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)
    self.logger.log("Initialize momentum gradient descendent softmax regression solver",\
                    verbosity_level=2)

  def train(self, train_X, train_y, validation_X, validation_y,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,):

    self.classes = len(np.unique(train_y))
    start = time.time()

    reg_str = "momentum without reg"
    if beta != 0:
      reg_str = "momentum with reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}, momentum={}"
      .format(reg_str, alpha, batch_size, beta, speed), verbosity_level = 2)

    theta = np.zeros((train_X.shape[1], self.classes))
    cost_history = []
    last_momentum = None
    for epoch in range(epochs):
      for i in np.arange(0, train_X.shape[0], batch_size):

        current_X = train_X[i : i + batch_size, :]
        current_y = train_y[i : i + batch_size]

        loss, gradient = self.computeLossAndGradient(current_X, current_y, theta, beta)

        if last_momentum is not None:
          momentum = speed * last_momentum + alpha * gradient
        else:
          momentum = alpha * gradient

        theta -= momentum
        last_momentum = momentum

      cost_history.append(loss)

      if epoch == 0:
        continue

      diff = cost_history[-2] - cost_history[-1]


      if abs(diff) < self.epsilon:
        self.logger.log("eph#{}, delta(cost) < epsilon"
                      .format(epoch), tabs = 1, verbosity_level = 2)
        self.save_results(theta, cost_history, epoch + 1)

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for {} training = {:.3f}s".format(reg_str, stop - start))

    return stop - start

if __name__=='__main__':
  print("Library module. No main function")


