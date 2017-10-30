from abc import ABC, abstractmethod
import numpy as np
import time

class SgdSolver(ABC):

  def __init__(self, logger, epsilon):
    self.logger = logger
    self.epsilon = epsilon
    self.model = None
    self.cost_history = None
    self.epochs_to_convergence = None
    np.set_printoptions(precision = 6)

  def sigmoid(self, z):
    return 1.0 / (1 + np.exp(-z))

  def save_results(self, theta, cost_history, epochs):
    self.model = theta
    self.cost_history = cost_history
    self.epochs_to_convergence = epochs

  def print_statistics(self, epoch, current_cost, diff, gradient, theta):
    self.logger.log("cost_eph#{:>2} = {:.4f}; abs diff between current and last eph = {:.4f}"
                    .format(epoch, current_cost, abs(diff)), tabs = 1, verbosity_level = 2)
    self.logger.log("eph#{:>2}, gradient[380:385] = {}"
                    .format(epoch, gradient[380:385]), tabs = 1, verbosity_level = 1)

  def cost_function(self, X, y, theta, beta):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + \
        (beta / (2 * m) * np.sum(theta[1:] ** 2))

  @abstractmethod
  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):
    pass

class SimpleSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)
    self.logger.log("Initialize simple gradient descendent logisitic regression solver")

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):

    start = time.time()

    reg_str = "simple without reg"
    if beta != 0:
      reg_str = "simple with reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}"
      .format(reg_str, alpha, batch_size, beta), verbosity_level = 2)

    theta = np.array(initial_theta)
    cost_history = []
    gradient = None
    last_theta = None
    for epoch in range(epochs):
      for i in np.arange(0, train_X.shape[0], batch_size):

        current_X = train_X[i : i + batch_size, :]
        current_y = train_y[i : i + batch_size]

        predicted = self.sigmoid(np.dot(current_X, theta))
        residual = predicted - current_y
        gradient = current_X.T.dot(residual) / current_X.shape[0]
        gradient[1:] += beta * theta[1:] / current_X.shape[0]

        theta -= alpha * gradient

      cost_history.append(self.cost_function(train_X, train_y, theta, beta))
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
        self.logger.log("eph#{}, delta(cost) < epsilon ==> early stopping"
                      .format(epoch, abs(diff)), tabs = 1, verbosity_level = 2)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for {} training = {:.3f}s".format(reg_str, stop - start))

    return stop - start

class MomentunSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)
    self.logger.log("Initialize momentun gradient descendent logisitic regression solver")

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):

    start = time.time()

    reg_str = "momentum without reg"
    if beta != 0:
      reg_str = "momentum with reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}, momentum={}"
      .format(reg_str, alpha, batch_size, beta, speed), verbosity_level = 2)

    theta = np.array(initial_theta)
    cost_history = []
    gradient = None
    last_momentum = None
    last_theta = None
    for epoch in range(epochs):
      for i in np.arange(0, train_X.shape[0], batch_size):

        current_X = train_X[i : i + batch_size, :]
        current_y = train_y[i : i + batch_size]

        predicted = self.sigmoid(np.dot(current_X, theta))
        residual = predicted - current_y
        gradient = current_X.T.dot(residual) / current_X.shape[0]
        gradient[1:] += beta * theta[1:] / current_X.shape[0]

        if last_momentum is not None:
          momentun = speed * last_momentum + alpha * gradient
        else:
          momentun = alpha * gradient

        theta -= momentun
        last_momentum = momentun

      cost_history.append(self.cost_function(train_X, train_y, theta, beta))
      last_theta = theta

      if epoch == 0:
        continue

      diff = cost_history[-2] - cost_history[-1]

      if diff < 0:
        #alpha /= 2
        theta = last_theta
        self.logger.log("eph#{}, cost increased by {:.4f} ==> decrease alpha to {:.4f}"
          .format(epoch, abs(diff), alpha), tabs = 1, verbosity_level = 1)
      else:
        #alpha += alpha * 0.05
        self.logger.log("eph#{}, cost decreased by {:.4f} ==> increasing alpha to {:.4f}"
                        .format(epoch, abs(diff), alpha), tabs = 1, verbosity_level = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("eph#{}, delta(cost) < epsilon ==> early stopping"
                      .format(epoch), tabs = 1, verbosity_level = 2)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for {} training = {:.3f}s".format(reg_str, stop - start))

    return stop - start

class BoostingSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)

  def cost_function(self, X, y, theta, beta):
    m = y.size
    h = np.dot(X, theta)
    return 1 / (2 * m) * np.sum( (h - y) ** 2 ) + (beta / (2 * m) * np.sum(theta[1:] ** 2))

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0):
    pass
    """
    idx0 = np.where(train_y == 0)[0]
    idx1 = np.where(train_y == 1)[0]
    train_y[idx0] -= 10
    train_y[idx1] *= 10

    start = time.time()

    reg_str = "sgd boosting"

    self.logger.log("Start {} training: alpha={}, batchSz={}, n_boost={}"
      .format(reg_str, alpha, batch_size, n_boost), verbosity_level = 2)

    theta = np.array(initial_theta)
    current_theta = np.array(initial_theta)
    current_y = train_y
    cost_history = []
    gradient = None
    """

if __name__=='__main__':
  print("Library module. No main function")


