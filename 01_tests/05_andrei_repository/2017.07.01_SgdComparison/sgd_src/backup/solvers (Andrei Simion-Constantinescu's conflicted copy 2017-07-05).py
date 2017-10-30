from abc import ABC, abstractmethod
import numpy as np
import time

class SgdSolver(ABC):

  def __init__(self, hyperparameters, logger, epsilon):
    self.logger = logger
    self.default_alpha = hyperparameters.alpha
    self.hyperparameters = hyperparameters
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
    self.logger.log("Cost function value at the end of epoch #{} is: {:.4f}"
                    .format(epoch, current_cost), tabs = 1)
    self.logger.log("Difference between eph #{} and eph #{} is {:.4f}"
                    .format(epoch-1, epoch, abs(diff)), tabs = 1)

    self.logger.log("Values for gradient[380:385] at the end of epoch #{} are: {}"
                    .format(epoch, gradient[380:385]), tabs = 1)

  @abstractmethod
  def cost_function(self, X, y, theta, _lambda):
    pass

  @abstractmethod
  def train(self, train_X, train_y, validation_X, validation_y, initial_theta):
    pass

class SimpleSgdSolver(SgdSolver):

  def __init__(self, hyperparameters, logger, epsilon):
    super().__init__(hyperparameters, logger, epsilon)
    self.logger.log("Initialize simple gradient descendent logisitic regression solver")

  def cost_function(self, X, y, theta, _lambda):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + \
        (_lambda / (2 * m) * np.sum(theta[1:] ** 2))

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta):

    start = time.time()

    self.hyperparameters.alpha = self.default_alpha

    self.logger.log("Start simple training", tabs = 1)
    self.logger.log("Hyperparameters are:", tabs = 1)
    self.logger.log("learning rate: {:.4f}, batch size: {}, reg param: {}"
                    .format(self.hyperparameters.alpha, self.hyperparameters.batch_size, \
                    self.hyperparameters._lambda), tabs = 2)

    theta = np.array(initial_theta)

    self.logger.log("Start with theta[1:10]: {}".format(theta[:10]), tabs = 1)
    cost_history = []
    gradient = None
    last_theta = None
    for epoch in range(self.hyperparameters.epochs):
      for i in np.arange(0, train_X.shape[0], self.hyperparameters.batch_size):

        current_X = train_X[i : i + self.hyperparameters.batch_size, :]
        current_y = train_y[i : i + self.hyperparameters.batch_size]

        predicted = self.sigmoid(np.dot(current_X, theta))
        residual = predicted - current_y
        gradient = current_X.T.dot(residual) / current_X.shape[0]
        gradient[1:] += self.hyperparameters._lambda * theta[1:] /  \
                      current_X.shape[0]

        theta -= self.hyperparameters.alpha * gradient


      cost_history.append(self.cost_function(train_X, train_y, theta, \
                    self.hyperparameters._lambda))
      last_theta = theta

      if epoch == 0:
        continue

      diff = cost_history[-2] - cost_history[-1]

      if diff < 0:
        self.logger.log("At the end of epoch #{}, cost function increased by {:.4f}"
                      .format(epoch, abs(diff)), tabs = 1)
        self.hyperparameters.alpha /= 2
        theta = last_theta
        self.logger.log("Decresing alpha to {:.4f}"
                        .format(self.hyperparameters.alpha), tabs = 1)
      else:
        self.hyperparameters.alpha += self.hyperparameters.alpha * 0.05
        self.logger.log("Increasing alpha to {:.4f}"
                        .format(self.hyperparameters.alpha), tabs = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("Difference between iterations less than epsilon"
                      .format(abs(diff)), tabs = 1)
        self.logger.log("Stopped at epoch #{}".format(epoch), tabs = 1)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time to train was {:.3f}s".format(stop - start), tabs = 1)

    return stop - start

class MomentunSgdSolver(SgdSolver):

  def __init__(self, hyperparameters, logger, epsilon):
    super().__init__(hyperparameters, logger, epsilon)

  def cost_function(self, X, y, theta):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))
    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta):

    start = time.time()

    self.hyperparameters.alpha = self.default_alpha

    self.logger.log("Start simple training", tabs = 1)
    self.logger.log("Hyperparameters are:", tabs = 1)
    self.logger.log("learning rate: {:.4f}, batch size: {}, reg param: {}"
                    .format(self.hyperparameters.alpha, self.hyperparameters.batch_size, \
                    self.hyperparameters._lambda), tabs = 2)

    theta = np.array(initial_theta)

    self.logger.log("Start with theta[1:10]: {}".format(theta[:10]), tabs = 1)
    cost_history = []
    gradient = None
    last_momentum = None
    last_theta = None
    for epoch in range(self.hyperparameters.epochs):
      for i in np.arange(0, train_X.shape[0], self.hyperparameters.batch_size):

        current_X = train_X[i : i + self.hyperparameters.batch_size, :]
        current_y = train_y[i : i + self.hyperparameters.batch_size]

        predicted = self.sigmoid(np.dot(current_X, theta))
        residual = predicted - current_y
        gradient = current_X.T.dot(residual) / current_X.shape[0]
        #gradient[1:] += self.hyperparameters._lambda * theta[1:] /  \
                      #current_X.shape[0]
        if last_momentum is not None:
          momentun = self.hyperparameters.speed * last_momentum + \
                self.hyperparameters.alpha * gradient
        else:
          momentun = self.hyperparameters.alpha * gradient

        theta -= momentun
        last_momentum = momentun


      cost_history.append(self.cost_function(train_X, train_y, theta))
      last_theta = theta

      if epoch == 0:
        continue

      diff = cost_history[-2] - cost_history[-1]

      if diff < 0:
        self.logger.log("At the end of epoch #{}, cost function increased by {:.4f}"
                      .format(epoch, abs(diff)), tabs = 1)
        self.hyperparameters.alpha /= 2
        theta = last_theta
        self.logger.log("Decresing alpha to {:.4f}"
                        .format(self.hyperparameters.alpha), tabs = 1)
      else:
        self.hyperparameters.alpha += self.hyperparameters.alpha * 0.05
        self.logger.log("Increasing alpha to {:.4f}"
                        .format(self.hyperparameters.alpha), tabs = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("Difference between iterations less than epsilon"
                      .format(abs(diff)), tabs = 1)
        self.logger.log("Stopped at epoch #{}".format(epoch), tabs = 1)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time to train was {:.3f}s".format(stop - start), tabs = 1)

    return stop - start

class BoostingSgdSolver():

  def __init__(self, hyperparameters, logger, epsilon):
    super().__init__(hyperparameters, logger, epsilon)

  def train(self, hyperparameters, initial_theta):
    pass



