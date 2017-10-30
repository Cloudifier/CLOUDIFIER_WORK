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
    self.logger.log("cost_eph#{} = {:.4f}; abs diff between current and last eph = {}"
                    .format(epoch, current_cost, abs(diff)), tabs = 1)
    self.logger.log("gradient[380:385] at eph#{} = {}"
                    .format(epoch, gradient[380:385]), tabs = 1)

  def get_accuracy_validation(self, theta, validation_X, validation_y):
    prediction = self.sigmoid(validation_X.dot(theta))
    corrects = 0
    wrongs = 0

    for i in range(validation_X.shape[0]):
      current_digit = int(validation_y[i])
      res = current_digit - np.argmax(prediction[i])

      if res == 0:
        corrects += 1
      else:
        wrongs += 1

    accuracy = 100 * corrects / float(corrects + wrongs)

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

    reg_string = "simple with reg"
    if self.hyperparameters._lambda != 0:
      reg_string = "simple without reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}"
      .format(reg_string, self.hyperparameters.alpha, self.hyperparameters.batch_size,\
              self.hyperparameters._lambda), tabs = 1)

    theta = np.array(initial_theta)
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
        self.hyperparameters.alpha /= 2
        theta = last_theta
        self.logger.log("eph#{}, cost increased by {:.4f} ==> decrease alpha to {:.4f}"
          .format(epoch, abs(diff), self.hyperparameters.alpha), tabs = 1)
      else:
        self.hyperparameters.alpha += self.hyperparameters.alpha * 0.05
        self.logger.log("eph#{}, cost decreased by {:.4f} ==> increasing alpha to {:.4f}"
                        .format(epoch, abs(diff), self.hyperparameters.alpha), tabs = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("eph#{}, delta(cost) < epsilon ==> early stopping"
                      .format(abs(diff)), tabs = 1)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for simple training = {:.3f}s".format(stop - start), tabs = 1)

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

    self.logger.log("Start simple training: alpha={}, batchSz={}, beta={}, momentum={}"
      .format(self.hyperparameters.alpha, self.hyperparameters.batch_size,\
              self.hyperparameters._lambda, self.hyperparameters.speed), tabs = 1)

    theta = np.array(initial_theta)
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
        self.hyperparameters.alpha /= 2
        theta = last_theta
        self.logger.log("eph#{}, cost increased by {:.4f} ==> decrease alpha to {:.4f}"
          .format(epoch, abs(diff, self.hyperparameters.alpha)), tabs = 1)
      else:
        self.hyperparameters.alpha += self.hyperparameters.alpha * 0.05
        self.logger.log("eph#{}, cost decreased by {:.4f} ==> increasing alpha to {:.4f}"
                        .format(epoch, abs(diff), self.hyperparameters.alpha), tabs = 1)


      if abs(diff) < self.epsilon:
        self.logger.log("eph#{}, delta(cost) < epsilon ==> early stopping"
                      .format(abs(diff)), tabs = 1)
        self.save_results(theta, cost_history, epoch + 1)
        break

      if epoch % 3 == 0:
        self.print_statistics(epoch, cost_history[-1], diff, gradient, theta)

    self.save_results(theta, cost_history, epoch + 1)

    stop = time.time()

    self.logger.log("Time for momentum training = {:.3f}s".format(stop - start), tabs = 1)

    return stop - start

class BoostingSgdSolver(SgdSolver):

  def __init__(self, hyperparameters, logger, epsilon):
    super().__init__(hyperparameters, logger, epsilon)

  def cost_function(self, X, y, theta):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))
    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta):
    start = time.time()
    self.hyperparameters.alpha = self.default_alpha

    theta = np.array(initial_theta)
    tmp_theta = np.array(initial_theta)
    last_y = train_y
    for boosts in range(3):
      predicted = self.sigmoid(np.dot(train_X, tmp_theta))
      residuals = predicted - last_y
      tmp_theta = sgd(train_X, residuals, np.zeros(train_X.shape[1]), 0.01, 5, 10)

      theta -= tmp_theta
      last_y = residuals

    stop = time.time()
    return theta

if __name__=='__main__':
  print("Library module. No main function")


