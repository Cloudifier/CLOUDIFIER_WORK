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

  def logreg_cross_entropy_cost(self, X, y, theta, beta):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))
    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + \
        (beta / (2 * m) * np.sum(theta[1:] ** 2))


  def logreg_mse_cost(self, X, y, theta, beta):
    m = y.size
    h = self.sigmoid(np.dot(X, theta))
    return (1 / (2 * m)) * np.sum((h-y)**2) + (beta / (2 * m)) * (np.sum(theta[1:]) ** 2)

  def linreg_mse_cost(self, X, y, theta, beta):
    m = y.size
    h = np.dot(X, theta)
    return 1 / (2 * m) * np.sum( (h - y) ** 2 ) + (beta / (2 * m) * np.sum(theta[1:] ** 2))


  @abstractmethod
  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,
    is_linear_reg = False, cost_function='MSE'):
    pass

class SimpleSgdSolver(SgdSolver):

  def __init__(self, logger, epsilon):
    super().__init__(logger, epsilon)
    self.logger.log("Initialize simple gradient descendent logisitic regression solver")

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,
    is_linear_reg  = False, cost_function='MSE'):

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
        yhat = np.dot(current_X, theta)

        if not is_linear_reg:
          predicted = self.sigmoid(yhat)
        else:
          predicted = yhat

        residual = predicted - current_y
        if (is_linear_reg) or ((not is_linear_reg) and (cost_function is 'CrossEntropy')):
          gradient = current_X.T.dot(residual) / current_X.shape[0]
          gradient[1:] += beta * theta[1:] / current_X.shape[0]
        else:
          factor = np.exp(yhat) / ((np.exp(yhat) + 1) ** 2)
          gradient = (current_X.T * factor).dot(residual) / current_X.shape[0]
          gradient[1:] += beta * theta[1:] / current_X.shape[0]

        #gradient = current_X.T.dot(residual) / current_X.shape[0]
        #gradient[1:] += beta * theta[1:] / current_X.shape[0]

        theta -= alpha * gradient

      if not is_linear_reg:
        if cost_function is 'CrossEntropy':
          cost_history.append(self.logreg_cross_entropy_cost(train_X, train_y, theta, beta))
        elif cost_function is 'MSE':
          cost_history.append(self.logreg_mse_cost(train_X, train_y, theta, beta))
      else:
        cost_history.append(self.linreg_mse_cost(train_X, train_y, theta, beta))

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
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,
    is_linear_reg  = False, cost_function='MSE'):

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
        yhat = np.dot(current_X, theta)

        if not is_linear_reg:
          predicted = self.sigmoid(yhat)
        else:
          predicted = yhat

        residual = predicted - current_y
        if (is_linear_reg) or ((not is_linear_reg) and (cost_function is 'CrossEntropy')):
          gradient = current_X.T.dot(residual) / current_X.shape[0]
          gradient[1:] += beta * theta[1:] / current_X.shape[0]
        else:
          factor = np.exp(yhat) / ((np.exp(yhat) + 1) ** 2)
          gradient = (current_X.T * factor).dot(residual) / current_X.shape[0]
          gradient[1:] += beta * theta[1:] / current_X.shape[0]

        if last_momentum is not None:
          momentun = speed * last_momentum + alpha * gradient
        else:
          momentun = alpha * gradient

        theta -= momentun
        last_momentum = momentun

      if not is_linear_reg:
        if cost_function is 'CrossEntropy':
          cost_history.append(self.logreg_cross_entropy_cost(train_X, train_y, theta, beta))
        elif cost_function is 'MSE':
          cost_history.append(self.logreg_mse_cost(train_X, train_y, theta, beta))
      else:
        cost_history.append(self.linreg_mse_cost(train_X, train_y, theta, beta))

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

  def __init__(self, logger, epsilon, solver):
    super().__init__(logger, epsilon)
    self.solver = solver

  def train(self, train_X, train_y, validation_X, validation_y, initial_theta,
    epochs = 0, alpha = 0, batch_size = 0, beta = 0, speed = 0, n_boost = 0,
    is_linear_reg = False):

    start = time.time()

    reg_str = "boosting without reg"
    if beta != 0:
      reg_str = "boosting with reg"

    self.logger.log("Start {} training: alpha={}, batchSz={}, beta={}, n_boost={}"
      .format(reg_str, alpha, batch_size, beta, n_boost), verbosity_level = 2)

    theta = np.array(initial_theta)
    current_theta = np.array(initial_theta)
    current_y = train_y


    self.cost_history = []
    self.epochs_to_convergence = 0
    gradient = None

    for it in range(n_boost):
      self.logger.log("Boost #{}".format(it+1), verbosity_level = 2)
      self.solver.train(train_X, current_y, validation_X, validation_y, initial_theta, epochs, alpha, batch_size, beta, speed, 0, True)

      current_theta = self.solver.model

      predicted = np.dot(train_X, current_theta)
      residual = predicted - current_y
      residual = np.array([i if abs(i) > 5 else 0 for i in residual])

      print("Predicted {}".format(predicted[:5]))
      print("Residual {}".format(residual[:5]))

      self.cost_history.append(self.solver.cost_history)
      self.epochs_to_convergence += self.solver.epochs_to_convergence

      current_y = residual
      if it == 0:
        theta = current_theta
        #self.cost_history = self.solver.cost_history
        #self.epochs_to_convergence = self.solver.epochs_to_convergence
      else:
        theta -= current_theta

    self.model = theta

    stop = time.time()
    self.logger.log("Time for {} training = {:.3f}s".format(reg_str, stop - start))

    self.cost_history = np.hstack(self.cost_history).tolist()

    return stop - start

if __name__=='__main__':
  print("Library module. No main function")


