import numpy as np
from sklearn.metrics import mean_squared_error
from time import time
import matplotlib.pyplot as plt
np.set_printoptions(precision=1, suppress=True)

class LinearRegression:
  def __init__(self, logger, VERBOSITY=1):
    self.logger = logger
    self.VERBOSITY = VERBOSITY
    self.cost_history = list()

  def MSE(self, y_pred, y):
    return mean_squared_error(y, y_pred) / 2

  def plot_cost_history(self):
    plt.plot(np.arange(0, len(self.cost_history)), self.cost_history)
    plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()

  def train(self, train_X, train_y, learning_rate=0.01, epochs=25, batch_size=10):
    total_train_time = 0
    theta = np.zeros(train_X.shape[1] + 1).reshape(-1,1)
    m = train_y.shape[0]
    train_X = np.c_[ np.ones(m), train_X]

    self.logger._log("Training linreg model (initialized with 0)... epochs={}, alpha={:.2f}, batch_sz={}."
                    .format(epochs, learning_rate, batch_size))

    n_batches = train_X.shape[0] // batch_size

    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch+1, epochs))
      epoch_start_time = time()
      for i in range(n_batches):
        batch_X = train_X[(i * batch_size):((i + 1) * batch_size), :]
        batch_y = train_y[(i * batch_size):((i + 1) * batch_size)]

        y_pred = np.dot(batch_X, theta)
        residual = y_pred - batch_y

        if i % 10 == 0:
          self.logger._log('   [TRAIN Minibatch: {}] loss: {:.2f}'.format(i, self.MSE(y_pred, batch_y)))
          if self.VERBOSITY >= 10:
            d1_slice = batch_y.reshape(batch_y.size)[:3]
            d2_slice = y_pred.reshape(y_pred.size)[:3]
            self.logger._log('        yTrue:{}'.format(d1_slice))
            self.logger._log('        yPred:{}'.format(d2_slice))

        gradient = batch_X.T.dot(residual) / batch_X.shape[0]
        theta -= learning_rate * gradient
      epoch_time = time() - epoch_start_time
      total_train_time += epoch_time

      J = self.MSE(np.dot(train_X, theta), train_y)
      self.cost_history.append(J)
      self.logger._log('{:.2f}s - train_loss: {:.3f}\n'.format(epoch_time, J))

    self.logger._log('Total TRAIN time: {:.2f}s'.format(total_train_time))
    self.model = np.array(theta)


  def predict(self, test_X, test_y):
    m = test_y.shape[0]
    test_X = np.c_[np.ones(m), test_X]

    y_pred = np.dot(test_X, self.model)
    J = self.MSE(y_pred, test_y)
    self.logger._log("Predicting ... test_loss: {:.3f}".format(J))

    fig, ax = plt.subplots()
    ax.scatter(test_y, y_pred)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    return y_pred