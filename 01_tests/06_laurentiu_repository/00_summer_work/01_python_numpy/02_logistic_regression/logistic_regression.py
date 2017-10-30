import numpy as np
from time import time
np.set_printoptions(precision=1, suppress=True)
import matplotlib.pyplot as plt

class LogisticRegression:
  def __init__(self, logger, VERBOSITY=1):
    self.logger = logger
    self.VERBOSITY = VERBOSITY
    self.train_cost_history = list()
    self.validation_cost_history = list()
    self.classes = 10
    self.beta = 0
    self.theta = None

  def OneHotMatrix(self, y, classes=10):
    n_obs = y.shape[0]
    ohm = np.zeros((n_obs, classes))
    ohm[np.arange(n_obs), y.astype(int)] = 1
    return ohm

  def softmax(self, z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm

  def CrossEntropy(self, y_pred, y_ohm):
    m = y_ohm.shape[0]
    loss = (-1 / m) * np.sum(y_ohm * np.log(y_pred)) + (self.beta / 2) * np.sum(self.theta[1:] ** 2)
    return loss

  def computeLossAndGradient(self, X, y):
    m = X.shape[0]
    y_ohm = self.OneHotMatrix(y)
    y_pred = self.softmax(np.dot(X, self.theta))
    loss = self.CrossEntropy(y_pred, y_ohm)

    residual = y_pred - y_ohm

    gradient = X.T.dot(residual) / m
    gradient[1:] += self.beta * self.theta[1:] / m

    return y_pred, loss, gradient

  def ThetaNoBias(self):
    return self.theta[1:, :]

  def train(self, X_train, y_train, X_validation=None, y_validation=None,
            epochs=15, learning_rate=0.01, batch_size=10, beta=0,
            momentum_speed=0, decay_factor = 1):
    total_train_time = 0
    self.classes = len(np.unique(y_train))
    self.theta = np.zeros((X_train.shape[1] + 1, self.classes))
    self.beta = beta

    m = y_train.shape[0]
    X_train = np.c_[np.ones(m), X_train]
    if X_validation is not None:
      X_validation = np.c_[np.ones(y_validation.shape[0]), X_validation]

    self.logger._log("Training logreg model (initialized with 0)... epochs={}, alpha={:.2f}, batch_sz={}, beta={}, momentum={}, decay={}"
                     .format(epochs, learning_rate, batch_size, beta, momentum_speed, decay_factor))

    n_batches = X_train.shape[0] // batch_size

    last_momentum = None
    lr_patience = 0
    lr_plateau = 5
    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()

      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]

        y_pred, loss, gradient = self.computeLossAndGradient(X_batch, y_batch)

        if i % 1000 == 0:
          self.logger._log('   [TRAIN Minibatch: {}] loss: {:.2f}'.format(i, loss))
          if self.VERBOSITY >= 10:
            y_pred = np.argmax(y_pred, axis=1)
            n_to_slice = batch_size
            if n_to_slice > 10:
              n_to_slice = 10
            d1_slice = y_batch.reshape(y_batch.size)[:n_to_slice]
            d2_slice = y_pred.reshape(y_pred.size)[:n_to_slice]
            self.logger._log('        yTrue:{}'.format(d1_slice.astype(int)))
            self.logger._log('        yPred:{}'.format(d2_slice))

        if last_momentum is not None:
          momentum = momentum_speed * last_momentum + learning_rate * gradient
        else:
          momentum = learning_rate * gradient

        self.theta -= momentum
        last_momentum = momentum

      epoch_time = time() - epoch_start_time
      total_train_time += epoch_time

      y_pred_train = self.softmax(np.dot(X_train, self.theta))
      J_train = self.CrossEntropy(y_pred_train, self.OneHotMatrix(y_train))
      self.train_cost_history.append(J_train)
      acc_train = np.sum(y_train == np.argmax(y_pred_train, axis=1)) / float(y_train.shape[0])

      if (X_validation is not None) and (y_validation is not None):
        y_pred_validation = self.softmax(np.dot(X_validation, self.theta))
        J_valid = self.CrossEntropy(y_pred_validation, self.OneHotMatrix(y_validation))
        acc_valid = np.sum(y_validation == np.argmax(y_pred_validation, axis=1)) / float(y_validation.shape[0])

        if (epoch > 0) and (decay_factor != 1):
          if J_valid >= self.validation_cost_history[-1]:
            lr_patience += 1
            if self.VERBOSITY >= 10:
              self.logger._log('curr_loss >= last_loss - Increase lr_patience to {}'.format(lr_patience))
          else:
            if (self.validation_cost_history[-1] - J_valid) <= 1e-4:
              lr_patience += 1
              if self.VERBOSITY >= 10:
                self.logger._log('loss decreased slowly - Increase lr_patience to: {}'.format(lr_patience))

          if lr_patience >= lr_plateau:
            lr_patience = 0
            learning_rate *= decay_factor
            if self.VERBOSITY >= 10:
              self.logger._log('lr_patience == {} - alpha: {:.3f}'.format(lr_plateau, learning_rate))

        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}% - val_loss: {:.2f} - val_acc: {:.2f}%\n'
                         .format(epoch_time, J_train, acc_train * 100, J_valid, acc_valid * 100))
        self.validation_cost_history.append(J_valid)

      else:
        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}%\n'.format(epoch_time, J_train, acc_train * 100))

    self.logger._log('Total TRAIN time: {:.2f}s'.format(total_train_time))


  def predict(self, X_test, y_test):
    m = y_test.shape[0]
    X_test = np.c_[np.ones(m), X_test]

    probabilities = self.softmax(np.dot(X_test, self.theta))
    preds = np.argmax(probabilities, axis=1)
    accuracy = sum(preds == y_test) / (float(len(y_test)))
    J_test = self.CrossEntropy(probabilities, self.OneHotMatrix(y_test))

    self.logger._log("Predicting ... test_loss: {:.2f} - test_acc: {:.2f}%".format(J_test, accuracy * 100))

    return preds

  def plot_cost_history(self, cost_history):
    plt.plot(np.arange(0, len(cost_history)), cost_history)
    #plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()

  def plot_thetas(self):
    theta_no_bias = self.ThetaNoBias()

    fig = plt.figure(figsize = (20, 10))
    fig.suptitle('Parametrii theta')

    ax1 = fig.add_subplot(2, 5, 1)
    ax1.imshow(theta_no_bias[:, 0].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax1.set_title(str(0))

    ax2 = fig.add_subplot(2, 5, 2)
    ax2.imshow(theta_no_bias[:, 1].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax2.set_title(str(1))

    ax3 = fig.add_subplot(2, 5, 3)
    ax3.imshow(theta_no_bias[:, 2].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax3.set_title(str(2))

    ax4 = fig.add_subplot(2, 5, 4)
    ax4.imshow(theta_no_bias[:, 3].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax4.set_title(str(3))

    ax5 = fig.add_subplot(2, 5, 5)
    ax5.imshow(theta_no_bias[:, 4].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax5.set_title(str(4))

    ax6 = fig.add_subplot(2, 5, 6)
    ax6.imshow(theta_no_bias[:, 5].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax6.set_title(str(5))

    ax7 = fig.add_subplot(2, 5, 7)
    ax7.imshow(theta_no_bias[:, 6].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax7.set_title(str(6))

    ax8 = fig.add_subplot(2, 5, 8)
    ax8.imshow(theta_no_bias[:, 7].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax8.set_title(str(7))

    ax9 = fig.add_subplot(2, 5, 9)
    ax9.imshow(theta_no_bias[:, 8].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax9.set_title(str(8))

    ax10 = fig.add_subplot(2, 5, 10)
    ax10.imshow(theta_no_bias[:, 9].reshape(28, 28), cmap = 'gray', interpolation = 'nearest')
    ax10.set_title(str(9))

    fig.savefig('thetas.png', dpi = fig.dpi)
    plt.show()


if __name__=='__main__':
  print("Library module. No main function")
