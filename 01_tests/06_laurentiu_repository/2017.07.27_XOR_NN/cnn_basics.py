import numpy as np
from scipy.special import expit

def CrossEntropy(y, yhat):
  m = yhat.shape[0]

  J = y.T.dot(np.log(yhat)) + (1 - y).T.dot(np.log(1 - yhat))
  J = -J / (m)
  return J


class ConvLayer():
  def __init__(self):
    # HyperParameters #
    self.H = 28
    self.W = 28
    self.C = 1
    self.filter_size = 3
    self.stride = 1
    self.padding = (self.filter_size - 1) / 2
    self.num_filters = 64

    self.map_height = int((self.H - self.filter_size + 2 * self.padding) / self.stride + 1)
    self.map_width = int((self.W - self.filter_size + 2 * self.padding) / self.stride + 1)
    ####################

    self.weights = np.random.randn(self.num_filters, self.C, self.filter_size, self.filter_size)
    self.biases = np.random.rand(self.num_filters, 1)
    self.z_maps = np.zeros((self.num_filters, self.map_height, self.map_width))
    self.a_maps = None


  def relu(self, z):
    a = np.array(z)
    return np.maximum(0,a)

  def Drelu(self, z):
    a = (z > 0).astype(int)
    return a

  def convolve(self, X):
    self.z_maps = self.z_maps.reshape((self.num_filters, self.map_height * self.map_width))
    self.a_maps = self.a_maps.reshape((self.num_filters, self.map_height * self.map_width))

    """
    for i in range(self.num_filters):
      row = 0
      col = 0

      for j in range(self.map_height * self.map_width):
        self.z_maps[i][j] = np.sum(X[row:self.filter_size+row, col:self.filter_size+col] * self.weights[i]) + self.biases[i]
        col += self.stride

        if (self.filter_size + col) > self.W:
          col = 0
          row += self.stride

    self.z_maps.reshape((self.num_filters, self.map_height, self.map_width))
    self.a_maps = self.relu(self.z_maps)
    """

