import numpy as np
from sklearn.model_selection import train_test_split
import os
from scipy.special import expit

def minmax_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  os_home = os.path.expanduser("~")
  data_home = os.path.join(os_home, 'Google Drive/_cloudifier_data/09_tests/_MNIST_data')
  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home=data_home)

  X = mnist.data
  y = mnist.target

  X = minmax_scaler(X)
  X = np.c_[np.ones(y.size), X]
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.33,
                                                      random_state=42)

  data = { 'train': { 'X': X_train, 'y': y_train },
           'test': { 'X': X_test, 'y': y_test } }

  return data

def sigmoid(z):
  return expit(z)


def Dsigmoid(z):
  return sigmoid(z) * (1 - sigmoid(z))

def CrossEntropy(y, yhat):
  m = yhat.shape[0]

  J = y.T.dot(np.log(yhat)) + (1 - y).T.dot(np.log(1 - yhat))
  J = -J / (m)
  return J

def relu(z):
    a = np.array(z)
    return np.maximum(0,a)

def Drelu(z):
    a = (z > 0).astype(int)
    return a

architecture = [784, 28, 1]
learningRate = .01

theta_1 = np.random.randn(architecture[0]+1, architecture[1]) * np.sqrt(2/(architecture[0]+1))
theta_2 = np.random.randn(architecture[1]+1, architecture[2]) * np.sqrt(2/(architecture[1]+1))

np.set_printoptions(precision=5, suppress=True)
yhat = None

if __name__ == "__main__":
  data = fetch_data()

  data['train']['y'] = (data['train']['y'] == 1)
  data['test']['y'] = (data['test']['y'] == 1)

  y = np.array(data['train']['y'])[:500]
  y = y.reshape(-1,1)
  X = np.array(data['train']['X'])[:500]
  n_epochs = 2000
  for i in range(n_epochs):
    if i%100:
      print("Start epoch {}".format(i))
    # forward propagation
    z1 = np.dot(X, theta_1)
    a1 = relu(z1)
    a1 = np.c_[np.ones(X.shape[0]), a1]
    z2 = np.dot(a1, theta_2)
    yhat = sigmoid(z2)
    residual = yhat - y
    #print("Epoch {}: (FProp) --> error={}".format(i, CrossEntropy(y, yhat)))
    #####################

    # backward propagation
    delta_Z = residual
    delta_H = delta_Z.dot(theta_2[1:, :].T) * Drelu(z1)

    gradient_2 = a1.T.dot(delta_Z) / a1.shape[0]
    gradient_1 = X.T.dot(delta_H) / X.shape[0]

    theta_2 -= gradient_2 * learningRate
    theta_1 -= gradient_1 * learningRate
    #print("(BProp) --> gradient_1 = {};  gradient_2 = {}\n".format(gradient_1.reshape(-1)[:5], gradient_2.reshape(-1)[:5]))

  #print(y[:10])
  #print(yhat[:10])
  print("Error={}".format(CrossEntropy(y,yhat)))