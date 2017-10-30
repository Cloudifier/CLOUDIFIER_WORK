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

def CrossEntropy_OneHot(y, yhat):
  m = yhat.shape[0]

  J = np.sum(y * np.log(yhat))
  J = -J / (m)
  return J


def relu(z):
  a = np.array(z)
  return np.maximum(0,a)

def Drelu(z):
  a = (z > 0).astype(int)
  return a

def OneHot(targets, classes=10):
  return np.eye(classes)[targets.reshape(-1)]

def softmax(z):
  z -= np.max(z)
  sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
  return sm

architecture = [784, 28, 10]
learningRate = .01

theta_1 = np.random.randn(architecture[0]+1, architecture[1]) * np.sqrt(2/(architecture[0]+1))
theta_2 = np.random.randn(architecture[1]+1, architecture[2]) * np.sqrt(2/(architecture[1]+1))

np.set_printoptions(precision=2, suppress=True)
yhat = None

if __name__ == "__main__":
  data = fetch_data()

  y = np.array(data['train']['y'])[:500]
  y = OneHot(y.astype(int))
  X = np.array(data['train']['X'])[:500]
  n_epochs = 2000
  last_momentum_1 = None
  last_momentum_2 = None
  speed = 0.9
  for i in range(n_epochs):
    if i%100:
      print("Start epoch {}".format(i))
    # forward propagation
    z1 = np.dot(X, theta_1)
    a1 = relu(z1)
    a1 = np.c_[np.ones(X.shape[0]), a1]
    z2 = np.dot(a1, theta_2)
    yhat = softmax(z2)
    residual = yhat - y
    #print("Epoch {}: (FProp) --> error={}".format(i, CrossEntropy(y, yhat)))
    #####################

    # backward propagation
    delta_Z = residual
    delta_H = delta_Z.dot(theta_2[1:, :].T) * Drelu(z1)

    gradient_2 = a1.T.dot(delta_Z) / a1.shape[0]
    gradient_1 = X.T.dot(delta_H) / X.shape[0]

    if last_momentum_2 is not None:
      momentum_2 = speed * last_momentum_2 + learningRate * gradient_2
    else:
      momentum_2 = learningRate * gradient_2

    if last_momentum_1 is not None:
      momentum_1 = speed * last_momentum_1 + learningRate * gradient_1
    else:
      momentum_1 = learningRate * gradient_1


    theta_2 -= momentum_2
    theta_1 -= momentum_1

    last_momentum_2 = momentum_2
    last_momentum_1 = momentum_1
    #print("(BProp) --> gradient_1 = {};  gradient_2 = {}\n".format(gradient_1.reshape(-1)[:5], gradient_2.reshape(-1)[:5]))

  #print(y[:10])
  #print(yhat[:10])
  print("Error={}".format(CrossEntropy_OneHot(y,yhat)))