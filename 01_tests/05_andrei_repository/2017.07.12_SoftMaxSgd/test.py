import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

def binTargets(Y):
    zeros = [[0 for i in range(10)] for i in range(Y.shape[0])]
    return np.array([zero[:int(i)] + [1] + zero[int(i)+1:] for zero,i in zip(zeros,Y)])

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getLoss(X, y, theta, beta):
    m = X.shape[0]
    y_mat = binTargets(y)
    h = softmax(np.dot(X, theta))
    loss = (-1 / m) * np.sum(y_mat * np.log(h)) + (beta/2)*np.sum(theta ** 2)

    residual = y_mat - h

    gradient = -X.T.dot(residual) / m
    gradient[1:] += beta * theta[1:] / m

    return loss,gradient

def sgd(X, y, learningRate, n_epochs, batchSz, reg):
  theta = np.zeros((X.shape[1], 10))

  for epoch in range(n_epochs):
    for i in np.arange(0, X.shape[0], batchSz):
      #print(i)
      newX = X[i:i+batchSz, :]
      newY = y[i:i+batchSz]

      loss,grad = getLoss(newX,newY,theta,reg)

      theta -= learningRate * grad
    #end for i
    print("Done epoch {}".format(epoch))
    #print("Epoch #{0}, cost = {1}".format(epoch, cost_function(X, y, theta, reg)))
  #end for epoch

  return theta

def getProbsAndPreds(someX,w):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getAccuracy(someX,someY, w):
    prob,prede = getProbsAndPreds(someX, w)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy


if __name__ == "__main__":
  mnist = fetch_mldata('MNIST original', data_home="data")
  print("Ajung aici")
  cols = []
  for i in range(1, 785):
    cols.append("pixel_" + str(i))

  df = pd.DataFrame(data = np.c_[mnist.target, mnist.data],
                    columns = ['Target'] + cols)

  X = np.array(df.iloc[:, 1:].values)
  y = np.array(df.iloc[:, 0].values)

  m = y.size
  min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
  X = min_max_scaler.fit_transform(X)
  X = np.c_[np.ones(m), X]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
  print("Ajung")
  theta = sgd(X_train, y_train, 0.01, 15, 5, 0)
  print('Training Accuracy: ', getAccuracy(X_train,y_train, theta))
  print('Test Accuracy: ', getAccuracy(X_test,y_test, theta))