import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1)


mnist = fetch_mldata('mnist-original')
X = mnist.data
y = mnist.target
n_classes = np.unique(y).count()
Y = np.zeros((y.shape[0], n_classes))
for i in range(y.shape[0]):
    Y[i, y[i]] = 1

m, n = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

m_train, n_features = X_train.shape
theta = np.zeros((n_features, n_classes))

batch_size = 1000
epochs = 5


for epoch in range(epochs):
    iter = m_train // batch_size
    for i in range(iter):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start: batch_end]
        y_batch = Y[batch_start: batch_end]
        z = X_batch.dot(theta)
        yhat = softmax(z)

