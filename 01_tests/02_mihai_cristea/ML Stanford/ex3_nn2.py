# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:37:53 2017

@author: Mihai.Cristea
"""


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def disp(X):
    plt.imshow(X.reshape(20,20), cmap = 'gray')
    plt.show()

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.c_[np.ones((m, 1)), X]    
    A2 = np.c_[np.ones((m, 1)), sigmoid(X.dot(Theta1.T))]
    A3 = sigmoid(A2.dot(Theta2.T))
    predictions = 1 + np.argmax(A3, axis=1)
    return predictions


import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat


print('Loading and Visualizing Data ...')
data = loadmat('ex3data1.mat')
X = data['X']
#y = data['y'].flatten()
y = data['y']
m = X.shape[0]
sel = np.random.permutation(X)[:100]

print('Loading Saved Neural Network Parameters ...')
weights = loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

predictions = predict(Theta1, Theta2, X)
accuracy = 100 * np.mean(predictions == y.reshape(1,len(y)))
print('Training set accuracy: %0.2f %%' % accuracy)
#input('Press any key to continue\n')
random_X = np.random.permutation(X)
for i in range(m):
    example = random_X[i].reshape(1, -1)
    prediction = predict(Theta1, Theta2, example)
    print('Prediction: %d (digit %d)' % (prediction, prediction % 10))
    #disp(X, i)
    disp(example)