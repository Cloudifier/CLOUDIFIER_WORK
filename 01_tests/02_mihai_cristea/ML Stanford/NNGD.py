# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:40:45 2017

@author: Mihai.Cristea
"""

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

#def sigmoidPrime(z):
#    g = np.exp(-z)/((1 + np.exp(-z))**2)
#    return g

def sigmoidPrime(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g

#def gradientDescent(X,y,theta1, theta2, lr = 0.1, epochs = 10):
#    
#    
#    
#    #forward propagation
#
#    for epoch in range (epochs):
#        a = sigmoid(np.dot(X, theta1))
#        yhat = sigmoid(np.dot(a,theta2))
#        E = y - yhat
#        delta2 = lr * E
#        theta2 += np.dot(a.T,delta2)
#        delta1 = np.dot(delta2, theta2.T) * sigmoidPrime(np.dot(X,theta1))
#        theta1 +=  np.dot(X.T,delta1)
#        print( 'epoch: ', epoch, ' yhat : ', yhat)
#    

import numpy as np
import matplotlib.pyplot as plt
lr = 0.1
epochs = 15000
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

theta1 = np.random.uniform(size=(3,3))
theta2 = np.random.uniform(size=(3,1))
m = X.shape[0]
X = np.c_[np.ones((m,1)),X]
err = np.zeros(epochs)
ep = np.zeros(epochs)
#gradientDescent(X,y,theta1, theta2)
for epoch in range (epochs):

    a = sigmoid(np.dot(X, theta1))
    #a = np.c_[np.ones((m, 1)), sigmoid(X.dot(theta1))]
    yhat = sigmoid(np.dot(a,theta2))
    E = y - yhat
    delta2 = lr * E
    theta2 += np.dot(a.T,delta2)
    delta1 = np.dot(delta2, theta2.T) * sigmoidPrime(np.dot(X,theta1))
    theta1 +=  np.dot(X.T,delta1)
    ep[epoch] = epoch
    err[epoch] = sum(np.power((y-yhat),2))
    print( 'epoch: ', epoch, ' yhat : ', yhat, 'err :', sum((y-yhat)**2) )
    #print( 'epoch: ', epoch, ' yhat : ', yhat, 'accuracy :', np.sum(y==yhat)/y.shape[0] )
    
plt.plot(ep,err)
plt.show()