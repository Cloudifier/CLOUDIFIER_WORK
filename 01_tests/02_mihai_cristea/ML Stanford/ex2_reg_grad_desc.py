# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:14:07 2017

@author: Mihai.Cristea
"""

def costFunction(theta, X, y, lambda_reg):
    h = sigmoid(np.dot(X,theta))
    m = X.shape[0]
    unu = y * np.log(h)
    doi = (1-y) * np.log(1 - h)
    final = - unu - doi 
    #final = - 1/m*( unu + doi) + (lambda_reg / (2.0 * m)) * np.sum(theta ** 2)
    final = final + (lambda_reg / (2.0 )) * np.sum(theta ** 2)
    #final = - 1*( unu + doi) + (lambda_reg / (2.0 )) * theta.T.dot(theta)
    #return final 1/m*
    return np.mean(final)
    


def gradientDescent(X, y, theta, alpha=.01, lambda_reg = 1, epochs = 100):
    """gradient descent function"""
    #m = len(y)
    m = X.shape[0]
    n = X.shape[1]
    cost_iter = []
    print('theta shape',theta.shape)
    #print(theta, X, y)
    theta = theta - alpha / m * X.T.dot((sigmoid(np.dot(X, theta)) - y))
    i=1
    print( 'pre epoch theta : ', theta.shape)
    for epoch in range(1,epochs):
#        print( 'pre epoch: ', epoch, ' theta : ', theta.shape,'(sigmoid(np.dot(X, theta)', \
#        (sigmoid(np.dot(X, theta))-y).shape, 'X.T',  X.T.dot((sigmoid(np.dot(X, theta)) - y)).shape)
        theta = theta - alpha * (1. / m * X.T.dot((sigmoid(np.dot(X, theta)) - y)) + (lambda_reg/n * theta)) 
        cost = costFunction(theta, X, y, lambda_reg)
        print( 'epoch: ', epoch, ' theta : ', theta.shape, ' cost : ', cost)
        #input('Press any key to continue\n')
        cost_iter.append([i, cost])
        i+=1

    return theta, np.array(cost_iter)


def mapFeature(X1,X2):
    degree = 6
    m = X1.shape[0] if X1.shape else 1
    cols = np.ones(m)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            cols = np.vstack((cols,((X1 ** (i - j)) * (X2 ** j))))
    #return np.vstack(cols).T
    return cols.T

def predict (theta, X):
    m = np.size(X,1)
    p = np.array(np.zeros((m,1)))
    p = sigmoid(np.dot(X,theta))>=0.5
    
    return p


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def plotData(X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)

    plt.plot(X[pos,0], X[pos,1], marker = 'o', markersize = 7, color = 'y')
    plt.plot(X[neg,0], X[neg,1], marker = 'x', markersize = 7, color = 'k')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from numpy import e
import scipy.optimize as opt  

data = pd.read_csv('ex2data2.txt', header = None)

X = np.array(data.iloc[:,0:2].values)
y = np.array(data.iloc[:,2].values)
X_original = X
y = y.reshape((118,1))
X = mapFeature(X[:,0], X[:,1])

#print(X)
#print(X.shape)
m, n = X.shape
#initial_theta = np.zeros((n,1))
X = np.c_[np.ones((m,1)),X]
initial_theta = np.zeros((n+1,1))
lambda_reg = 1.0
theta, J = gradientDescent( X, y, initial_theta)

plt.plot(J[:,0],J[:,1])
plt.show()

pred = predict(theta, X)
accuracy = (y==pred).sum()/y.shape[0]
print('Train accuracy: %0.2f %%' % accuracy)




