# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:14:07 2017

@author: Mihai.Cristea
"""


def predict (theta, X):
    m = np.size(X,1)
    p = np.array(np.zeros((m,1)))
    p = sigmoid(np.dot(X,theta))>=0.5    
    return p

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoid2(theta,X):
    g = 1 / (1 + np.exp(-X.dot(theta)))

    return g

def featureNormalize(X):
    """Feature normalization"""
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_norm

def costFunction(theta, X, y):
    h = sigmoid2(theta,X)
    y = np.squeeze(y)
    unu = y * np.log(h)
    doi = (1-y) * np.log(1 - h)
    final = - unu - doi    
    return np.mean(final)
    


def gradientDescent(X, y, theta, alpha=.001, epochs = 250):
    """gradient descent function"""
    m = len(y)
    cost_iter = []
    print(theta.shape)
    #J_history = np.zeros(1)
    cost = costFunction(theta, X, y)
    i=0
    #change_cost=1
    
    for epoch in range(epochs):
        #old_cost = cost
        print( 'epoch: ', epoch, ' theta : ', theta.shape,'(sigmoid(np.dot(X, theta)', \
        sigmoid(np.dot(X, theta)).shape, 'X.T', X.T.dot(sigmoid(np.dot(X,theta))).shape)
        theta = theta - alpha / m * X.T.dot((sigmoid(np.dot(X, theta)) - y))#lipsea sigmoid
        #print(theta.shape)
        print( 'epoch: ', epoch, ' theta : ', theta.shape)
        cost_iter.append([i, cost])
        cost = costFunction(theta, X, y)
        print( 'epoch: ', epoch, ' theta : ', theta.shape, ' cost : ', cost)
        #np.append(J_history, cost, axis=0)
        #change_cost = old_cost - cost
        i+=1
        #print('theta: \n', theta)
        #print('cost ', cost_iter)
    #return theta, J_history
    return theta, np.array(cost_iter)


def plotData(X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
#print (np.shape(pos))
#plt.plot(X((y==1),0), X((y==1),1))
    plt.plot(X[pos,0], X[pos,1], marker = 'o', markersize = 7, color = 'y')
    plt.plot(X[neg,0], X[neg,1], marker = 'x', markersize = 7, color = 'k')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import e
import scipy.optimize as opt 
import math
#import numpy.matlib as ml 

data = pd.read_csv('ex2data1.txt', header = None)

X = np.array(data.iloc[:,0:2].values)
X=featureNormalize(X)
y = np.array(data.iloc[:,2].values)
y = y.reshape(100,1)
#print (y.size)
#print (X)
#print (y)
#print(np.shape(X))
#print(np.shape(y))


#plotData(X,y)
plt.show()

#input('Press any key to continue\n')

# ============ Part 2: Compute Cost and Gradient ============
[m,n] = np.shape(X)
#print(m)
#print(n)

X = np.c_[np.ones((m,1)),X]
#print(X)
initial_theta = np.zeros((n+1,1))

first_calc = X.T.dot(sigmoid2(initial_theta, X) - np.squeeze(y))



theta, J = gradientDescent(X, y, initial_theta )

#print("theta final: ", theta)
#print("gradient final", J)

plt.plot(J[:,0],J[:,1])
plt.show()

pred = predict(theta,X)

rez = (y==pred).sum()/y.shape[0]

print('rez:', rez)
#input('Press any key to continue\n')
