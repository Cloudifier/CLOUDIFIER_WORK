# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:36 2017

@author: Mihai.Cristea

"""
def normalEqn(X,y):
    """Normal Equation"""
    theta = np.ones(X.shape[1])
    theta = (np.linalg.pinv(X.T.dot(X)))* (X.T.dot(y));
    return theta

def featureNormalize(X):
    """Feature normalization"""
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0, ddof=1)
    m = X.shape[0] #m = len(X[:,0])    
    X_norm = (X - ml.repmat(mu,m,1)) / ml.repmat(sigma,m,1) 
    return X_norm, mu, sigma

    
def computeCost(X, y, theta):
    """calc of cost function"""
    m = len(y)
    J = 0
    m = len(X)
    predictions = np.dot(X,theta)
    sqrErrors = (predictions - y).A ** 2
    J = 1/(2*m) * sqrErrors.sum()
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """gradient descent function"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range (0, num_iters):
        theta = theta - alpha / m * X.T * (X * theta - y)
        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history

import numpy as np
#from numpy import np.newaxis, r_, c_, mat
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy.matlib as ml

data = np.loadtxt('ex1data2.txt', delimiter=',')
#X = data[:, 0:2][:, np.newaxis]
X = np.mat(data[:, 0:2][:, np.newaxis])
y = data[:, 2][:, np.newaxis]
m = X.shape[0]

#print(X[1:10,:])
print(np.column_stack( (X[:10], y[:10])))
#input('Press any key to continue\n')


X, mu, sigma = featureNormalize(X)

X = np.c_[np.ones((m)),X]
#print(np.round(X,4))
#print(np.round(mu,4))
#print(np.round(sigma,4))

theta = np.zeros((3,1))  # initialize fitting parameters
iterations = 50;
alpha = 0.1;

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print(theta)
print(J_history)
input('Press any key to continue\n')

# Plot the convergence graph
plt.plot(range(J_history.size), J_history, "-b", linewidth=2 )
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


# Display gradient descent's result
print('Theta computed from gradient descent: ')
print("{:f}, {:f}, {:f}".format(theta[0,0], theta[1,0], theta[2,0]))
print("")

# Estimate the price of a 1650 sq-ft, 3 br house
price = 0

price = ([1650, 3] - mu)/sigma
price = np.squeeze(np.asarray(price))
theta = np.squeeze(np.asarray(theta))
price = np.insert(price, 0, 1)

price = price.dot(theta)
print(theta)
print(price)
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ")
print(round(price,4))




# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')


data = np.loadtxt('ex1data2.txt', delimiter=',')
#X = data[:, 0:2][:, np.newaxis]
X = np.mat(data[:, 0:2][:, np.newaxis])
y = data[:, 2][:, np.newaxis]
m = X.shape[0]

# Add intercept term to X
#X = np.c_[np.ones(m, 1), X]
X = np.column_stack((np.ones((m,1)), X)) 

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print('\n', theta)
print('\n')


price = 0; 

price = [1, 1650, 3] * theta

# ============================================================

print(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price)
    
    
#https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
#https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.html

        