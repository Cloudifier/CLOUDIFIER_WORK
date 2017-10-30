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
    #g = np.zeros(z.shape)
    #g = 1 / (1+e**(-z))
    g = 1 / (1 + np.exp(-z))
    return g


def costFunction(theta,X,y):
    """Calculam costul si gradientul"""
    m = y.size
    J = 0
    grad = np.zeros(theta.size)
    
    h = sigmoid (np.dot(X,theta))
    
    J = (1/m)* ((-np.dot(y.T,(np.log(h)))) - np.dot((1 - y).T,(np.log(1-h))))
    
    #grad = (1/m) * np.dot(X.T,(h-y))
    grad = (1/m) * np.dot((h.T - y), X).T
    
    return J, grad

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

data = pd.read_csv('ex2data1.txt', header = None)

X = np.array(data.iloc[:,0:2].values)
y = np.array(data.iloc[:,2].values)
#print (y.size)
#print (X)
#print (y)
#print(np.shape(X))
#print(np.shape(y))


plotData(X,y)
plt.show()

#input('Press any key to continue\n')

# ============ Part 2: Compute Cost and Gradient ============
[m,n] = np.shape(X)
print(m)
print(n)

X = np.c_[np.ones((m,1)),X]
#print(X)
initial_theta = np.zeros((n+1,1))
print(initial_theta)

#print(sigmoid(np.dot(X.T,initial_theta)))


cost, grad = costFunction(initial_theta, X, y)

print("cost: ", cost)
print("gradient", grad)
#input('Press any key to continue\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
[cost, grad] = costFunction(test_theta, X, y)
print("cost at non zero theta: ", cost)
print("gradient at non zero theta", grad)
#input('Press any key to continue\n')


# ============= Part 3: Optimizing using fminunc equivalent


result = opt.fmin_tnc(func=costFunction, x0=initial_theta,  args=(X, y))  
theta = result[0]

date = costFunction(theta,X,y)
cost = date[0]
print ('Cost at theta found by fminunc: %f', cost)
print ('theta: %s', theta)
# plot the decision boundary ============================
plot_x = np.array([X[:, 1].min() - 2, X[:, 1].max() + 2])
plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
plotData(X[:, 1:], y)
plt.plot(plot_x, plot_y)
plt.show()

# ============== Part 4: ==================================

prob = sigmoid (np.dot(np.array([1, 45, 85]), theta))
print ('For a student with scores 45 and 85, we predict an admission probability: ', prob)
#
p = predict(theta, X)
p = np.mean(p==y) * 100

print ('Accuracy : ',p)





