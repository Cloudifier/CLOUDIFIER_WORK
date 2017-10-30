# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:14:07 2017

@author: Mihai.Cristea
"""

def costFunctionReg (theta, X, y, lambda_reg):
    m = np.size(y)
    grad = np.zeros(np.size((theta)))
    J_base, grad = costFunction(theta, X, y)
    

    reg_cost = (lambda_reg / (2.0 * m)) * np.sum(theta[1:] ** 2)
    
    reg_gradient = (lambda_reg / m) * theta
    reg_gradient[0] = 0
    cost = J_base + reg_cost
    return cost, grad + reg_gradient

def mapFeature(X1,X2):
    degree = 6
#    col = 1
#    res = np.ones(( X1.shape[0], sum(range(degree + 2)) ))
#    #res = np.ones(( X1.shape[0]))
#    for i in range (1,degree +1):
#        for j in range (0,i+1):
#            #res = np.stack((res,(X1**(i-j))*(X2**j)),axis=j)
#            #res = np.hstack((res,(X1**(i-j))*(X2**j)))
#            res[:,col] = (X1**(i-j))*(X2**j)
#            col = col + 1
#            #print(res)
#    return res
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
#print (y.size)
#print (X)
#print (y)
#print(np.shape(X))
#print(np.shape(y))

plotData(X,y)
plt.show()

#input('Press any key to continue\n')

X = mapFeature(X[:,0], X[:,1])
#print(X)

m, n = X.shape

test_theta = np.ones(n);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

print('\nCost at test theta (with lambda = 10): ', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' %f \n', grad[0:5])
# ============ Part 2: Compute Cost and Gradient ============
m, n = X.shape
initial_theta = np.zeros(n)
lambda_reg = 1.0
cost, grad = costFunctionReg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta (zeros): %f' % cost)
result = opt.minimize(
    costFunctionReg,
    initial_theta,
    args=(X, y, lambda_reg),
    method='CG',
    jac=True,
    options={
        'maxiter': 400,
        'disp': False,
    }
)
theta = result.x
# plot the decision boundary
plotData(X_original, y)
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((u.size, v.size))
for i in range(u.size):
    for j in range(v.size):
        z[i, j] = mapFeature(u[i], v[j]).dot(theta)
        #z[i,j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])),theta)
        z=z.T
#plt.contour(u, v, z, [0.0, 0.0])
# Here is the grid range

plt.contour(u,v,z, levels=[0.0])
#plt.contour(u, v, z, levels=[0.0], linewidth=2)

plt.show()
predictions = predict(theta, X)
accuracy = 100 * np.mean(predictions == y)
print('Train accuracy: %0.2f %%' % accuracy)




