# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:52:56 2017

@author: Mihai.Cristea
"""


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def costFunction(theta,X,y):
    """Calculam costul si gradientul"""
    m = X.shape[0]
    J = 0
    h = sigmoid (np.dot(X,theta))
    
    J = (1/m)* ((-np.dot(y.T,(np.log(h)))) - np.dot((1 - y).T,(np.log(1-h))))
    
    #grad = (1/m) * np.dot(X.T,(h-y))
    grad = (1/m) * np.dot((h.T - y), X).T
    
    return J, grad

def lrCostFunction(theta,X,y, lambda_reg):
    """Calculam costul si gradientul"""
    m = np.size(y)
    grad = np.zeros(np.size((theta)))
    J_base, grad = costFunction(theta, X, y)
    

    reg_cost = (lambda_reg / (2.0 * m)) * np.sum(theta[1:] ** 2)
    
    reg_gradient = (lambda_reg / m) * theta
    reg_gradient[0] = 0
    cost = J_base + reg_cost
    return cost, grad + reg_gradient

def one_vs_all(X, y, num_labels, lambda_reg):
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n))
    for c in range(1, num_labels + 1):
        initial_theta = np.zeros(n)
        target = np.vectorize(int)(y == c)
        result = opt.minimize(
            lrCostFunction,
            initial_theta,
            args=(X, target, lambda_reg),
            method='CG',
            jac=True,
            options={
                'maxiter': 50,
                'disp': False,
            }
        )
        theta = result.x
        cost = result.fun
        print('Training theta for label %d | cost: %f' % (c, cost))
        all_theta[c - 1, :] = theta
    return all_theta

def predict_one_vs_all(theta, X):
    # adaugam 1 pt ca index pleaca de la zero 
    return 1 + np.argmax(sigmoid(X.dot(theta.T)), axis=1)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt

data = sio.loadmat('ex3data1.mat')
#print(data)
X = np.array(data['X'])
y = np.array(data['y'])
y = y.flatten()
#plt.imshow(X[200,:].reshape(20,20), cmap = 'gray')
#plt.show()
#print(y)

m = X.shape[0]
#print(m)

#rand_indices = np.random.permutation(m)
#sel = X[rand_indices[:100],:]
#sel_disp = sel.reshape(10,10,400)

#for i in range(0,10):
#    for j in range (0,10):
#        plt.imshow(sel_disp[i,j,:].reshape(20,20), cmap = 'gray')
#        plt.show()

X = np.c_[np.ones((m, 1)), X]
all_theta = one_vs_all(X, y, 10, 0.01)
predictions = predict_one_vs_all(all_theta, X)
accuracy = np.sum(predictions == y)/y.shape[0] * 100
#accuracy = 100 * np.mean(predictions == y)
print('Train accuracy: %0.2f %%' % accuracy)
