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

def predict (X, theta1, theta2):
    z1 = X.dot(theta1)
    a1 = sigmoid(z1)

    z2 = a1.dot(theta2)
    a2 = sigmoid(z2)
    yhat = a2
    predict = (yhat >= 0.5)*1
    return predict  

import numpy as np
import matplotlib.pyplot as plt
alpha = 0.1
epochs = 5415
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

theta1 = np.random.rand(2,3)
theta2 = np.random.rand(3,1)
m = X.shape[0]
#X = np.c_[np.ones((m,1)),X]
err = np.zeros(epochs)
ep = np.zeros(epochs)
steps = epochs // 5 
for epoch in range (epochs):
    z1 = X.dot(theta1)
    a1 = sigmoid(z1)
    z2 = a1.dot(theta2)
    a2 = sigmoid(z2)

    yhat = a2

     
    #delta2 = y - yhat
    delta2 = yhat - y
    grd2 = 1/m * a1.T.dot(delta2)
    #grd2 = np.mean(delta2 * a1, axis=0)
    sig1 = a1 * (1 - a1)
    grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * sig1)
    #grd1 = 1/m * (delta2 * a1) 
    theta1 = theta1 - alpha * grd1
    theta2 = theta2 - alpha * grd2
 
    

    ep[epoch] = epoch
    err[epoch] = 1/m * np.sum((y-yhat)**2)
    #if (epoch % steps) == 0:
        #print( 'epoch: ', epoch, ' yhat : ', yhat, 'err :', err[epoch] )
    #print( 'epoch: ', epoch, ' yhat : ', yhat, 'accuracy :', np.sum(y==yhat)/y.shape[0] )
print( 'epoch: ', epoch, ' yhat : ', yhat, 'accuracy :', np.sum(y==yhat)/y.shape[0] )    
pred = predict(X,theta1,theta2)    
plt.plot(ep,err)
plt.title("MSE with bias")
plt.show()
print('theta1: \n', theta1)
print('theta2: \n', theta2)
print('Predict: \n', pred)
#a = np.c_[np.ones((m, 1)), sigmoid(X.dot(theta1))]