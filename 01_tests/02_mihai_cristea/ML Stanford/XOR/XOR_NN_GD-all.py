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

def relu(x):
    return np.maximum(0, x)

def reluPrime(x):
    return np.where(x > 0, 1.0, 0.0)

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


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
X = np.c_[np.ones((m,1)),X]

alpha = 0.1
epochs = 20000
#mse bias
max_mse_bias = []
i = 0
print("MSE_BIAS", flush=True)
for i in range(10):
    theta1 = np.random.rand(3,4)
    theta2 = np.random.rand(4,1)

    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = sigmoid(z1)
        #a1 = np.c_[np.ones((m, 1)), a1]
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
        yhat = a2
         
        delta2 = 2 * (y - yhat) * (-yhat) * (1 - yhat)
        grd2 = 1/m * a1.T.dot(delta2)
        sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * sig1)
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2     
        pred = predict(X,theta1,theta2)
        
        ep[epoch] = epoch
        err[epoch] = 1/m * np.sum((y-yhat)**2)
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch), flush=True)
            max_mse_bias.append(epoch)
            break


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
max_mse_nobias = []
print("MSE_NO_BIAS", flush=True)
i = 0
for i in range(10):
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
            
        delta2 = 2 * (y - yhat) * (-yhat) * (1 - yhat)
        grd2 = 1/m * a1.T.dot(delta2)

        sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * sig1)
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2
        ep[epoch] = epoch
        err[epoch] = 1/m * np.sum((y-yhat)**2)
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch), flush=True)
            max_mse_nobias.append(epoch)
            break    
        
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
max_cross_nobias = []
print("CROSS_NO_BIAS")
i = 0
for i in range(10):
    theta1 = np.random.rand(2,3)
    theta2 = np.random.rand(3,1)
    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = sigmoid(z1)
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
    
        yhat = a2
    
         
        #delta2 = (y - yhat) *(-1)
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
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch))
            max_cross_nobias.append(epoch)
            break         
        
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
X = np.c_[np.ones((m,1)),X]
max_cross_bias = []
print("CROSS_BIAS")
i = 0
for i in range(10):
    theta1 = np.random.rand(3,4)
    theta2 = np.random.rand(4,1)
    m = X.shape[0]
    
    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = sigmoid(z1)
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
    
        yhat = a2
    
         
        delta2 = yhat - y
        grd2 = 1/m * a1.T.dot(delta2)
        sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * sig1)
        #grd1 = 1/m * (delta2 * a1) 
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch))
            max_cross_bias.append(epoch)
            break  


#max_mse_bias = np.array(max_mse_bias)
#print('Max epochs for mse bias sigmoid :  ', np.max(max_mse_bias))
#max_mse_nobias = np.array(max_mse_nobias)
#print('Max epochs for mse no bias sigmoid :  ', np.max(max_mse_nobias))
#max_cross_nobias = np.array(max_cross_nobias)
#print('Max epochs for cross no bias sigmoid :  ', np.max(max_cross_nobias))
#max_cross_bias = np.array(max_cross_bias)
#print('Max epochs for cross bias sigmoid :  ', np.max(max_cross_bias))

#============================ReLU=====================================

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
X = np.c_[np.ones((m,1)),X]

alpha = 0.01
epochs = 20000
#mse bias
max_mse_bias_relu = []
i = 0
print("MSE_BIAS_RELU", flush=True)
for i in range(10):
    theta1 = np.random.rand(3,4)
    theta2 = np.random.rand(4,1)

    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = relu(z1)
        #a1 = np.c_[np.ones((m, 1)), a1]
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
        yhat = a2
         
        delta2 = 2 * (y - yhat) * (-yhat) * (1 - yhat)
        grd2 = 1/m * a1.T.dot(delta2)
        #sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * reluPrime(a1))
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2     
        pred = predict(X,theta1,theta2)
        
        ep[epoch] = epoch
        err[epoch] = 1/m * np.sum((y-yhat)**2)
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch), flush=True)
            max_mse_bias_relu.append(epoch)
            break


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
max_mse_nobias_relu = []
print("MSE_NO_BIAS_RELU", flush=True)
i = 0
for i in range(10):
    theta1 = np.random.rand(2,3)
    theta2 = np.random.rand(3,1)
    m = X.shape[0]
    #X = np.c_[np.ones((m,1)),X]
    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = relu(z1)
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
    
        yhat = a2
            
        delta2 = 2 * (y - yhat) * (-yhat) * (1 - yhat)
        grd2 = 1/m * a1.T.dot(delta2)

        sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * reluPrime(a1))
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2
        ep[epoch] = epoch
        err[epoch] = 1/m * np.sum((y-yhat)**2)
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch), flush=True)
            max_mse_nobias_relu.append(epoch)
            break    
        
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
max_cross_nobias_relu = []
print("CROSS_NO_BIAS_RELU")
i = 0
for i in range(10):
    theta1 = np.random.rand(2,3)
    theta2 = np.random.rand(3,1)
    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = relu(z1)
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
    
        yhat = a2
    
         
        #delta2 = (y - yhat) *(-1)
        delta2 = yhat - y
        grd2 = 1/m * a1.T.dot(delta2)
        #grd2 = np.mean(delta2 * a1, axis=0)
        #sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * reluPrime(a1))
        #grd1 = 1/m * (delta2 * a1) 
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2        
        ep[epoch] = epoch
        err[epoch] = 1/m * np.sum((y-yhat)**2)
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch))
            max_cross_nobias_relu.append(epoch)
            break   




X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
m = X.shape[0]
X = np.c_[np.ones((m,1)),X]
max_cross_bias_relu = []
print("CROSS_BIAS_RELU")
i = 0
for i in range(10):
    theta1 = np.random.rand(3,4)
    theta2 = np.random.rand(4,1)
    m = X.shape[0]
    
    err = np.zeros(epochs)
    ep = np.zeros(epochs)
    steps = epochs // 5 
    for epoch in range (epochs):
        z1 = X.dot(theta1)
        a1 = relu(z1)
        z2 = a1.dot(theta2)
        a2 = sigmoid(z2)
    
        yhat = a2
    
         
        delta2 = yhat - y
        grd2 = 1/m * a1.T.dot(delta2)
        #sig1 = a1 * (1 - a1)
        grd1 = 1/m * X.T.dot(delta2.dot(theta2.T) * reluPrime(a1))
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2
        if(((yhat > 0.5).astype(int)) == y).sum() == 4:
            print('  Convergenta la epoch: ' + str(epoch))
            max_cross_bias_relu.append(epoch)
            break  

max_mse_bias = np.array(max_mse_bias)
print('Max epochs for mse bias sigmoid :  ', np.max(max_mse_bias))
max_mse_nobias = np.array(max_mse_nobias)
print('Max epochs for mse no bias sigmoid :  ', np.max(max_mse_nobias))
max_cross_nobias = np.array(max_cross_nobias)
print('Max epochs for cross no bias sigmoid :  ', np.max(max_cross_nobias))
max_cross_bias = np.array(max_cross_bias)
print('Max epochs for cross bias sigmoid :  ', np.max(max_cross_bias))
max_mse_bias_relu = np.array(max_mse_bias_relu)
print('Max epochs for mse bias relu :  ', np.max(max_mse_bias_relu))
max_mse_nobias_relu = np.array(max_mse_nobias_relu)
print('Max epochs for mse no bias relu :  ', np.max(max_mse_nobias_relu))
max_cross_nobias_relu = np.array(max_cross_nobias_relu)
print('Max epochs for cross no bias relu :  ', np.max(max_cross_nobias))
max_cross_bias_relu = np.array(max_cross_bias_relu)
print('Max epochs for cross bias relu :  ', np.max(max_cross_bias_relu))