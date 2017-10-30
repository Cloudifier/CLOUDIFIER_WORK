# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:30:36 2017

@author: Mihai.Cristea

"""

def computeCost(X, y, theta):
    """calc of cost function"""
    m = len(y)
    J = 0
    m = len(X)
    predictions = np.dot(X,theta)
    #predictions = np.dot(X,theta)
    sqrErrors = (predictions - y) ** 2
    J = 1/(2*m) * sqrErrors.sum()
    #J = 1/(2*m)*np.sum(np.square(h-y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    """gradient descent function"""
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range (0, num_iters):
        theta = theta - alpha / m * X.T.dot((np.dot(X, theta) - y))
        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history

import numpy as np
#from numpy import np.newaxis, r_, c_, mat
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#dataset = pd.read_csv('ex1data1.txt', header=None)
#plt.plot(dataset)
#plt.show

#print(dataset)
#X = dataset.iloc[:,0].values

#X = np.array(dataset.iloc[:,0].values)[:, np.newaxis]
#y = np.array(dataset.iloc[:,1].values)[:, np.newaxis]
#m = len(y)

data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.array(data[:, 0][:, np.newaxis])
y = np.array(data[:, 1][:, np.newaxis])
m = X.shape[0]

print(m)

plt.plot(X,y,'rx', markersize=7)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()
theta = np.zeros((2,1))  # initialize fitting parameters
#theta = np.zeros(2)[:, np.newaxis]
print('theta :', theta)
X = np.c_[np.ones((m)),X]

iterations = 1500;
alpha = 0.01;
print(len(X))

input('Press any key to continue\n')
z = computeCost(X,y,theta)
print('Cost initial :', z)

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
input('Press any key to continue\n')

print('theta dupa iteratii: \n', theta)
input('Press any key to continue\n')
plt.plot(X[:, 1],y,'rx', markersize=7)
plt.plot(X[:, 1],X.dot(theta), '-')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend(['Training data', 'Linear regression'])
plt.show()
input('Press any key to continue\n')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range (len(theta0_vals)):
    for j in range (len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = computeCost(X, y, t)

J_vals = J_vals.T


theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals) # necessary for 3D graph
#theta1_vals, theta0_vals = np.meshgrid(theta1_vals, theta0_vals) # necessary for 3D graph
# Surface plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta1_vals, theta0_vals,  J_vals, rstride=8, cstride=8, alpha=0.3,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_0$')
ax.set_zlabel(r'J($\theta$)')
plt.show()

input('Press any key to continue\n')
# # Contour plot

plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=5)
plt.xlabel(r'$\Theta_0$'); plt.ylabel(r'$\Theta_1$')
plt.show()



        