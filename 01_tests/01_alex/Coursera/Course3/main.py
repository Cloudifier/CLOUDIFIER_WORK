import numpy as np
import math as math
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(theta, X, y):
    m = y.shape[0]
    J = 0
    grad = np.zeros((theta.shape[0], 1))

    for i in range(m):
        sigmoidValue = sigmoid(np.matmul(X.ix[i,:], theta))
        J += 1 / m * (-y.iloc[i, 0] * np.log(sigmoidValue) - (1 - y.iloc[i, 0]) * np.log(1 - sigmoidValue))

        for j in range(3):
            grad[j, 0] += 1 / m * (sigmoidValue - y.iloc[i, 0]) * X.iloc[i, j]

    return [J, grad]

def gradientDescent(X, y, alpha, iterations):
    J = 0
    m = X.shape[0]
    theta = np.zeros(X.shape[1], 1)

    for iter in range(iterations):
        for i in range(m):
            sigmoidValue = sigmoid(np.matmul(X.ix[i, :], theta))

            for j in range(theta.shape[0]):
                theta[j, 0] = theta[j, 0] - alpha / m * (sigmoidValue - y.iloc[i, 0]) * X.iloc[i, j]

def costFunctionReg(theta, X, y, lmbd):
    m = y.shape[0]
    J = 0
    grad = np.zeros((theta.shape[0], 1))

    for i in range(m):
        sigmoidValue = sigmoid(np.matmul(X.ix[i, :], theta))
        J += 1 / m * (-y.iloc[i, 0] * np.log(sigmoidValue) - (1 - y.iloc[i, 0]) * np.log(1 - sigmoidValue))
        for j in range(theta.shape[0]):
            grad[j, 0] += 1 / m * (sigmoidValue - y.iloc[i, 0]) * X.iloc[i, j]
            if j > 1:
                grad[j, 0] += (lmbd / m) * theta[j, 0];

    J += lmbd / (2 * m) * np.sum(np.power(theta.iloc[1:,1], 2), axis=1)
    return [J, grad]

def mapFeatures(X1, X2):
    degree = 6;
    out = pd.DataFrame(np.ones((X1.shape[0], 1)))
    print(out)
    for i in range(degree):
        for j in range(i):
            out = out.join(pd.DataFrame(np.multiply(np.power(X1, (i - j)), np.power(X2, j))))
    return


data = pd.read_csv('ex2data1.txt', sep=",", header=None, names=('A','B', 'C'))

X = pd.DataFrame(data.ix[:,0:2])
y = pd.DataFrame(data.ix[:,2])
X = pd.DataFrame(np.ones(X.shape[0])).join(X)

[m, n] = X.shape

# Initialize fitting parameters
initial_theta = np.zeros((n, 1));
# Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
print('Cost at initial theta (zeros): %f', cost);
print('Expected cost (approx): 0.693');
print('Gradient at initial theta (zeros):', grad);
print('Expected gradients (approx):\n -0.1000 -12.0092 -11.2628');

test_theta = np.array([[-24], [0.2], [0.2]]);
[cost, grad] = costFunction(test_theta, X, y);
print('Cost at test theta:', cost);
print('Expected cost (approx): 0.218');
print('Gradient at test theta:', grad);
print('Expected gradients (approx): 0.043 2.566 2.647');


data = pd.read_csv('ex2data2.txt', sep=",", header=None, names=('A','B', 'C'))

X = pd.DataFrame(data.ix[:,0:2])
y = pd.DataFrame(data.ix[:,2])
X = pd.DataFrame(np.ones(X.shape[0])).join(X)

X = mapFeatures(X.ix[:,0], X.ix[:,2])