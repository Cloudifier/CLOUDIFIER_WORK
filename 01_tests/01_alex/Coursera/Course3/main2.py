import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GDResult():
    def __init__(self, iteration, cost, theta):
        self.iteration = iteration
        self.cost = cost
        self.theta = theta

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def gradientDescent(X, y, alpha, lmbd, iterations):
    m, n = X.shape
    theta = np.zeros((X.shape[1], 1))
    gd_results = list()
    for i in range(iterations):
        reg_theta = theta[:]
        reg_theta[0, 0] = 0
        reg_term = lmbd / m * reg_theta
        grad_term = 1 / m * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta = theta - alpha * (grad_term + reg_term)
        J = calculateCostFunction(X, y, theta, lmbd)
        gd_results.append(GDResult(i, J, theta))

    return gd_results

def calculateCostFunction(X, y, theta, lmbd):
    sigmoid_value = sigmoid(X.dot(theta))
    m, n = X.shape
    theta_reg = theta[:]
    theta_reg[0,0] = 0
    cost_term = 1 / m * (- y * (np.log(sigmoid_value)) - (1 - y) * np.log(1 - sigmoid_value))
    #cost_term = 0
    reg_term = lmbd / (2 * m) * (theta_reg ** 2)
    #reg_term = 0
    J = np.sum(cost_term) + np.sum(reg_term)

    return J

def mapFeatures(X):
    degree = 6
    m, n = X.shape
    out = np.ones((m,1))
    for i in range(degree):
        for j in range(degree):
            out = np.append(out, 0, np.ones(m), axis=1)



def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

def featureNormalization(X):
    mean = np.mean(X, axis=0)
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return (X - mean) / (max - min)

def getData():
    data = pd.read_csv("ex2data1.txt")
    X = np.array(data.iloc[:,0:2])
    X = featureNormalization(X)
    m, n = X.shape
    y = np.array(data.iloc[:, 2]).reshape(m, 1)
    X = np.insert(X, 0, np.ones(m), axis=1)

    return X, y


alpha = 0.1
lmbd = 10
iterations = 400
X, y = getData()

gd_results = gradientDescent(X, y, alpha, lmbd, iterations)
x_values = [x.iteration for x in gd_results]
y_values = [x.cost for x in gd_results]

min_cost = sorted(gd_results, key = lambda x:x.cost, reverse=True)[1]

print(gd_results[-1].theta)
print(min_cost.iteration)
print(min_cost.cost)
print(min_cost.theta)

theta_min = min_cost.theta

# plt.plot(x_values, y_values)
# plt.show()

# print([x.theta for x in gd_results])

p = predict(X, theta_min)
print((p == y).sum() / len(y) * 100)