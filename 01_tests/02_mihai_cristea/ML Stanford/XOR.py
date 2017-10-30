import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model

class GDResult():
    def __init__(self, iteration, cost, theta):
        self.iteration = iteration
        self.cost = cost
        self.theta = theta


def normalEquation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

def costFunction(X, theta, y):
    m, n = X.shape
    #J = 1 / (2 * m) * np.sum((X.dot(theta) - y) ** 2)
    J = 1/(2*m)*np.sum(np.square(X.dot(theta)-y))
    return J

def gradientDescent(X, y, alpha, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    gd_results = list()

    for i in range(iterations):
        theta = theta - alpha/m * X.T.dot(X.dot(theta) - y)
        J = costFunction(X, theta, y)
        gd_results.append(GDResult(i, J, theta))

    return gd_results

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

X = np.c_[np.ones((4,1)),X]

theta_normal = normalEquation(X, y)
gd_results = gradientDescent(X, y, 0.01, 200)
min_cost = sorted(gd_results, key = lambda x:x.cost)[0]
x_values = [x.iteration for x in gd_results]
y_values = [x.cost for x in gd_results]

plt.plot(x_values, y_values)
plt.show()

print("### Values calculated with custom code ###")
print("Min cost calculated with GD: ", min_cost.cost)
print("Theta for min cost calculated with GD: ")
print(min_cost.theta)
print("Min cost calculated with normal equation: ", )
print("Theta for min cost calculated with GD: ", costFunction(X, theta_normal, y))
print(theta_normal)

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)
theta2 = regr.coef_
m, n = X.shape
#J = 1 / (2 * m) * np.sum((X.dot(theta) - y) ** 2)

# The coefficients
print('Theta calculated with sklearn: \n', regr.coef_)
