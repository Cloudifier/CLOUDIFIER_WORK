import numpy as np
import math as math
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pause():
    input("Hit <ENTER> key to continue...")

# ==================== Part 1: Basic Function ====================
def warmUpExercise():
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')
    print(np.identity(5))

# ======================= Part 2: Plotting =======================
def plotData(X, y):
    print('Plotting Data ...\n')
    plt.scatter(X, y, marker='x', color='red')

    # Add a legend
    plt.legend()
    # Show the plot
    plt.show()

# =================== Part 3: Cost and Gradient descent ===================
def costAndGradient(X, y):
    X = pd.DataFrame(np.ones(X.shape[0])).join(X)
    theta = np.zeros((2, 1))
    print(theta)

    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...\n')
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = ', J)
    print('Expected cost value (approx) 32.07\n')

    # further testing of thecost function
    J = computeCost(X, y, np.array([[-1], [2]]))
    print('\nWith theta = [-1 ; 2]\nCost computed = ', J)
    print('Expected cost value (approx) 54.24\n')

    print('\nRunning Gradient Descent ...\n')
    theta = gradientDescent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent:', theta)
    print('Expected theta values (approx): -3.6303 1.1664\n')
    plotDataLR(X, y, theta)

    #Predict values for population sizes of 35, 000 and 70, 000
    predict1 = np.matmul([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of %f\n', predict1 * 10000)
    predict2 = np.matmul([1, 7], theta)
    print('For population = 70,000, we predict a profit of %f\n', predict2 * 10000)

def computeCost(X, y, theta):
    J = 0
    cost = 0
    m = y.shape[0]
    ct = 1 / (2 * m)
    for i in range(m):
        prediction = 0
        for j in range(theta.shape[0]):
            prediction += X.iloc[i,j] * theta[j,0]
        cost += math.pow(prediction - y.iloc[i, 0], 2)
    return ct * cost

def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0]
    for iter in range(iterations):
        for i in range(m):
            prediction = 0
            for j in range(theta.shape[0]):
                prediction += X.iloc[i, j] * theta[j, 0]
            for j in range(theta.shape[0]):
                theta[j, 0] = theta[j, 0] - alpha / m * (prediction - y.iloc[i, 0]) * X.iloc[i, j]

    return theta

def plotDataLR(X, y, theta):
    print('Plotting Data ...\n')
    plt.scatter(X.ix[:,1], y, marker='x', color='red')

    plt.plot(X.ix[:,1], np.matmul(X, theta))
    # Add a legend
    plt.legend()
    # Show the plot
    plt.show()

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
def visualizeJ(X):
    # Grid over which we will calculate J
    X = pd.DataFrame(np.ones(X.shape[0])).join(X)
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    for i in range(theta0_vals.shape[0]):
        for j in range(theta1_vals.shape[0]):
            t = np.array([[theta0_vals[i]], [theta1_vals[i]]])
            J_vals[i, j] = computeCost(X, y, t)

    fig = plt.figure()
    ax = Axes3D(fig)
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# # PART 1
# data = pd.read_csv('ex1data1.txt', sep=",", header=None, names=('A','B'))
#
# X = pd.DataFrame(data.ix[:,0])
# y = pd.DataFrame(data.ix[:,1])
#
# warmUpExercise()
# print('Program paused. Press enter to continue.\n')
# #pause()
#
# print('Program paused. Press enter to continue.\n')
# #plotData(X,y)
# #pause()
#
# print('Program paused. Press enter to continue.\n')
# costAndGradient(X, y)
#
# print('Program paused. Press enter to continue.\n')
# visualizeJ(X)
#

# PART 2
def computeCostMulti(X, y, theta):
    m = X.shape[0]
    ct = 1 / (2 * m)
    cost = np.subtract(np.matmul(X, theta), y)
    J = ct * np.sum(np.power(cost, 2), axis=0)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = np.zeros((num_iters, 1))
    for i in range(1):
        theta = theta - alpha / m * (np.matmul(X.transpose(), np.subtract(np.matmul(X, theta), y)))
        J_history[i, 0] = computeCostMulti(X, y, theta)

    return [theta, J_history]

data = pd.read_csv('ex1data2.txt', sep=",", header=None, names=('A','B', 'C'))

X = pd.DataFrame(data.ix[:,0:2])

y = pd.DataFrame(data.ix[:,2])
X = pd.DataFrame(np.ones(X.shape[0])).join(X)

alpha = 0.1
num_iters = 100
theta = np.zeros((3, 1))
print(theta)
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

price = np.matmul([1, 1650, 3], theta);

print(price)
