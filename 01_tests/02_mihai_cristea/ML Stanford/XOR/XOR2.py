def normalEquation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


def computeCost(X, y, theta):
    m = y.size
    J = 0   
    h = X.dot(theta) 
    J = 1/(2*m)*np.sum(np.square(h-y))   
    return(J)

def gradientDescent(X, y,  theta, alpha=0.01, num_iters=250):
    m = y.size
    J_history = np.zeros(num_iters)    
    for iter in np.arange(num_iters):
        print('theta before iter: ', iter,' theta : \n', theta)
        theta = theta - alpha*(1/m)*(X.T.dot(X.dot(theta)-y))
        print('theta after iter: ', iter,' theta : \n', theta)
        J_history[iter] = computeCost(X, y, theta)
        #input('Press any key to continue\n')
    return(theta, J_history)



import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model

#class GDResult():
#    def __init__(self, iteration, cost, theta):
#        self.iteration = iteration
#        self.cost = cost
#        self.theta = theta





X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

X = np.c_[np.ones((4,1)),X]
theta = np.zeros((3,1))
# theta for minimized cost J
theta , Cost_J = gradientDescent(X, y, theta)


plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations');

theta_normal = normalEquation(X, y)
gd_results = Cost_J
min_cost = np.argmin(Cost_J, axis=0)


print("### Values calculated with custom code ###")
print('theta: \n',theta)
print("Min cost calculated with GD: ", Cost_J[min_cost])


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)
theta2 = regr.coef_
m, n = X.shape
#J = 1 / (2 * m) * np.sum((X.dot(theta) - y) ** 2)
#J = 1/(2*m)*np.sum(np.square(X.dot(theta2.T)-y))
# The coefficients
print('Theta calculated with sklearn: \n', regr.coef_)
#print("Min cost calculated with sklearn theta: ", J)