import numpy as np
import scipy.io as sio
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def lrCostFunction(X, y, theta, lmbd):
#     [m, n] = X.shape
#     sigmoid_value = sigmoid(np.matmul(X, theta))
#     J = 1 / m * np.sum((- y * np.log(sigmoid_value) - (1 - y) * np.log(1 - sigmoid_value)), axis=0) + lmbd / (2 * m) * np.sum(np.power(theta[1:,0], 2), axis=0)
#     grad = 1 / m * np.matmul(X.transpose(), np.subtract(sigmoid_value, y))
#
#     temp = theta
#     temp[0, 0] = 0
#     grad = grad + lmbd / m * temp
#
#     return [J, grad]

def lrCostFunction(X, y, theta, lmbd):
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

def gradientDescent(X, y, theta, iterations, alpha, lmbd):
    J_min = 1000000000
    m, n = X.shape
    theta = np.zeros((X.shape[1], 1))
    gd_results = list()
    for i in range(iterations):
        reg_theta = theta[:]
        reg_theta[0, 0] = 0
        reg_term = lmbd / m * reg_theta
        grad_term = 1 / m * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta = theta - alpha * (grad_term + reg_term)
        J = lrCostFunction(X, y, theta, lmbd)
        #gd_results.append(GDResult(i, J, theta))
        if(J < J_min):
            J_min = J
            theta_min = theta
    return theta_min


# def gradientDescent(X, y, theta, iterations, alpha, lmbd):
#     #TODO: this can be optimized to used a dict type dataset to store cost-theta values, get the min cost value index and get tetha on that index
#     J_min = 1000000000
#     theta_min = np.array((theta.shape))
#     for i in range(iterations):
#         [J, grad] = lrCostFunction(X, y, theta, lmbd)
#         theta = theta - alpha * grad
#         if(J < J_min):
#             J_min = J
#             theta_min = theta
#
#     return theta_min

def oneVsAll(X, y, num_labels, iterations, alpha, lmbd):
    [m,n] = X.shape
    all_thetas = np.zeros((n, num_labels))
    for i in range(num_labels):
        theta_label = np.zeros((n, 1))
        theta_label = gradientDescent(X, (y == (i + 1)).astype(int), theta_label, iterations, alpha, lmbd)
        all_thetas[:,i] = theta_label.ravel()

    return all_thetas

def predictOneVsAll(all_thetas, X):
    [Xm, Xn] = X.shape
    [tm, tn] = all_thetas.shape
    predictions = np.zeros((Xm, 1))
    for i in range(Xm):
        temp_predict = np.zeros(tn)
        for j in range(tn):
            temp_predict[j] = (X[i,:].dot(all_thetas[:,j]))
        predictions[i] = temp_predict.argmax() + 1

    return predictions

def predictNN(theta1, theta2, X):
    a = 2

mat_contents = sio.loadmat('ex3data1.mat')

X = mat_contents['X']
y = mat_contents['y']
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

#theta_t = np.array([[-2], [-1], [1], [2]])
#X_t = np.array([[1, 0.1, 0.6, 1.1], [1, 0.2, 0.7, 1.2], [1, 0.3, 0.8, 1.3],[1, 0.4, 0.9, 1.4], [1, 0.5, 1, 1.5]])
#y_t = np.array([[1], [0], [1], [0], [1]])
#lambda_t = 3
#[J, grad] = lrCostFunction(X_t, y_t, theta_t, lambda_t)

#print('Cost: ', J)
#print('Expected cost: 2.534819')
#print('Gradients:', grad)
#print('Expected gradients: 0.146561 -0.548558 0.724722 1.398003')


# ============ Part 2b: One-vs-All Training ============
#print('\nTraining One-vs-All Logistic Regression...\n')
num_labels = 10
lmbd = 0.1
iterations = 200
alpha = 0.9

all_theta = oneVsAll(X, y, num_labels, iterations, alpha, lmbd)

# ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)

print('\nTraining Set Accuracy: ', (pred == y).astype(int).sum() / X.shape[0] * 100);

# =========== Part 1: Loading and Visualizing Data =============
input_layer_size  = 400
hidden_layer_size = 25


weights = sio.loadmat('ex3weights.mat')
#print(weights.shape)
#pred = predict(Theta1, Theta2, X)

