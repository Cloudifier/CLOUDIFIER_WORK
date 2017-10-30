import numpy as np
from scipy.special import expit


def sigmoid(z):
    return expit(z)


def Dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def MSE(y, yhat):
    m = yhat.shape[0]

    J = np.sum((yhat-y)**2)
    J = J / (2 * m)
    return J


def CrossEntropy(y, yhat):
    m = yhat.shape[0]

    J = y.T.dot(np.log(yhat)) + (1 - y).T.dot(np.log(1 - yhat))
    J = -J / (m)
    return J

def relu(z):
    a = np.array(z)
    return np.maximum(0,a)

def Drelu(z):
    a = (z > 0).astype(int)
    return a

architecture = [2, 2, 1]

n_epochs_MSE = 100
n_epochs_log = 100
learningRate = .5

X = np.array([[1, 0,0],[1, 0,1],[1, 1,0],[1, 1,1]])
y = np.array([[0],[1],[1],[0]])

low = -0.5
high = 0.5
theta_1 = np.random.uniform(low=low, high=high, size=(architecture[0]+1, architecture[1]))
theta_2 = np.random.uniform(low=low, high=high, size=(architecture[1]+1, architecture[2]))

theta_1_MSE = np.array(theta_1)
theta_2_MSE = np.array(theta_2)

theta_1_log = np.array(theta_1)
theta_2_log = np.array(theta_2)

theta_1_relu = np.random.randn(architecture[0]+1, architecture[1]) * np.sqrt(2/(architecture[0]+1))
theta_2_relu = np.random.randn(architecture[1]+1, architecture[2]) * np.sqrt(2/(architecture[1]+1))
 
np.set_printoptions(precision=5, suppress=True)

yhat = None
print("MSE")
print("===")
for i in range(n_epochs_MSE):

    # forward propagation
    z1 = np.dot(X, theta_1_MSE)
    a1 = sigmoid(z1)
    a1 = np.c_[np.ones(X.shape[0]), a1]
    z2 = np.dot(a1, theta_2_MSE)
    yhat = sigmoid(z2)
    residual = yhat - y

    if i == 99:
        print("Epoch {}: (FProp) --> target={};  yhat={};  residual={};  error={:.5f}"
            .format(i, y.reshape(-1), yhat.reshape(-1), residual.reshape(-1), MSE(y, yhat)))

    #####################

    # backward propagation
    delta_Z = residual * Dsigmoid(z2)
    delta_H = delta_Z.dot(theta_2_MSE[1:,:].T) * Dsigmoid(z1)

    gradient_2 = a1.T.dot(delta_Z) / a1.shape[0]
    gradient_1 = X.T.dot(delta_H) / X.shape[0]

    theta_2_MSE -=  gradient_2 * learningRate
    theta_1_MSE -=  gradient_1 * learningRate
    if i == 99:
        print("(BProp) --> gradient_1 = {};  gradient_2 = {}\n".format(gradient_1.reshape(-1), gradient_2.reshape(-1)))
    ########################

print(yhat)

yhat = None
print("Cross-Entropy | Sigmoid")
print("========================")
for i in range(n_epochs_log):

    # forward propagation
    z1 = np.dot(X, theta_1_log)
    a1 = sigmoid(z1)
    a1 = np.c_[np.ones(X.shape[0]), a1]
    z2 = np.dot(a1, theta_2_log)
    yhat = sigmoid(z2)
    residual = yhat - y

    if i == 99:
        print("Epoch {}: (FProp) --> target={};  yhat={};  residual={};  error={:.5f}"
            .format(i, y.reshape(-1), yhat.reshape(-1), residual.reshape(-1), MSE(y, yhat)))
    #####################

    # backward propagation
    delta_Z = residual
    delta_H = delta_Z.dot(theta_2_log[1:, :].T) * Dsigmoid(z1)

    gradient_2 = a1.T.dot(delta_Z) / a1.shape[0]
    gradient_1 = X.T.dot(delta_H) / X.shape[0]

    theta_2_log -= gradient_2 * learningRate
    theta_1_log -= gradient_1 * learningRate

    if i == 99:
        print("(BProp) --> gradient_1 = {};  gradient_2 = {}\n".format(gradient_1.reshape(-1), gradient_2.reshape(-1)))
    #####################


print(yhat)


yhat = None
print("Cross-Entropy | Relu")
print("========================")
for i in range(60000):

    # forward propagation
    z1 = np.dot(X, theta_1_relu)
    a1 = relu(z1)
    a1 = np.c_[np.ones(X.shape[0]), a1]
    z2 = np.dot(a1, theta_2_relu)
    yhat = sigmoid(z2)
    residual = yhat - y

    if i == 99:
        print("Epoch {}: (FProp) --> target={};  yhat={};  residual={};  error={:.5f}"
            .format(i, y.reshape(-1), yhat.reshape(-1), residual.reshape(-1), MSE(y, yhat)))
    #####################

    # backward propagation
    delta_Z = residual
    delta_H = delta_Z.dot(theta_2_relu[1:, :].T) * Drelu(z1)

    gradient_2 = a1.T.dot(delta_Z) / a1.shape[0]
    gradient_1 = X.T.dot(delta_H) / X.shape[0]

    theta_2_relu -= gradient_2 * learningRate
    theta_1_relu -= gradient_1 * learningRate
    if i == 99:
        print("(BProp) --> gradient_1 = {};  gradient_2 = {}\n".format(gradient_1.reshape(-1), gradient_2.reshape(-1)))
    #####################


print(yhat)
