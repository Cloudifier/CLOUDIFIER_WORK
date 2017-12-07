import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(z, 0)

def reluPrime(z):
    return (relu(z) > 0).astype(int)

def gradientDescent(X_orig, y_orig, theta1_shape, theta2_shape, cost, func):
    min_values = list()
    for i in range(2):
        X = X_orig[:]
        y = y_orig[:]
        theta1 = np.random.rand(theta1_shape[0], theta1_shape[1])
        theta2 = np.random.rand(theta2_shape[0], theta2_shape[1])

        m, n = X.shape
        alpha = 0.01
        for i in range(200000):
            z1 = X.dot(theta1)
            if func == '___':
              a1 = relu(z1)
            else:
              a1 = sigmoid(z1)

            z2 = a1.dot(theta2)
            a2 = 
            yhat = a2

            if (((yhat > 0.5).astype(int)) == y).sum() == 4:
                print('Convergenta la iteratia: ' + str(i))
                min_values.append(i)
                break

            if cost == '_____':
                delta2 = 2 * (yhat - y) *  ____
            else:
                delta2 = (yhat - y)

            grad2 = _____
            grad1 = (delta2.dot(theta2.T) * reluPrime( ____  )).T.dot(X).T * 1 / m

            theta1 = theta1 - alpha * grad1
            theta2 = theta2 - alpha * grad2

    return min_values

values = list()
print('1. Convergenta pentru crossentropy cu bias: ')
#1. crossentropy_withbias
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
crossentropy_withbias = gradientDescent(X, y, (3, 4), (4, 1), 'cross')
if len(crossentropy_withbias) > 1:
    print('Max: ' + str(np.max(crossentropy_withbias)))
else:
    print('Max: no max found ?')

print('2. Convergenta pentru crossentropy fara bias: ')
#2. crossentropy_withoutbias
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
crossentropy_withoutbias = gradientDescent(X, y, (2, 3), (3, 1), 'cross')
if len(crossentropy_withoutbias) > 1:
    print('Max: ' + str(np.max(crossentropy_withoutbias)))
else:
    print('Max: no max found ?')

print('3. Convergenta pentru mse cu bias: ')
#3. mse with bias
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
mse_withbias = gradientDescent(X, y, (3, 4), (4, 1), 'mse')
if len(mse_withbias) > 1:
    print('Max: ' + str(np.max(mse_withbias)))
else:
    print('Max: no max found ?')

print('4. Convergenta pentru mse fara bias: ')
#4. mse without bias
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
mse_withoutbias = gradientDescent(X, y, (2,3), (3,1), 'mse')
if len(mse_withbias) > 1:
    print('Max: ' + str(np.max(mse_withoutbias)))
else:
    print('Max: no max found ?')
