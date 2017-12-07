import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
m,n = X.shape

theta1 = np.random.rand(3, 4)
theta2 = np.random.rand(4, 1)

values = list()
alpha = 0.01
for i in range(100000):
    z1 = X.dot(theta1)
    a1 = sigmoid(z1)

    z2 = a1.dot(theta2)
    a2 = sigmoid(z2)
    yhat = a2

    delta2 = yhat - y
    grad2 = np.reshape(np.mean(delta2 * a1, axis=0), theta2.shape)
    grad1 = (delta2.dot(theta2.T) * a1 * (1 - a1)).T.dot(X).T * 1 / m

    theta1 = theta1 - alpha * grad1
    theta2 = theta2 - alpha * grad2
    E = np.sum((y-yhat) ** 2)
    values.append((i, E, (theta1, theta2)))

x = [x[0] for x in values]
y = [x[1] for x in values]
theta1, theta2 = sorted(values, key=lambda x: x[1])[0][2]

z1 = X.dot(theta1)
a1 = sigmoid(z1)

z2 = a1.dot(theta2)
a2 = sigmoid(z2)
yhat = a2
predict = 1 * (yhat >= 0.5)

print('Theta1: ')
print(theta1)
print('Theta2: ')
print(theta2)
print('Predict: ')
print(predict)

plt.plot(x, y)
plt.title("Crossentroy with bias")
plt.show()