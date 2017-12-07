import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def reluPrime(z):
    return (relu(z) > 0).astype(int)

def normalize(X):
    return (X - X.mean()) / (X.max() - X.min())

def accuracy(yhat, y):
    m, n = y.shape
    acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
    return round(acc * 100, 2)

def cleanData(df):
    m, n = df.shape
    #fill missing dates
    for i in range(m):
        name = str(df.ix[i, 'name']).upper()
        age = df.ix[i, 'age']
        if not age or math.isnan(age):
            if 'MISS' in name:
                df.ix[i, 'age'] = 20
            elif 'MASTER' in name:
                df.ix[i, 'age'] = 15
            elif 'MR' in name:
                df.ix[i, 'age'] = 40
            elif 'MRS' in name:
                df.ix[i, 'age'] = 35
            elif 'DR' in name:
                df.ix[i, 'age'] = 55
            else:
                df.ix[i, 'age'] = 35

    #remove string columns. for now
    df.drop('name', axis=1, inplace=True)
    df.drop('ticket', axis=1, inplace=True)
    df.drop('cabin', axis=1, inplace=True)
    df.drop('boat', axis=1, inplace=True)
    df.drop('body', axis=1, inplace=True)
    df.drop('home.dest', axis=1, inplace=True)

    #fill nan with 0
    df = df.fillna(0)

    #get_dummies
    df = pd.get_dummies(df)

    return df

def forwardProp(X, theta1, theta2, theta3):
    z1 = X.dot(theta1)
    a1 = relu(z1)

    z2 = a1.dot(theta2)
    a2 = relu(z2)

    z3 = a2.dot(theta3)
    a3 = sigmoid(z3)
    yhat = a3

    return z1, a1, z2, a2, z3, a3, yhat

def predict(X, theta1, theta2, theta3):
    z1, a1, z2, a2, z3, a3, yhat = forwardProp(X, theta1, theta2, theta3)
    return yhat

def titanic_nn(X_train, y_train, alpha, epochs, batch_size):
    values = list()
    m, n = X_train.shape
    theta1 = np.random.rand(n, 6)
    theta2 = np.random.rand(6, 3)
    theta3 = np.random.rand(3, 1)
    for i in range(epochs):
        if i % 10 == 0 and i > 0:
            acc = str(values[-1][2])
            print(f'#{i} epochs. Train accuracy: {acc}%')

        for j in range(int(m / batch_size)):
            start = j * batch_size
            end = m if ((j + 1) * batch_size  > m) else (j + 1) * batch_size

            X = X_train[start : end]
            y = y_train[start : end]

            #forward prop
            z1, a1, z2, a2, z3, a3, yhat = forwardProp(X, theta1, theta2, theta3)

            #backprop
            delta3 = yhat - y
            grad3 = a2.T.dot(delta3)
            delta2 = delta3.dot(theta3.T) * reluPrime(z2)
            grad2 = a1.T.dot(delta2)
            grad1 = X.T.dot(delta2.dot(theta2.T) * reluPrime(z1))

            theta1 = theta1 - alpha * grad1
            theta2 = theta2 - alpha * grad2
            theta3 = theta3 - alpha * grad3

        yhat = predict(X_train, theta1, theta2, theta3)
        E = np.sum((yhat - y_train) ** 2)
        acc = accuracy(yhat, y_train)
        values.append((i, E[0], acc, (theta1, theta2, theta3)))

    return values

#prepare data
df = pd.read_csv('titanic3.csv')
df = cleanData(df)

m, n = df.shape
X = df.drop('survived', axis = 1)
X = normalize(X)
X.insert(0, 'bias', 1)
y = df.ix[:, 'survived'].values.reshape(m, 1)

#set alg values
alpha = 0.05
epochs = 200
batch_size = 8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#train
values = titanic_nn(X_train, y_train, alpha, epochs, batch_size)

#predict
theta1, theta2, theta3 = sorted(values, key=lambda x: x[1])[0][3]
best_train_acc = sorted(values, key=lambda x: x[2], reverse=True)[0][2]
yhat = predict(X_test, theta1, theta2, theta3)
print('Best train accuracy: ' + str(best_train_acc))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))

#plot cost
x = [x[0] for x in values]
y = [x[1] for x in values]
plt.plot(x, y)
plt.show()
