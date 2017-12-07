import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from enum import Enum
from typing import List


import warnings
#warnings.filterwarnings('error')

class ActivationFuncEnum(Enum):
    NONE = 0
    SIGMOID = 1
    RELU = 2

class CostFuncEnum(Enum): 
    MSE = 0
    CROSSENTROPY = 1

class ActivationFunctions:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoidPrime(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    @staticmethod
    def reluPrime(z):
        return (ActivationFunctions.relu(z) > 0) * 1

class CostFunctions:
    @staticmethod
    def crossentropy(yhat, y):
        return - y * np.log(yhat) - (1 - y) * np.log(1 - yhat)

    @staticmethod
    def mse(yhat, y):
        return np.sum((yhat - y) ** 2)

    @staticmethod
    def crossentropyPrime(yhat, y):
        return yhat - y

    @staticmethod
    def msePrime(yhat, y):
        return 2 * (yhat - y)

class Theta:
    values = None
    def __init__(self, nrRows: int, nrColumns: int):
        self.nrRows = nrRows
        self.nrColumns = nrColumns
        self.values = np.random.rand(nrRows, nrColumns)

class LearningStatus:
    def __init__(self, error, accuracy, thetas):
        self.error = error
        self.accuracy = accuracy
        self.thetas = thetas

class Layer:
    nrNeurons:int = 0
    activationFunc = None
    linearity = None
    activation = None
    delta = None
    theta:Theta = None

    def __init__(self, nrNeurons: int, activationFunc=ActivationFuncEnum.NONE):
        self.nrNeurons = nrNeurons
        self.activationFunc = activationFunc

#intrare->hidden layers->output
class NN:
    def __init__(self, layers: List[Layer], epochs, batchSize, alpha, costFunction):
        self.nnLayers = layers
        self.epochs = epochs
        self.batchSize = batchSize
        self.alpha = alpha
        self.nnCostFunc = costFunction
        self.learningStatus = list()
        self.__setTheta()

    def __setTheta(self):
        for i in range(1, len(self.nnLayers)): #skip first layer(input)
            currentLayer = self.nnLayers[i]
            previousLayer = self.nnLayers[i - 1]
            thetaLayer = Theta(previousLayer.nrNeurons, currentLayer.nrNeurons)
            currentLayer.theta = thetaLayer

    def __accuracy(self, yhat, y):
        m, n = y.shape
        acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
        return round(acc * 100, 2)

    def __getThetas(self):
        thetas = list()
        for idx, layer in enumerate(self.nnLayers):
            if idx > 0:
                thetas.append(self.nnLayers[idx].theta.values)

    def __calculateDelta(self, layerPosition, y):
        currentLayer = self.nnLayers[layerPosition]
        if layerPosition == len(self.nnLayers) - 1:
            if self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
                currentLayer.delta = CostFunctions.crossentropyPrime(currentLayer.activation, y)
            elif self.nnCostFunc == CostFuncEnum.MSE:
                currentLayer.delta = CostFunctions.msePrime(currentLayer.activation, y) * ActivationFunctions.sigmoidPrime(currentLayer.activation)
        else:
            previousLayer = self.nnLayers[layerPosition + 1]
            if currentLayer.activationFunc == ActivationFuncEnum.RELU:
                currentLayer.delta = previousLayer.delta.dot(previousLayer.theta.values.T) * ActivationFunctions.reluPrime(currentLayer.linearity)
            elif currentLayer.activationFunc == ActivationFuncEnum.SIGMOID:
                currentLayer.delta = previousLayer.delta.dot(previousLayer.theta.values.T) * ActivationFunctions.sigmoidPrime(currentLayer.activation)

    def __calculateGrad(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        nextLayer = self.nnLayers[layerPosition - 1]

        currentLayer.theta.values -= (self.alpha / self.batchSize) * nextLayer.activation.T.dot(currentLayer.delta)

    def __forwardPropagation(self, X):
        previousActivation = X
        self.nnLayers[0].activation = X
        for i in range(1, len(self.nnLayers)):
            layer = self.nnLayers[i]
            layer.linearity = previousActivation.dot(layer.theta.values)
            if layer.activationFunc == ActivationFuncEnum.NONE:
                layer.activation = layer.linearity[:]
            elif layer.activationFunc == ActivationFuncEnum.SIGMOID:
                layer.activation = ActivationFunctions.sigmoid(layer.linearity)
            elif layer.activationFunc == ActivationFuncEnum.RELU:
                layer.activation = ActivationFunctions.relu(layer.linearity)
            previousActivation = layer.activation[:]

        return previousActivation

    def __backPropagation(self, y):
        for pos, layer in reversed(list(enumerate(self.nnLayers))):
            if pos != 0:
                self.__calculateDelta(pos, y)
        for pos, layer in reversed(list(enumerate(self.nnLayers))):
            if pos != 0:
                self.__calculateGrad(pos)

    def __setLearningStatus(self, X, y):
        yhat = self.__forwardPropagation(X)
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            E = np.mean(CostFunctions.crossentropy(yhat, y))
        elif self.nnCostFunc == CostFuncEnum.MSE:
            E = np.mean(CostFunctions.mse(yhat, y))
        self.learningStatus.append(LearningStatus(E, self.__accuracy(yhat, y), self.__getThetas()))

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X_train and y_train should have same dimensions!')
        if y_train.shape[1] != 1:
            raise ValueError('y_train should be a column vector!')

        m, n = X_train.shape
        for i in range(self.epochs):
            print(f'Epoch {i}')
            if self.batchSize == m:
                self.__forwardPropagation(X_train)
                self.__backPropagation(y_train)
            else:
                mod = m % self.batchSize
                iterations = int(m / self.batchSize) + (1 if mod != 0 else 0)
                for j in range(iterations):
                    start = j * self.batchSize
                    end = (j * self.batchSize + mod) if mod != 0 and j == iterations - 1 else (j + 1) * self.batchSize
                    self.__forwardPropagation(X_train[start: end])
                    self.__backPropagation(y_train[start: end])

            self.__setLearningStatus(X_train, y_train)

    def predict(self, X_test):
        return self.__forwardPropagation(X_test)

    def getLearningStatus(self):
        return self.learningStatus

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)
    return norm, mean, max, min

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def cleanData(df):
    m, n = df.shape
    # parse name, extract title
    df["Title"] = ""
    for i in range(m):
        name = str(df.loc[i, 'Name']).upper()
        title = find_between(name, ',', '.')
        df.loc[i, "Title"] = title if title != '' else 'MR'

    #remove string columns. for now
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    #fill some nan on specific columns
    df['Embarked'] = df['Embarked'].fillna('S')

    df = pd.get_dummies(df)
    df = df.fillna(0)
    return df

#train for age
df = pd.read_csv('train.csv')
df = cleanData(df)

dfTrain = df[df['Age'] == df['Age']]
dfTrain,mean,max,min = meanNormalization(dfTrain)
X_train = dfTrain.drop('Age', axis=1)
X_train.insert(0, 'Bias', 1)
m, n = dfTrain.shape
y_train = dfTrain.loc[:, 'Age'].values.reshape(m, 1)

dfTest = df[df['Age'] == df['Age']]
dfTest,mean_test,max_test,min_test = meanNormalization(dfTest)
X_test = dfTest.drop('Age', axis=1)
X_test.insert(0, 'Bias', 1)
m, _ = X_test.shape
y_test = dfTest.loc[:, 'Age'].values.reshape(m, 1)

layers = []
layers.append(Layer(nrNeurons=n))
layers.append(Layer(nrNeurons=6, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.NONE))

nn = NN(layers, epochs=200, batchSize=8, alpha=2.7, costFunction=CostFuncEnum.MSE)
nn.fit(X_train, y_train)
learningStatus = nn.getLearningStatus()
yhat = nn.predict(X_test)
print(yhat * (max_test['Age'] - min_test['Age']) + mean_test['Age'])

x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.error for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos)
plt.show()

# X = df.drop('Age', axis=1)
# X = meanNormalization(X)
# X.insert(0, 'Bias', 1)
# y = df.loc[:, 'Age'].values.reshape(m, 1)
#
# print(X[0:5])
#
# m, n = df.shape
# X = df.drop('Survived', axis=1)
# X = meanNormalization(X)
# X.insert(0, 'Bias', 1)
# y = df.loc[:, 'Survived'].values.reshape(m, 1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# # dfTest = pd.read_csv('test.csv')
# # y = cleanData(dfTrain)
#
#
# m, n = X_train.shape
# layers = []
# layers.append(Layer(nrNeurons=n))
# layers.append(Layer(nrNeurons=6, activationFunc=ActivationFuncEnum.RELU))
# layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.RELU))
# layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.SIGMOID))
#
# nn = NN(layers, epochs=10, batchSize=8, alpha=0.1, costFunction=CostFuncEnum.CROSSENTROPY)
# nn.fit(X_train, y_train)
# learningStatus = nn.getLearningStatus()
#
# #plot cost
# x_pos = [idx for idx, x in enumerate(learningStatus)]
# y_pos = [x.accuracy / 100 for idx, x in enumerate(learningStatus)]
# plt.plot(x_pos, y_pos)
# #plot accuracy
# y_pos = [x.error[0] for idx, x in enumerate(learningStatus)]
# plt.plot(x_pos, y_pos)
# plt.show()
#
# bestAcc = sorted(learningStatus, key=lambda x:x.accuracy, reverse=True)[0]
# print(bestAcc.accuracy)
#
