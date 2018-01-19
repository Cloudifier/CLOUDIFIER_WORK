import numpy as np
from enum import Enum
from typing import List
import sys
from sklearn import metrics

class ActivationFuncEnum(Enum):
    NONE = 0
    SIGMOID = 1
    RELU = 2
    SOFTMAX = 3

class CostFuncEnum(Enum):
    MSE = 0
    CROSSENTROPY = 1
    CROSSENTROPY_SOFTMAX = 2

class ActivationFunctions:
    # @staticmethod
    # def sigmoid(z):
        #return 1 / (1 + np.exp(-z))

    #http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    @staticmethod
    def relu(X):
        np.clip(X, 0, np.finfo(X.dtype).max, out=X)
        return X

    # @staticmethod
    # def relu(z):
    #     return np.maximum(0, z)

    # # #i always get overflow exception with this function.
    # @staticmethod
    # def softmax(z):
    #     m, n = z.shape
    #     sm = np.exp(z) / np.exp(z).sum(axis=1).reshape(m, 1)
    #     return sm


    #tooked this one from sklearn
    @staticmethod
    def softmax(X):
        tmp = X - X.max(axis=1)[:, np.newaxis]
        np.exp(tmp, out=X)
        X /= X.sum(axis=1)[:, np.newaxis]
        eps = 1e-6
        X = np.clip(X, eps, 1 - eps)
        return X

    @staticmethod
    def sigmoidPrime(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    @staticmethod
    def reluPrime(z):
        # return (ActivationFunctions.relu(z) > 0) * 1
        return (z > 0) * 1

class CostFunctions:
    @staticmethod
    def crossentropy(yhat, y):
        return np.mean(- y * np.log(yhat) - (1 - y) * np.log(1 - yhat))

    @staticmethod
    def mse(yhat, y):
        return np.mean((yhat - y) ** 2)

    @staticmethod
    def crossentropySoftmax(yhat, y):
        #res = - np.sum(y * np.log(yhat)) / y.shape[0]
        # eps = 1e-15

        #method inspired from sklearn log_loss
        # Clipping
        # yhat = np.clip(yhat, eps, 1 - eps)
        res = - np.sum(y * np.log(yhat)) / y.shape[0]

        return res

    @staticmethod
    def crossentropyPrime(yhat, y):
        return yhat - y

    @staticmethod
    def msePrime(yhat, y):
        return 2 * (yhat - y) / len(y)

class Theta:
    weights = None
    bias = None
    def __init__(self, nrRows: int, nrColumns: int):
        self.weights = np.random.uniform(-.001, .001, (nrRows, nrColumns))
        self.bias = np.random.uniform(-.001, .001, (1, nrColumns))

class LearningStatus:
    def __init__(self, epochNr, error, accuracy, thetas, yhat):
        self.epochNr = epochNr
        self.error = error
        self.accuracy = accuracy
        self.thetas = thetas
        self.yhat = yhat

class Layer:
    nrNeurons: int = 0
    activationFunc = None
    linearity = None
    activation = None
    delta = None
    grad = None
    theta: Theta = None

    def __init__(self, nrNeurons: int, activationFunc=ActivationFuncEnum.NONE):
        self.nrNeurons = nrNeurons
        self.activationFunc = activationFunc

class NN:
    epochs = None
    batchSize = None
    alpha = None
    lmbd = None

    def __init__(self, layers: List[Layer], costFunction, useBias, seed=1234):
        self.nnLayers = layers
        self.nnCostFunc = costFunction
        self.learningStatus: List[LearningStatus] = list()
        self.useBias = useBias
        np.random.seed(seed)
        self.__initTheta()

    def __initTheta(self):
        for i in range(1, len(self.nnLayers)): #skip first layer(input)
            currentLayer = self.nnLayers[i]
            previousLayer = self.nnLayers[i - 1]
            currentLayer.theta = Theta(previousLayer.nrNeurons, currentLayer.nrNeurons)

    def __accuracy(self, yhat, y):
        m, n = y.shape
        acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
        return acc * 100

    def __getThetas(self):
        thetas = list()
        for idx, layer in enumerate(self.nnLayers):
            if idx > 0:
                thetas.append(self.nnLayers[idx].theta)
        return thetas

    def __layerLinearity(self, currentLayer: Layer, previousActivation):
        m, n = previousActivation.shape
        currentLayer.linearity = previousActivation.dot(currentLayer.theta.weights)
        if self.useBias == True:
            layerBias = np.ones((m, 1)).dot(currentLayer.theta.bias)
            currentLayer.linearity += layerBias

    def __layerActivation(self, currentLayer: Layer):
        if currentLayer.activationFunc == ActivationFuncEnum.NONE:
            currentLayer.activation = currentLayer.linearity
        elif currentLayer.activationFunc == ActivationFuncEnum.RELU:
            currentLayer.activation = ActivationFunctions.relu(currentLayer.linearity)
        elif currentLayer.activationFunc == ActivationFuncEnum.SIGMOID:
            currentLayer.activation = ActivationFunctions.sigmoid(currentLayer.linearity)
        elif currentLayer.activationFunc == ActivationFuncEnum.SOFTMAX:
            currentLayer.activation = ActivationFunctions.softmax(currentLayer.linearity)

    def __layerPrimeActivation(self, currentLayer: Layer):
        if currentLayer.activationFunc == ActivationFuncEnum.NONE:
            return currentLayer.linearity
        elif currentLayer.activationFunc == ActivationFuncEnum.RELU:
            return ActivationFunctions.reluPrime(currentLayer.linearity)
        elif currentLayer.activationFunc == ActivationFuncEnum.SIGMOID:
            return ActivationFunctions.sigmoidPrime(currentLayer.linearity)

    def __layerCost(self, currentLayer: Layer, y):
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            currentLayer.delta = CostFunctions.crossentropyPrime(currentLayer.activation, y)
        elif self.nnCostFunc == CostFuncEnum.MSE:
            currentLayer.delta = CostFunctions.msePrime(currentLayer.activation,y)  # * ActivationFunctions.sigmoidPrime(currentLayer.activation)

    def __dJdzLastLayer(self, currentLayer, y):
        if(self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX and currentLayer.activationFunc == ActivationFuncEnum.SOFTMAX):
            # print(currentLayer.activation - y)
            currentLayer.delta = currentLayer.activation - y

    def __calculateDelta(self, layerPosition, y):
        currentLayer = self.nnLayers[layerPosition]
        if layerPosition == len(self.nnLayers) - 1:
            self.__dJdzLastLayer(currentLayer, y)
        else:
            previousLayer = self.nnLayers[layerPosition + 1]
            currentLayer.delta = previousLayer.delta.dot(previousLayer.theta.weights.T) * self.__layerPrimeActivation(currentLayer)

    def __calculateGrad(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        nextLayer = self.nnLayers[layerPosition - 1]
        currentLayer.grad = 1 / self.batchSize * nextLayer.activation.T.dot(currentLayer.delta) + self.lmbd / (2 * self.batchSize) * currentLayer.theta.weights

    def __calculateWeights(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        currentLayer.theta.weights -= self.alpha * currentLayer.grad

    def __calculateBias(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        m, n = currentLayer.delta.shape
        currentLayer.theta.bias -= self.alpha * 1 / self.batchSize * currentLayer.delta.sum(axis=0)

    def __forwardPropagation(self, X):
        previousActivation = X
        self.nnLayers[0].activation = X
        for i in range(1, len(self.nnLayers)):
            layer = self.nnLayers[i]
            self.__layerLinearity(layer, previousActivation)
            self.__layerActivation(layer)
            previousActivation = layer.activation[:]
        return previousActivation

    def __backPropagation(self, y):
        for pos, layer in reversed(list(enumerate(self.nnLayers))):
            if pos != 0:
                self.__calculateDelta(pos, y)
                self.__calculateGrad(pos)
                self.__calculateWeights(pos)
                if self.useBias == True:
                    self.__calculateBias(pos)

    def __setLearningStatus(self, X, y, epochNr):
        yhat = self.__forwardPropagation(X)
        # if 0 in yhat:
        #     print(yhat)

        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            E = CostFunctions.crossentropy(yhat, y)
        elif self.nnCostFunc == CostFuncEnum.MSE:
            E = CostFunctions.mse(yhat, y)
        elif self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX:
            E = CostFunctions.crossentropySoftmax(yhat, y)

        predict = (np.argmax(yhat, axis=1))
        y_true = np.argmax(y, axis=1)
        acc = np.mean((predict == y_true) * 1) * 100
        print("Train acc: {:.2f}%, Loss: {:.3f}".format(np.mean((predict == y_true) * 1) * 100, E))
        thetas = self.__getThetas()
        lastTheta = thetas[-1]
        print('-----THETA-----')
        print(lastTheta.weights[0:2, 0:4])
        print(lastTheta.bias)
        print('~~~~~THETA~~~~~')

        print('-----LINEARITY-----')
        lastLayer = self.nnLayers[-1]
        print(lastLayer.linearity[0,:])
        print('~~~~~LINEARITY~~~~~')

        print('-----PREV ACTIVATION-----')
        prevLayer = self.nnLayers[-2]
        print(prevLayer.activation[0, 0:5])
        print('~~~~~PREV ACTIVATION~~~~~')
        z = prevLayer.activation.dot(lastTheta.weights) + lastTheta.bias
        print(z[0])

        self.learningStatus.append(LearningStatus(epochNr, E, acc, self.__getThetas(), yhat))

    def __sortByAccuracy(self):
        return sorted(self.learningStatus, key=lambda x: (x.accuracy), reverse=True)

    def __restoreBestThetas(self):
        print('Sortam dupa best acc')
        sorted = self.__sortByAccuracy()
        print(f'Gasit best acc la epoca: {sorted[0].epochNr}, acc: {sorted[0].accuracy}, eroare: {sorted[0].error}')
        bestThetas = sorted[0].thetas
        for i in range(1, len(self.nnLayers)):
            self.nnLayers[i].theta = bestThetas[i - 1]

    def fit(self, X_train, y_train, epochs, batchSize, alpha, lmbd):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X_train and y_train should have same dimensions!')
        # if y_train.shape[1] != 1:
        #     raise ValueError('y_train should be a column vector!')

        self.epochs = epochs
        self.batchSize = batchSize
        self.alpha = alpha
        self.lmbd = lmbd
        m, n = X_train.shape
        for i in range(self.epochs):
            print('Epoch {:}'.format(i + 1))
            mod = m % self.batchSize
            iterations = int(m / self.batchSize) + (1 if mod != 0 else 0)
            for j in range(iterations):
                # print(f'Iteration {j}')
                start = j * self.batchSize
                end = (j * self.batchSize + mod) if mod != 0 and j == iterations - 1 else (j + 1) * self.batchSize
                self.__forwardPropagation(X_train[start: end])
                self.__backPropagation(y_train[start: end])
            self.__setLearningStatus(X_train, y_train, i)

    def predict(self, X_test):
        return self.__forwardPropagation(X_test)

    def predictWithBestTheta(self, X_test):
        self.__restoreBestThetas()
        return self.__forwardPropagation(X_test)

    def getLearningStatus(self):
        return self.learningStatus

    def getBestEpochStatus(self):
        return self.__sortByAccuracy()[0]
