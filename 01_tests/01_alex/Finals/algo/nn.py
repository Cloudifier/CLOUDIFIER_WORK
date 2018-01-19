import numpy as np
import warnings
from enum import Enum
from typing import List
from sklearn.metrics import r2_score
from scipy.special import expit
#warnings.filterwarnings("error")
np.set_printoptions(threshold=np.inf)

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
    @staticmethod
    def sigmoid(z):
        return expit(z)

    @staticmethod
    def relu(z):
        a = np.array(z)
        np.maximum(a, 0, a)
        return a

    # @staticmethod
    # def softmax(z):
    #     z -= np.max(z)
    #     ez = np.exp(z)
    #     sm = (ez.T / np.sum(ez, axis=1)).T
    #
    #     return sm

    @staticmethod
    def softmax(X):
        # print(X[0][0])
        X= np.asarray(X)
        X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        if np.sum(np.isnan(X)) > 0:
            raise Exception('INF/NAN')

        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X
    # @staticmethod
    # def sigmoid(z):
    #     z = np.array(z, dtype=np.float32)
    #     return 1 / (1 + np.exp(-z))
    #
    # #http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # # @staticmethod
    # # def sigmoid(x):
    # #     if x >= 0:
    # #         z = np.exp(-x)
    # #         return 1 / (1 + z)
    # #     else:
    # #         z = np.exp(x)
    # #         return z / (1 + z)
    #
    # @staticmethod
    # # def relu(X):
    # #     return np.clip(X, 0, np.finfo(X.dtype).max)
    #
    # # @staticmethod
    # def relu(z):
    #     return np.maximum(0, z)
    #
    # # # #i always get overflow exception with this function.
    # # @staticmethod
    # # def softmax(z):
    # #     m, n = z.shape
    # #     sm = np.exp(z) / np.exp(z).sum(axis=1).reshape(m, 1)
    # #     return sm
    #
    #
    # # tooked this one from sklearn
    # @staticmethod
    # def softmax(X):
    #     # # eps = 1e-6
    #     # print(X.max(axis=1))
    #     # tmp = X - X.max(axis=1)[:, np.newaxis]
    #     # X = np.exp(tmp)
    #     # X /= X.sum(axis=1)[:, np.newaxis]
    #     # # X = np.clip(X, eps, 1 - eps)
    #     # return X
    #     X = np.copy(X)
    #     max_prob = np.max(X, axis=1).reshape((-1, 1))
    #     X -= max_prob
    #     np.exp(X, X)
    #     sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    #     X /= sum_prob
    #     return X

    @staticmethod
    def reluPrime(z):
        a = (z > 0).astype(int)
        return a

    @staticmethod
    def sigmoidPrime(z):
        return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))

    # @staticmethod
    # def sigmoidPrime(z):
    #     return ActivationFunctions.sigmoid(z) * (1 - ActivationFunctions.sigmoid(z))
    #
    # @staticmethod
    # def reluPrime(z):
    #     return (ActivationFunctions.relu(z) > 0) * 1
    #     #return (z > 0) * 1

class CostFunctions:
    @staticmethod
    def crossentropy(yhat, y):
        return np.sum(- y * np.log(yhat) - (1 - y) * np.log(1 - yhat)) / y.shape[0]

    @staticmethod
    def mse(yhat, y):
        return np.mean((yhat - y) ** 2)

    @staticmethod
    def crossentropySoftmax(yhat, y):
        #eps = 1e-6
        #tmp = np.clip(eps, 1-eps, yhat)
        #return - np.sum(y * np.log(tmp)) / y.shape[0]
        return - np.sum(y * np.log(yhat)) / y.shape[0]

    @staticmethod
    def crossentropyPrime(yhat, y):
        return yhat - y

    @staticmethod
    def msePrime(yhat, y):
        return 2 * (yhat - y) / len(y)

class Theta:
    weights = None
    bias = None
    def __init__(self, nrRows: int, nrColumns: int, theta_init=None):
        ##mnist init
        # self.weights = np.random.uniform(-.001, .001, (nrRows, nrColumns))
        # self.bias = np.random.uniform(-.001, .001, (1, nrColumns))
        ##titanic init
        # self.weights = np.random.uniform(-.5, .5, (nrRows, nrColumns))
        # self.bias = np.random.uniform(-.5, .5, (1, nrColumns))
        ## boston init
        # self.weights = np.random.uniform(-.5, .5, (nrRows, nrColumns))
        # self.bias = np.random.uniform(-.5, .5, (1, nrColumns))

        # if theta_init is None:
        #     self.weights = np.random.uniform(-.001, .001, (nrRows, nrColumns))
        #     self.bias = np.random.uniform(-.001, .001, (1, nrColumns))
        # else:
        #     #print(theta_init[0], theta_init[1])
        #     self.weights = np.random.uniform(theta_init[0], theta_init[1], (nrRows, nrColumns))
        #     self.bias = np.random.uniform(theta_init[0], theta_init[1], (1, nrColumns))

        np.random.seed(1234)
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

    def __init__(self, layers, costFunction, useBias, seed=1234, early_stopping=False, tol=None, theta_init=None):
        self.nnLayers = layers
        self.nnCostFunc = costFunction
        self.learningStatus: List[LearningStatus] = list()
        self.useBias = useBias
        self.early_stopping = early_stopping
        self.tol = tol

        np.random.seed(seed)
        self._initTheta(theta_init)

    def _initTheta(self, theta_init):
        for i in range(1, len(self.nnLayers)): #skip first layer(input)
            currentLayer = self.nnLayers[i]
            previousLayer = self.nnLayers[i - 1]
            currentLayer.theta = Theta(previousLayer.nrNeurons, currentLayer.nrNeurons, theta_init)

    def _accuracy(self, yhat, y):
        m, n = y.shape
        acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
        return acc * 100

    def _getThetas(self):
        thetas = list()
        for idx, layer in enumerate(self.nnLayers):
            if idx > 0:
                thetas.append(self.nnLayers[idx].theta)
        return thetas

    def _layerLinearity(self, currentLayer: Layer, previousActivation):
        m, n = previousActivation.shape
        currentLayer.linearity = previousActivation.dot(currentLayer.theta.weights)
        if self.useBias == True:
            layerBias = np.ones((m, 1)).dot(currentLayer.theta.bias)
            currentLayer.linearity += layerBias

    def _layerActivation(self, currentLayer: Layer):
        lin = currentLayer.linearity[:]
        if currentLayer.activationFunc == ActivationFuncEnum.NONE:
            currentLayer.activation = lin
        elif currentLayer.activationFunc == ActivationFuncEnum.RELU:
            currentLayer.activation = ActivationFunctions.relu(lin)
        elif currentLayer.activationFunc == ActivationFuncEnum.SIGMOID:
            currentLayer.activation = ActivationFunctions.sigmoid(lin)
        elif currentLayer.activationFunc == ActivationFuncEnum.SOFTMAX:
            currentLayer.activation = ActivationFunctions.softmax(lin)

    def _layerPrimeActivation(self, currentLayer: Layer):
        if currentLayer.activationFunc == ActivationFuncEnum.NONE:
            return currentLayer.linearity[:]
        elif currentLayer.activationFunc == ActivationFuncEnum.RELU:
            return ActivationFunctions.reluPrime(currentLayer.linearity[:])
        elif currentLayer.activationFunc == ActivationFuncEnum.SIGMOID:
            return ActivationFunctions.sigmoidPrime(currentLayer.linearity[:])

    def _layerCost(self, currentLayer, y):
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX:
            currentLayer.delta = CostFunctions.crossentropyPrime(currentLayer.activation, y)
        elif self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            currentLayer.delta = CostFunctions.crossentropyPrime(currentLayer.activation, y) # ???? not enough
        elif self.nnCostFunc == CostFuncEnum.MSE:
            currentLayer.delta = CostFunctions.msePrime(currentLayer.activation, y)  # * ActivationFunctions.sigmoidPrime(currentLayer.activation)

    def _calculateDelta(self, layerPosition, y):
        currentLayer = self.nnLayers[layerPosition]
        if layerPosition == len(self.nnLayers) - 1:
            self._layerCost(currentLayer, y)
        else:
            nextLayer = self.nnLayers[layerPosition + 1]
            currentLayer.delta = nextLayer.delta.dot(nextLayer.theta.weights.T) * self._layerPrimeActivation(currentLayer)

    def _calculateGrad(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        previousLayer = self.nnLayers[layerPosition - 1]
        currentLayer.grad = 1 / self.batchSize * previousLayer.activation.T.dot(currentLayer.delta) + self.lmbd / (2 * self.batchSize) * currentLayer.theta.weights
        # print('---------------')
        # print(f'Grad sign: {"+" if currentLayer.grad[0][0] > 0 else "-" }, Grad: {currentLayer.grad[0][0]}, Inmultire: {np.sum(previousLayer.activation.T.dot(currentLayer.delta))}, Weight: {currentLayer.theta.weights[0][0]}, Layer Pos: {layerPosition}')
        # print('---------------')

    def _calculateWeights(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        currentLayer.theta.weights -= self.alpha * currentLayer.grad

    def _calculateBias(self, layerPosition):
        currentLayer = self.nnLayers[layerPosition]
        currentLayer.theta.bias -= self.alpha / self.batchSize * currentLayer.delta.sum(axis=0) + self.lmbd / (2 * self.batchSize) * currentLayer.delta.sum(axis=0)

    def _calculateNNCost(self, yhat, y):
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            E = CostFunctions.crossentropy(yhat, y)
        elif self.nnCostFunc == CostFuncEnum.MSE:
            E = CostFunctions.mse(yhat, y)
        elif self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX:
            E = CostFunctions.crossentropySoftmax(yhat, y)
        return E

    def _forwardPropagation(self, X):
        previousActivation = X
        self.nnLayers[0].activation = X
        for i in range(1, len(self.nnLayers)):
            layer = self.nnLayers[i]
            self._layerLinearity(layer, previousActivation)
            self._layerActivation(layer)
            previousActivation = layer.activation[:]
        return previousActivation

    def _backPropagation(self, y):
        for pos, layer in reversed(list(enumerate(self.nnLayers))):
            if pos != 0:
                self._calculateDelta(pos, y)
                self._calculateGrad(pos)
                self._calculateWeights(pos)
                if self.useBias == True:
                    self._calculateBias(pos)

    def _setLearningStatus(self, X, y, y_train, epochNr):
        yhat, acc = self.predict_with_accuracy(X, y)
        E = self._calculateNNCost(yhat, y_train)
        print("Train acc: {:.2f}%, Loss: {:.3f}".format(acc, E))
        self.learningStatus.append(LearningStatus(epochNr, E, acc, self._getThetas(), yhat))

    def _sortByAccuracy(self):
        return sorted(self.learningStatus, key=lambda x: (x.accuracy), reverse=True)

    def _restoreBestThetas(self):
        print('Sortam dupa best acc')
        sorted = self._sortByAccuracy()
        print(f'Gasit best acc la epoca: {sorted[0].epochNr}, acc: {sorted[0].accuracy}, eroare: {sorted[0].error}')
        bestThetas = sorted[0].thetas
        for i in range(1, len(self.nnLayers)):
            self.nnLayers[i].theta = bestThetas[i - 1]

    def _setY(self, y_true):
        y = None
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX:
            y = np.zeros((len(y_true), 10))
            for i in range(len(y_true)):
                y[i][int(y_true[i])] = 1
        else:
            y = y.reshape(y_true, 1)
        return y

    def fit(self, X_train, y_train, epochs, batchSize, alpha, lmbd):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('X_train and y_train should have same dimensions!')

        self.epochs = epochs
        self.batchSize = batchSize
        self.alpha = alpha
        self.lmbd = lmbd

        y_true = y_train[:]
        y_train = self._setY(y_train)
        m, n = X_train.shape
        for i in range(self.epochs):
            print('Epoch {:}'.format(i + 1))
            mod = m % self.batchSize
            iterations = int(m / self.batchSize) + (1 if mod != 0 else 0)
            for j in range(iterations):
                start = j * self.batchSize
                end = (j * self.batchSize + mod) if mod != 0 and j == iterations - 1 else (j + 1) * self.batchSize
                # print(f'Iteration: {j}, start: {start}, end: {end}')
                self._forwardPropagation(X_train[start: end])
                self._backPropagation(y_train[start: end])
            self._setLearningStatus(X_train, y_true, y_train, i)

            if self.early_stopping == True and len(self.learningStatus) >= 2:
                last_stat = self.learningStatus[-1]
                penultimate_stat = self.learningStatus[-2]
                if np.absolute(last_stat.error - penultimate_stat.error) <= self.tol:
                    print('Error is not improving, decided to stop epochs. Tol: {}'.format(self.tol))
                    break


    def predict(self, X_test):
        return self._forwardPropagation(X_test)

    def predictWithBestTheta(self, X_test):
        self._restoreBestThetas()
        return self._forwardPropagation(X_test)

    def predict_with_accuracy(self, X, y):
        acc = 0
        m, n = X.shape

        y_pred = self._forwardPropagation(X)
        if self.nnCostFunc == CostFuncEnum.CROSSENTROPY_SOFTMAX:
            pred = np.argmax(y_pred, axis=1).reshape(y_pred.shape[0], 1)
            acc = round(np.sum(pred == y) / m * 100, 2)
        elif self.nnCostFunc == CostFuncEnum.CROSSENTROPY:
            y_pred = ((y_pred > 0.5) * 1).reshape(y_pred.shape[0], 1)
            acc = round(np.sum(y_pred == y) / m * 100, 2)
        elif self.nnCostFunc == CostFuncEnum.MSE:
            acc = r2_score(y, y_pred)
        return (y_pred, acc)

    def getLearningStatus(self):
        return self.learningStatus

    def getBestEpochStatus(self):
        return self._sortByAccuracy()[0]
