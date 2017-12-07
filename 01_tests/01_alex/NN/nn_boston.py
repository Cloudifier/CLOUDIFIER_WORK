import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN.nn import *
from sklearn.datasets import load_boston

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)
    return norm, mean, max, min

def meanNormalizationWithParams(X, mean, max, min):
    norm = (X - mean) / (max - min)
    return norm

def rsquared(yhat, y):
    top = np.sum((yhat - y) ** 2)
    bottom = np.sum(y - np.sum(y) / len(y)) ** 2
    return 1 - top / bottom

boston = load_boston()
X = boston.data
y = boston.target.reshape(boston.target.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, mean, max, min = meanNormalization(X_train)
X_test = meanNormalizationWithParams(X_test, mean, max, min)
# y_train, mean, max, min = meanNormalization(y_train)
m_train, n_train = X_train.shape

layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=16, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=8, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=4, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.NONE))

nn = NN(layers, costFunction=CostFuncEnum.MSE, useBias=False)
nn.fit(X_train, y_train, epochs=1000, batchSize=8, alpha=0.03, lmbd=1)
learningStatus = nn.getLearningStatus()
yhat = nn.predict(X_test)
print(np.array((y_test, yhat)).T)
print(rsquared(yhat, y_test))
print(np.mean(yhat))

# plot cost
x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.error for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos)
plt.show()