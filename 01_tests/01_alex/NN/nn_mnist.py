import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN.nn import *
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('mnist-original')
X = mnist.data
y = mnist.target

m, n = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

m_train, n_train = X_train.shape
layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=64, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=10, activationFunc=ActivationFuncEnum.SOFTMAX))
Y_train = np.zeros((len(y_train), 10))

for i in range(len(y_train)):
    Y_train[i, int(y_train[i])] = 1

nn = NN(layers, costFunction=CostFuncEnum.CROSSENTROPY_SOFTMAX, useBias=True)
nn.fit(X_train, Y_train, epochs=6, batchSize=16, alpha=0.03, lmbd=0)
learningStatus = nn.getLearningStatus()
yhat = nn.predict(X_test)
m, n = yhat.shape
predict = (np.argmax(yhat, axis=1))
print("Test acc: {:.2f}%".format(np.mean((predict == y_test) * 1) * 100))

yhat_train = nn.predict(X_train)
print("Train acc: {:.2f}%".format(100 * ((np.argmax(yhat_train, axis=1) == y_train.ravel()).sum() / y_train.shape[0])))

#plot cost
x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.error for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos, c='red')
# plt.show()

x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.accuracy for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos, c='blue')
# plt.show()

