import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NN.nn import *
from sklearn.datasets import load_digits

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)
    return norm, mean, max, min

def meanNormalizationWithParams(X, mean, max, min):
    norm = (X - mean) / (max - min)
    return norm

digits = load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target.reshape(n_samples, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train, mean, max, min = meanNormalization(X_train)
X_test = meanNormalizationWithParams(X_test, mean, max, min)
m_train, n_train = X_train.shape

#
layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=32, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=16, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=len(digits.target_names), activationFunc=ActivationFuncEnum.SOFTMAX))
Y_train = np.zeros((len(y_train), len(digits.target_names)))

for i in range(len(y_train)):
    Y_train[i, y_train[i]] = 1

nn = NN(layers, costFunction=CostFuncEnum.CROSSENTROPY_SOFTMAX, useBias=True)
nn.fit(X_train, Y_train, epochs=5, batchSize=8, alpha=0.005, lmbd=0)
learningStatus = nn.getLearningStatus()
yhat = nn.predict(X_test)
m, n = yhat.shape
predict = (np.argmax(yhat, axis=1)).reshape(m, 1)
print(np.mean((predict == y_test) * 1) * 100)


yhat_train = nn.predict(X_train)
print("Train acc: {:.2f}%".format(
    100 * ((np.argmax(yhat_train, axis=1) == y_train.ravel()).sum() / y_train.shape[0])))

#plot cost
x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.error for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos)
plt.show()

#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
# # Now apply the transformations to the data:
# X_train = scaler.transform(X_train)
# # X_test = scaler.transform(X_test)
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 10), activation='relu', batch_size=8, alpha=0.01, max_iter=10, solver='sgd')
# mlp.fit(X_train,y_train.ravel())
# predictions = mlp.predict(X_test)
# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))