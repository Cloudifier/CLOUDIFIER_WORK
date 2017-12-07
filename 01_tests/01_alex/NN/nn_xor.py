import matplotlib.pyplot as plt
from NN.nn import *

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# X_train = (X_train - np.mean(X_train)) / (np.max(X_train) - np.min(X_train))
y_train = np.array([[0], [1], [1], [0]])
m_train, n_train = X_train.shape

layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.SIGMOID))
layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.SIGMOID))
nn = NN(layers, costFunction=CostFuncEnum.CROSSENTROPY, useBias=True)
nn.fit(X_train, y_train, epochs=100000, batchSize=1, alpha=0.03, lmbd=1)
#primele 5 grafice sunt cu MSE
#apoi cu CROSSE

learningStatus = nn.getLearningStatus()
yhat = nn.predictWithBestTheta(X_train)
print((yhat >= 0.5) * 1)
print(nn.getBestEpochStatus().epochNr)

# plot cost
x_pos = [idx for idx, x in enumerate(learningStatus)]
y_pos = [x.error for idx, x in enumerate(learningStatus)]
plt.plot(x_pos, y_pos)
plt.show()


sort = sorted(learningStatus, key=lambda x: (x.accuracy), reverse=True)
for i in range(1, 20):
    print(f'Acc: {sort[i].accuracy}, error: {sort[i].error}')
