import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from Finals.algo.classification import Classification
from Finals.algo.nn import Layer, ActivationFuncEnum, CostFuncEnum

# def nfold(folds=5, callback=None):
#     for use_case in range(1):
#         #m, n = X.shape
#         m = 1547
#         batch = m // folds
#         X, y = 2 * (None,)
#
#         use_case_accs = list()
#         for fold in range(folds):
#             start = fold * batch
#             end = start + batch if fold <= folds - 2 else start + (m - start)
#             X_validation = X[start: end]
#             y_validation = y[start: end]
#             X_train = np.delete(X, (start, end), axis=0)
#             y_train = np.delete(y, (start, end), axis=0)
#             nn.fit(X_train)
#
#             train_pred = nn.pred(y_train)
#             validation_pred = nn.pred(y_train)
#             use_case_accs.append((train_pred, validation_pred))
#
#         use_case_mean_acc = np.mean(use_case_accs)
#         print(f'Train mean acc {use_case_mean_acc[0]}%, Validation mean acc {use_case_mean_acc[1]}%')


stats = list()
mnist = fetch_mldata('mnist-original')
data = mnist.data / 255
X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size=0.0005, random_state=1234)
m_train, n_train = X_train.shape

classif = Classification(name='MNIST', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, folds=10)
# stats.append(classif.kmeans(n_clusters=10))
# stats.append(classif.gaussian_naive_bayes())
# # stats.append(classif.knn())
# stats.append(classif.decission_tree())
# stats.append(classif.logistic_regression())

layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=64, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=10, activationFunc=ActivationFuncEnum.SOFTMAX))

# stats.append(classif.NN(epochs=10, useBias=True, layers=layers, costFunction=CostFuncEnum.CROSSENTROPY_SOFTMAX,
#                         alphas=[0.02], lmbds=[0.1], batch_sizes=[8], theta_init=(-0.001, 0.001)))

stats.append(classif.NN(epochs=10, useBias=True, layers=layers, costFunction=CostFuncEnum.CROSSENTROPY_SOFTMAX, early_stopping=True, tol=1e-4))

df = pd.DataFrame.from_records([x.to_dict() for x in stats])
df.to_csv('_dataframes/mnistNN.csv', index=False)


