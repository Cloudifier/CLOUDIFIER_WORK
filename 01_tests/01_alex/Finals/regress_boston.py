import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from Finals.algo.regression import Regression
from Finals.algo.nn import ActivationFuncEnum, CostFuncEnum, Layer


boston = load_boston()
X = boston.data
y = boston.target.reshape(boston.target.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X), preprocessing.minmax_scale(y), test_size=0.3, random_state=1234)
m_train, n_train = X_train.shape

stats = list()
regress = Regression(name='BOSTON', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
stats.append(regress.decission_tree())
stats.append(regress.linear_regression())


layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=6, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.NONE))
stats.append(regress.NN(epochs=1000, layers=layers, costFunction=CostFuncEnum.MSE, useBias=False, early_stopping=True, tol=1e-5, theta_init=(-0.5, 0.5)))

df = pd.DataFrame.from_records([x.to_dict() for x in stats])
df.to_csv('_dataframes/boston.csv', index=False)