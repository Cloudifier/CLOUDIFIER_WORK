from BasicMethods.regression import Regression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target.reshape(boston.target.shape[0], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

regress = Regression(name='BOSTON', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
regress.decission_tree()
print('\n')
regress.linear_regression()
