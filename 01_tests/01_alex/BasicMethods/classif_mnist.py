import numpy as np
from BasicMethods.classification import Classification
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

mnist = fetch_mldata('mnist-original')
data = scale(mnist.data)
X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size=0.0005, random_state=1234)

classif = Classification(name='MNIST', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#classif.kmeans()
#classif.gaussian_naive_bayes()
#classif.knn()
#classif.decission_tree()
classif.logistic_regression()
