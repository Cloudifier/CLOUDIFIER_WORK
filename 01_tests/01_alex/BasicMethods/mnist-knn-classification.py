from sklearn import neighbors
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from datetime import datetime
from random import shuffle

mnist = fetch_mldata('mnist-original')
data = scale(mnist.data)
X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size=0.3)
X_train = X_train[0:10000]
X_test = X_test[0:1000]
y_train = y_train[0:10000]
y_test = y_test[0:1000]

weights = ['uniform', 'distance']
algs = ['auto', 'ball_tree', 'kd_tree', 'brute']

for weight in weights:
    for alg in algs:
        for n_neighbors in range(10, 15):
            start_ctr = datetime.now()
            clf = neighbors.KNeighborsClassifier(n_neighbors, algorithm=alg, weights=weight)
            end_ctr = datetime.now()

            start_fit = datetime.now()
            clf.fit(X_train, y_train)
            end_fit = datetime.now()

            start_train_pred = datetime.now()
            train_pred = clf.predict(X_train)
            end_train_pred = datetime.now()

            start_test_pred = datetime.now()
            test_pred = clf.predict(X_test)
            end_test_pred = datetime.now()

            print('Ctr time: {:}, fit time {:}, train time {:}, test time {:}'.format(str(end_ctr - start_ctr), str(end_fit - start_fit), str(end_train_pred - start_train_pred), str(end_test_pred - start_test_pred)))
            print('Weights: {:}, Algorithm: {:}, Neighbors: {:}, Train acc {:}, Test acc: {:}'.format(weight, alg, n_neighbors, accuracy_score(train_pred, y_train) * 100, accuracy_score(test_pred, y_test) * 100))
            print('\n')