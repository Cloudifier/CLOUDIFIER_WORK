import itertools
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(filename='kmeans_{:}.txt'.format(datetime.now().strftime('%Y%m%d%H%M%S')),level=logging.DEBUG, format='%(message)s')

mnist = fetch_mldata('mnist-original')
#data = scale(mnist.data)
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.1, random_state=1234)
initializations = ['k-means++', 'random']
max_iters = [1, 5, 10, 50, 100]
n_inits = [1, 5, 10, 15]
algorithms = ['elkan', 'full']

use_cases = list(itertools.product(initializations, max_iters, n_inits, algorithms))
best_train_acc, best_test_acc = 2 * (0,)
best_init, best_max_iter, best_n_init, best_algo = 4 * (None,)

for init, max_iter, n_init, algo in use_cases:
    print('Init: {:}, max_iter: {:}, n_init: {:}, algo: {:}'.format(init, max_iter, n_init, algo))
    logging.info('Init: {:}, max_iter: {:}, n_init: {:}, algo: {:}'.format(init, max_iter, n_init, algo))
    clf = KMeans(init=init, n_clusters=10, n_init=n_init, max_iter=max_iter)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    print('Train acc: {:}, Test acc: {:}'.format(train_acc, test_acc))
    logging.info('Train acc: {:}, Test acc: {:}'.format(train_acc, test_acc))

    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_train_acc = train_acc
        best_init, best_max_iter, best_n_init, best_algo = init, max_iter, n_init, algo

    print('\n\n')
    logging.info('\n\n')
print('Best init: {:}, Best max iter: {:}, best n init {:}, best algo: {:}'.format(best_init, best_max_iter, best_n_init, best_algo))
logging.info('Best init: {:}, Best max iter: {:}, best n init {:}, best algo: {:}'.format(best_init, best_max_iter, best_n_init, best_algo))

