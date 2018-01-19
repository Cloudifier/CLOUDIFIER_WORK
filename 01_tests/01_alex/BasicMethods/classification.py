import itertools
import logging
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import linear_model
from sklearn import svm, metrics

class Classification(object):
    def __init__(self, name, X_train, X_test, y_train, y_test):
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __setLog(self, name, method):
        logging.basicConfig(filename='{:}_{:}_{:}.txt'.format(name, method, datetime.now().strftime('%Y%m%d%H%M%S')), level=logging.DEBUG, format='%(message)s')

    def __printAndLog(self, str):
        print(str)
        logging.info(str)

    def kmeans(self):
        start = datetime.now()
        self.__setLog(self.name, 'kmeans')
        self.__printAndLog('Start kmeans')

        initializations = ['k-means++', 'random']
        max_iters = [1, 5, 10, 50, 100]
        n_inits = [1, 5, 10, 15]
        algorithms = ['elkan', 'full']

        use_cases = list(itertools.product(initializations, max_iters, n_inits, algorithms))
        best_train_acc, best_test_acc = 2 * (0,)
        best_init, best_max_iter, best_n_init, best_algo = 4 * (None,)

        for init, max_iter, n_init, algo in use_cases:
            self.__printAndLog('Init: {:}, max_iter: {:}, n_init: {:}, algo: {:}'.format(init, max_iter, n_init, algo))
            clf = KMeans(init=init, n_clusters=10, n_init=n_init, max_iter=max_iter)
            clf.fit(self.X_train, self.y_train)

            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)

            train_acc = accuracy_score(self.y_train, train_pred) * 100
            test_acc = accuracy_score(self.y_test, test_pred) * 100

            self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_init, best_max_iter, best_n_init, best_algo = init, max_iter, n_init, algo

            self.__printAndLog('\n')

        self.__printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, Best init: {:}, Best max iter: {:}, Best n init {:}, Best algo: {:}'.format(best_train_acc, best_test_acc, best_init, best_max_iter, best_n_init, best_algo))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End kmeans')

    def multinomial_naive_bayes(self):
        start = datetime.now()
        self.__setLog(self.name, 'naivebayes')
        self.__printAndLog('Start multinomial naive bayes')

        alphas = np.arange(0.0, 2.0, 0.1)
        fit_priors = [True, False]

        use_cases = list(itertools.product(alphas, fit_priors))
        best_train_acc, best_test_acc = 2 * (0,)
        best_alpha, best_fit = 2 * (None,)
        for alpha, fit_prior in use_cases:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(self.X_train, self.y_train)
            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)
            train_acc = accuracy_score(self.y_train, train_pred) * 100
            test_acc = accuracy_score(self.y_test, test_pred) * 100

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_alpha = alpha
                best_fit = fit_prior

            self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
            self.__printAndLog('\n')

        self.__printAndLog('Best train acc: {:.2f}%, best test acc: {:.2f}%, best alpha: {:}, best fit_prior: {:}'.format(best_train_acc, best_test_acc, best_alpha, best_fit))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End multinomial naive bayes')

    def gaussian_naive_bayes(self):
        start = datetime.now()
        self.__setLog(self.name, 'naivebayes')
        self.__printAndLog('Start gaussian naive bayes')

        clf = GaussianNB()
        clf.fit(self.X_train, self.y_train)
        train_pred = clf.predict(self.X_train)
        test_pred = clf.predict(self.X_test)
        train_acc = accuracy_score(self.y_train, train_pred) * 100
        test_acc = accuracy_score(self.y_test, test_pred) * 100

        self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
        end = datetime.now()

        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End gaussian naive bayes')

    def knn(self):
        start = datetime.now()
        self.__setLog(self.name, 'knn')
        self.__printAndLog('Start knn')

        weights = ['uniform', 'distance']
        algs = ['ball_tree', 'kd_tree', 'brute']
        n_neighbors = [1, 3, 5, 7, 9, 10, 15]
        use_cases = list(itertools.product(weights, algs, n_neighbors))
        best_train_acc, best_test_acc = 2 * (0,)
        best_weight, best_alg, best_n_neighbors = 3 * (None,)
        for weight , alg, n_neighbor in use_cases:
            start_ctr = datetime.now()
            clf = neighbors.KNeighborsClassifier(n_neighbor, algorithm=alg, weights=weight)
            end_ctr = datetime.now()

            start_fit = datetime.now()
            clf.fit(self.X_train, self.y_train)
            end_fit = datetime.now()

            start_train_pred = datetime.now()
            train_pred = clf.predict(self.X_train)
            end_train_pred = datetime.now()

            start_test_pred = datetime.now()
            test_pred = clf.predict(self.X_test)
            end_test_pred = datetime.now()

            train_acc = accuracy_score(self.y_train, train_pred) * 100
            test_acc = accuracy_score(self.y_test, test_pred) * 100

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_weight = weight
                best_alg = alg
                best_n_neighbors = n_neighbor
            print('Weights: {:}, Algorithm: {:}, Neighbors: {:}, Train acc {:.2f}%, Test acc: {:.2f}%'.format(weight, alg,
                                                                                                      n_neighbor,
                                                                                                      train_acc,
                                                                                                      test_acc))
            print('Ctr time: {:}, fit time {:}, train time {:}, test time {:}'.format(str(end_ctr - start_ctr),
                                                                                      str(end_fit - start_fit),
                                                                                      str(end_train_pred - start_train_pred),
                                                                                      str(end_test_pred - start_test_pred)))
            print('\n')

        self.__printAndLog('Best train acc: {:.2f}%, best test acc: {:.2f}%, best weight: {:}, best alg: {:}, best n neighbors: {:}'.format(best_train_acc, best_test_acc, best_weight, best_alg, best_n_neighbors))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End knn')

    def decission_tree(self):
        start = datetime.now()
        self.__setLog(self.name, 'decissiontree')
        self.__printAndLog('Start decission tree')

        depths = range(2, 20)
        leafs = range(2, 20)
        use_cases = list(itertools.product(depths, leafs))
        best_test_acc, best_train_acc, best_max_depth, best_max_leafs = 4 * (0,)
        for depth, leaf in use_cases:
            clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            clf = clf.fit(self.X_train, self.y_train)
            y_train = self.y_train.reshape(len(self.y_train), 1)
            y_test = self.y_test.reshape(len(self.y_test), 1)

            pred_train = clf.predict(self.X_train).reshape(self.X_train.shape[0], 1)
            pred_test = clf.predict(self.X_test).reshape(self.X_test.shape[0], 1)
            train_acc = np.sum(pred_train == y_train) / y_train.shape[0] * 100
            test_acc = np.sum(pred_test == y_test) / y_test.shape[0] * 100
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_max_depth = depth
                best_max_leafs = leaf

            self.__printAndLog('Max depth: {:}, Max leafs: {:}, Train acc: {:.2f}%, Test acc: {:.2f}%'.format(depth, leaf, train_acc, test_acc))
            self.__printAndLog('\n')

        self.__printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, max depth: {:}, max leaf: {:}'.format(best_train_acc, best_test_acc, best_max_depth, best_max_leafs))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End decission tree')

    def logistic_regression(self):
        start = datetime.now()
        self.__setLog(self.name, 'logisticregression')
        self.__printAndLog('Start logistic regression')

        regr = linear_model.LogisticRegression()
        regr.fit(self.X_train, self.y_train)

        test_pred = regr.predict(self.X_test)
        train_pred = regr.predict(self.X_train)

        self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(accuracy_score(self.y_train, train_pred) * 100, accuracy_score(self.y_test, test_pred) * 100))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End logistic regression')

    def svc(self):
        start = datetime.now()
        self.__setLog(self.name, 'svc')
        self.__printAndLog('Start svc')

        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        decision_functions = ['ovo', 'ovr']
        use_cases = list(itertools.product(kernels , decision_functions))
        best_train_acc, best_test_acc = 2 * (0,)
        best_kernel, best_decission = 2 * (None,)
        for kernel, decision in use_cases:
            print('Kernel: {:}, decision: {:}'.format(kernel, decision))
            classifier = svm.SVC(gamma=0.001, cache_size=500, kernel=kernel, decision_function_shape=decision)
            classifier.fit(self.X_train, self.y_train)
            train_pred = classifier.predict(self.X_train)
            test_pred = classifier.predict(self.X_test)

            train_acc = np.sum(train_pred == self.y_train) / self.y_train.shape[0] * 100
            test_acc = np.sum(test_pred == self.y_test) / self.y_test.shape[0] * 100
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_kernel = kernel
                best_decission = decision

            self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
            self.__printAndLog('\n')

        self.__printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, best kernel: {:}, best decission: {:}'.format(best_train_acc,
                                                                                                    best_test_acc,
                                                                                                    best_kernel,
                                                                                                    best_decission))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End svc')