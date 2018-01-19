import itertools
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from scipy.stats import itemfreq
from Finals.algo.base import Base
from Finals.models.model_statistics import ModelStatistics

class Classification(Base):
    def _cluster_acc(self, y_true, y_pred):
        new_labels = list()
        unique_labels = np.unique(y_true)
        for label in unique_labels:
            cluster = list()
            for idx, item in enumerate(y_true):
                if label == item:
                    cluster.append(y_pred[idx])
            challengers = (itemfreq(cluster))
            max = 0
            winner = None
            for cluster_label, count in challengers:
                if count > max:
                    max = count
                    winner = cluster_label

            new_labels.append((label, winner))

        # you now know the cluster index
        # set this cluster index in y_true
        for idx, item in enumerate(y_true):
            for old, new in new_labels:
                if item == old:
                    y_true[idx] = new

        # now make predictions
        return round(np.sum(y_true == y_pred) * 1 / y_true.shape[0] * 100, 2)

    def kmeans(self, n_clusters):
        start = datetime.now()
        self._setLog(self.name, 'kmeans')
        self._printAndLog('Start kmeans')

        initializations = ['k-means++', 'random']
        max_iters = [1, 5, 10, 50, 100]
        n_inits = [1, 5, 10, 15]
        algorithms = ['elkan', 'full']

        accs = list()
        use_cases = list(itertools.product(initializations, max_iters, n_inits, algorithms))
        best_train_acc, best_test_acc = 2 * (0,)
        best_init, best_max_iter, best_n_init, best_algo = 4 * (None,)

        for init, max_iter, n_init, algo in use_cases:
            self._printAndLog('Init: {:}, max_iter: {:}, n_init: {:}, algo: {:}'.format(init, max_iter, n_init, algo))
            clf = KMeans(init=init, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, algorithm=algo)
            clf.fit(self.X_train, self.y_train)

            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)

            # train_acc = accuracy_score(self.y_train, train_pred) * 100
            # test_acc = accuracy_score(self.y_test, test_pred) * 100
            y_train = np.copy(self.y_train)
            y_test = np.copy(self.y_test)
            train_acc = round(self._cluster_acc(y_train, train_pred), 2)
            test_acc = round(self._cluster_acc(y_test, test_pred), 2)
            accs.append((test_acc, train_acc))

            self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_init, best_max_iter, best_n_init, best_algo = init, max_iter, n_init, algo

            self._printAndLog('\n')
        print(self.y_train)
        print(self.y_test)
        self._printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, Best init: {:}, Best max iter: {:}, Best n init {:}, Best algo: {:}'.format(best_train_acc, best_test_acc, best_init, best_max_iter, best_n_init, best_algo))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End kmeans')

        mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('kmeans', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'init= {:}; max_iter= {:}; n_init= {:}; algorithm= {:}'.format(best_init, best_max_iter, best_n_init, best_algo), '')

    def multinomial_naive_bayes(self):
        start = datetime.now()
        self._setLog(self.name, 'naivebayes')
        self._printAndLog('Start multinomial naive bayes')

        alphas = np.arange(0.0, 2.0, 0.1)
        fit_priors = [True, False]

        accs = list()
        use_cases = list(itertools.product(alphas, fit_priors))
        best_train_acc, best_test_acc = 2 * (0,)
        best_alpha, best_fit = 2 * (None,)
        for alpha, fit_prior in use_cases:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(self.X_train, self.y_train)
            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)

            train_acc = round(accuracy_score(self.y_train, train_pred) * 100, 2)
            test_acc = round(accuracy_score(self.y_test, test_pred) * 100, 2)
            accs.append((test_acc, train_acc))

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_alpha = alpha
                best_fit = fit_prior

            self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
            self._printAndLog('\n')

        self._printAndLog('Best train acc: {:.2f}%, best test acc: {:.2f}%, best alpha: {:}, best fit_prior: {:}'.format(best_train_acc, best_test_acc, best_alpha, best_fit))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End multinomial naive bayes')

        mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('multinomial naive bayes', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'alpha= {:}; fit_prior= {:}'.format(best_alpha, best_fit), '')

    def gaussian_naive_bayes(self):
        start = datetime.now()
        self._setLog(self.name, 'naivebayes')
        self._printAndLog('Start gaussian naive bayes')

        clf = GaussianNB()
        clf.fit(self.X_train, self.y_train)
        train_pred = clf.predict(self.X_train)
        test_pred = clf.predict(self.X_test)
        train_acc = round(accuracy_score(self.y_train, train_pred) * 100, 2)
        test_acc = round(accuracy_score(self.y_test, test_pred) * 100, 2)

        self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End gaussian naive bayes')

        return ModelStatistics('gaussian naive bayes', 1, test_acc, train_acc, test_acc, test_acc, '', '')

    def knn(self):
        start = datetime.now()
        self._setLog(self.name, 'knn')
        self._printAndLog('Start knn')

        weights = ['uniform', 'distance']
        algs = ['ball_tree', 'kd_tree', 'brute']
        n_neighbors = [1, 3, 5, 7, 9, 10, 15]

        accs = list()
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

            train_acc = round(accuracy_score(self.y_train, train_pred) * 100, 2)
            test_acc = round(accuracy_score(self.y_test, test_pred) * 100, 2)
            accs.append((test_acc, train_acc))

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

        self._printAndLog('Best train acc: {:.2f}%, best test acc: {:.2f}%, best weight: {:}, best alg: {:}, best n neighbors: {:}'.format(best_train_acc, best_test_acc, best_weight, best_alg, best_n_neighbors))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End knn')

        mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('knn', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'n_neighbor= {:}; algorithm= {:}; weights= {:}'.format(best_n_neighbors, best_alg, best_weight), '')

    def decission_tree(self):
        start = datetime.now()
        self._setLog(self.name, 'decissiontree')
        self._printAndLog('Start decission tree')

        depths = range(2, 20)
        leafs = range(2, 20)
        use_cases = list(itertools.product(depths, leafs))
        accs = list()

        best_test_acc, best_train_acc, best_max_depth, best_max_leafs = 4 * (0,)
        for depth, leaf in use_cases:
            clf = tree.DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)
            clf = clf.fit(self.X_train, self.y_train)
            y_train = self.y_train.reshape(len(self.y_train), 1)
            y_test = self.y_test.reshape(len(self.y_test), 1)

            pred_train = clf.predict(self.X_train).reshape(self.X_train.shape[0], 1)
            pred_test = clf.predict(self.X_test).reshape(self.X_test.shape[0], 1)

            train_acc = round(np.sum(pred_train == y_train) / y_train.shape[0] * 100, 2)
            test_acc = round(np.sum(pred_test == y_test) / y_test.shape[0] * 100, 2)
            accs.append((test_acc, train_acc))

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_max_depth = depth
                best_max_leafs = leaf

            self._printAndLog('Max depth: {:}, Max leafs: {:}, Train acc: {:.2f}%, Test acc: {:.2f}%'.format(depth, leaf, train_acc, test_acc))
            self._printAndLog('\n')

        self._printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, max depth: {:}, max leaf: {:}'.format(best_train_acc, best_test_acc, best_max_depth, best_max_leafs))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End decission tree')

        mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('decission tree', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'max_depth= {:}; min_samples_leaf= {:}'.format(best_max_depth, best_max_leafs), '')

    def logistic_regression(self):
        start = datetime.now()
        self._setLog(self.name, 'logisticregression')
        self._printAndLog('Start logistic regression')

        regr = linear_model.LogisticRegression()
        regr.fit(self.X_train, self.y_train)

        test_pred = regr.predict(self.X_test)
        train_pred = regr.predict(self.X_train)
        train_acc = round(accuracy_score(self.y_train, train_pred) * 100, 2)
        test_acc = round(accuracy_score(self.y_test, test_pred) * 100, 2)

        self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End logistic regression')

        return ModelStatistics('logistic regression', 1, test_acc, train_acc, test_acc, train_acc, '', '')

    def svc(self):
        start = datetime.now()
        self._setLog(self.name, 'svc')
        self._printAndLog('Start svc')

        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        decision_functions = ['ovo', 'ovr']
        use_cases = list(itertools.product(kernels , decision_functions))
        accs = list()

        best_train_acc, best_test_acc = 2 * (0,)
        best_kernel, best_decission = 2 * (None,)
        for kernel, decision in use_cases:
            print('Kernel: {:}, decision: {:}'.format(kernel, decision))
            classifier = svm.SVC(gamma=0.001, cache_size=500, kernel=kernel, decision_function_shape=decision)
            classifier.fit(self.X_train, self.y_train)
            train_pred = classifier.predict(self.X_train)
            test_pred = classifier.predict(self.X_test)

            train_acc = round(np.sum(train_pred == self.y_train) / self.y_train.shape[0] * 100, 2)
            test_acc = round(np.sum(test_pred == self.y_test) / self.y_test.shape[0] * 100, 2)
            accs.append((test_acc, train_acc))

            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_kernel = kernel
                best_decission = decision

            self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc))
            self._printAndLog('\n')

        self._printAndLog('Best train acc: {:.2f}%, Best test acc: {:.2f}%, best kernel: {:}, best decission: {:}'.format(best_train_acc,
                                                                                                    best_test_acc,
                                                                                                    best_kernel,
                                                                                                    best_decission))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End svc')

        mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('svc', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'kernel= {:}; decision_function_shape= {:}'.format(best_kernel, best_decission), '')