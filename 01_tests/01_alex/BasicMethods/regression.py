import itertools
import logging
from datetime import datetime
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model


class Regression(object):
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

    def decission_tree(self):
        start = datetime.now()
        self.__setLog(self.name, 'decissiontree')
        self.__printAndLog('Start decission tree')

        depths = range(2, 20)
        leafs = range(2, 20)
        use_cases = list(itertools.product(depths, leafs))
        best_test_acc = 0
        best_train_acc = 0
        best_max_depth = 0
        best_max_leafs = 0
        for depth, leaf in use_cases:
            clf = tree.DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf)
            clf = clf.fit(self.X_train, self.y_train)
            y_train = self.y_train.reshape(len(self.y_train), 1)
            y_test = self.y_test.reshape(len(self.y_test), 1)

            pred_train = clf.predict(self.X_train).reshape(self.X_train.shape[0], 1)
            pred_test = clf.predict(self.X_test).reshape(self.X_test.shape[0], 1)
            train_acc = r2_score(y_train, pred_train) * 100
            test_acc = r2_score(y_test, pred_test) * 100
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

    def linear_regression(self):
        start = datetime.now()
        self.__setLog(self.name, 'liniarregression')
        self.__printAndLog('Start liniar regression')

        regr = linear_model.LinearRegression()
        regr.fit(self.X_train, self.y_train)

        test_pred = regr.predict(self.X_test)
        train_pred = regr.predict(self.X_train)

        self.__printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(r2_score(self.y_train, train_pred) * 100, r2_score(self.y_test, test_pred) * 100))
        end = datetime.now()
        self.__printAndLog('Total execution time: {:}'.format(str(end - start)))
        self.__printAndLog('End liniar regression')