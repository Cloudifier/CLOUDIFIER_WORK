import itertools
import numpy as np
from datetime import datetime
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn import linear_model
from Finals.algo.base import Base
from Finals.models.model_statistics import ModelStatistics

class Regression(Base):
    def decission_tree(self):
        start = datetime.now()
        self._setLog(self.name, 'decissiontree')
        self._printAndLog('Start decission tree')

        depths = range(2, 20)
        leafs = range(2, 20)
        use_cases = list(itertools.product(depths, leafs))
        accs = list()

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

            train_acc = round(r2_score(y_train, pred_train) * 100, 2)
            test_acc = round(r2_score(y_test, pred_test) * 100, 2)
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

    def linear_regression(self):
        start = datetime.now()
        self._setLog(self.name, 'liniarregression')
        self._printAndLog('Start liniar regression')

        regr = linear_model.LinearRegression()
        regr.fit(self.X_train, self.y_train)

        test_pred = regr.predict(self.X_test)
        train_pred = regr.predict(self.X_train)

        train_acc = round(r2_score(self.y_train, train_pred) * 100, 2)
        test_acc = round(r2_score(self.y_test, test_pred) * 100, 2)

        self._printAndLog('Train acc: {:.2f}%, Test acc: {:.2f}%'.format(train_acc, test_acc ))
        end = datetime.now()
        self._printAndLog('Total execution time: {:}'.format(str(end - start)))
        self._printAndLog('End liniar regression')

        return ModelStatistics('logistic regression', 1, test_acc, train_acc, test_acc, train_acc, '', '')
