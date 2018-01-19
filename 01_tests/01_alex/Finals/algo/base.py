import sys
import logging
import itertools
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
from Finals.algo.nn import CostFuncEnum, NN
from Finals.models.model_statistics import ModelStatistics

class Base(object):
    def __init__(self, name, X_train, X_test, y_train, y_test, folds=5):
        self.name = name
        self.X_full_train = X_train
        self.X_test = X_test
        self.y_full_train = y_train
        self.y_test = y_test
        self.folds = folds

    def _printAndLog(self, str):
        print(str)
        logging.info(str)

    def _setLog(self, name, method):
        logging.basicConfig(filename='_logs/{:}_{:}_{:}.txt'.format(name, method, datetime.now().strftime('%Y%m%d%H%M%S')), level=logging.DEBUG, format='%(message)s')

    def _splitData(self, fold):
        m_full_train, n_full_train = self.X_full_train.shape
        batch = m_full_train // self.folds
        start = fold * batch
        end = start + batch if fold <= self.folds - 2 else start + (m_full_train - start - 1)
        X_temp = np.copy(self.X_full_train)
        y_temp = np.copy(self.y_full_train)
        X_validation = self.X_full_train[start: end]
        y_validation = self.y_full_train[start: end]
        y_validation = y_validation.reshape(end - start, 1)
        X_train = np.delete(X_temp, (start, end), axis=0)
        y_train = np.delete(y_temp, (start, end), axis=0)
        y_train = y_train.reshape(y_train.shape[0], 1)

        return X_train, X_validation, y_train, y_validation

    def NN(self, epochs, layers, costFunction, alphas=None, batch_sizes=None, lmbds=None, useBias=True,
           early_stopping=False, tol=None, theta_init=None):
        start_time = datetime.now()
        self._setLog(self.name, 'nn')
        self._printAndLog('Start NN')

        accs = list()
        best_test_acc, best_train_acc = 2 * (0,)
        best_alpha, best_batch_size, best_lmbd = 3 * (None,)

        # use cases
        if alphas is None:
            alphas = np.arange(0.001, 0.3, 0.03)

        if batch_sizes is None:
            batch_sizes = [8, 16]

        if lmbds is None:
            lmbds = np.arange(0, 1, 0.1)

        use_cases = list(itertools.product(alphas, batch_sizes, lmbds))
        self._printAndLog(f'Grid search: {len(use_cases)} use cases, {self.folds} folds, {len(use_cases) * self.folds} total iterations')

        working_use_cases = list()
        not_working_use_cases = list()
        for alpha, batch_size, lmbd in use_cases:
            self._printAndLog("Batch size: {}, Alpha: {}, Lmbd: {}".format(batch_size, alpha, lmbd))
            use_case_accs = list()
            for fold in range(self.folds):
                self._printAndLog(f'Fold nr {fold + 1}/{self.folds}. Mean acc (train, validation): {0 if len(use_case_accs) == 0 else np.mean(use_case_accs, axis=0)}%')
                try:
                    X_train, X_validation, y_train, y_validation = self._splitData(fold)
                    nn = NN(layers, costFunction=costFunction, useBias=useBias, early_stopping=early_stopping, tol=tol, theta_init=theta_init)
                    nn.fit(X_train, y_train, epochs=epochs, batchSize=batch_size, alpha=alpha, lmbd=lmbd)

                    train_pred, train_acc = nn.predict_with_accuracy(X_train, y_train)
                    validation_pred, validation_acc = nn.predict_with_accuracy(X_validation, y_validation)
                    use_case_accs.append((train_acc, validation_acc))

                    self._printAndLog("Batch size: {}, Alpha: {}, Lmbd: {}, Train acc: {}%, Test acc: {}%".format(batch_size, alpha, lmbd, train_acc, validation_acc))

                    if best_test_acc < validation_acc:
                        best_test_acc = validation_acc
                        best_train_acc = train_acc
                        best_alpha = alpha
                        best_lmbd = lmbd
                        best_batch_size = batch_size
                    working_use_cases.append((alpha, batch_size, lmbd))
                except  Exception as e:
                    self._printAndLog(f'Use case not converging. {e}')
                    not_working_use_cases.append((alpha, batch_size, lmbd))
                    raise

            if len(use_case_accs) > 0:
                accs.append(np.mean(use_case_accs))

        if len(working_use_cases) > 0:
            with open('_dataframes/nn_working.csv', 'w') as csv_file:
                for item in working_use_cases:
                    csv_file.write("\n alpha= {};batch_size= {}; lmbd= {}".format(item[0], item[1], item[2]))

        if len(not_working_use_cases) > 0:
            with open('_dataframes/nn_not_working.csv', 'w') as csv_file:
                for item in not_working_use_cases:
                    csv_file.write("\n alpha= {};batch_size= {}; lmbd= {}".format(item[0], item[1], item[2]))

        self._printAndLog(
            'Best train acc: {:.2f}%, best test acc: {:.2f}%, best alpha: {:}, best batch size: {:}, best lmbd: {:}'.format(
                best_train_acc, best_test_acc, best_alpha, best_batch_size, best_lmbd))
        end_time = datetime.now()
        self._printAndLog('Total execution time: {}'.format(str(end_time - start_time)))
        self._printAndLog('End NN')

        mean_acc = (0, 0)
        if len(accs) > 0:
            mean_acc = np.mean(accs, axis=0)
        return ModelStatistics('NN', len(use_cases), best_test_acc, best_train_acc, round(mean_acc[0], 2), round(mean_acc[1], 2), 'batch_size= {:}; alpha= {:}, lmbd= {:}'.format(best_batch_size, best_alpha, best_lmbd),'')