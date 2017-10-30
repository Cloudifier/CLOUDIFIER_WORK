import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class LinearRegressionSgd():

    def __init__(self, df, test_size):
        self.df = df.copy()
        self.test_size = test_size
        self.model = None

    def normalize_values(self):

        cols_to_norm = self.df.columns.tolist()[:-1]
        self.df[cols_to_norm] = self.df[cols_to_norm].apply(lambda x: \
                                 (x - x.min()) / (x.max() - x.min()))

    def compute_cost(self, X, y, theta):

        m = y.size
        predictions = X.dot(theta)
        errors = (predictions - y)

        cost = (1.0 / (2 * m)) * errors.T.dot(errors)

        return cost


    def gradient_descent(self, X, y, theta, alpha, num_iters):

        m = y.size
        cost_function_history = np.zeros(shape = (num_iters, 1))

        for i in range(num_iters):

            predictions = X.dot(theta)

            theta_size = theta.size

            for it in range(theta_size):

                temp = X[:, it]
                temp.shape = (m, 1)

                errors = (predictions - y) * temp

                theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors.sum()

            cost_function_history[i, 0] = self.compute_cost(X, y, theta)

        return theta, cost_function_history

    def train_model(self, num_iters):

        train_df = self.df[:-self.test_size]
        num_predictors = train_df.shape[1] - 2

        X = np.array(train_df.ix[:, 1:-1].values)
        y = np.array(train_df.ix[:, -1].values)

        m = y.size
        y.shape = (m, 1)

        new_X = np.ones(shape = (m, num_predictors + 1))
        new_X[:, 1 : num_predictors + 1] = X

        alpha = 0.01
        theta = np.zeros(shape = (num_predictors + 1, 1))

        theta, cost_history = self.gradient_descent(new_X, y, theta, alpha, \
                                                                    num_iters)

        self.model = theta

        plt.plot(np.arange(num_iters), cost_history)
        plt.title('Convergence plot of gradient descendent')
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function')
        plt.show()


    def predict_values(self):

        test_df = self.df[-self.test_size:]
        X = np.array(test_df.ix[:, 1:-1].values)
        num_predictors = test_df.shape[1] - 2

        m = self.test_size
        new_X = np.ones(shape=(m, num_predictors + 1))
        new_X[:, 1 : num_predictors + 1] = X

        predictions = new_X.dot(self.model)

        for i in range(self.test_size):
            crt_idx = len(self.df) - self.test_size + i
            print("Real price {0:0.2f}".format(self.df.ix[crt_idx][-1]))
            print("Predicted price {0:0.2f}".format(predictions[i][0]))
            print()

if __name__ == "__main__":
    os_home = os.path.expanduser("~")
    csv_file = os.path.join(os_home, 'Google Drive/_cloudifier_data/09_tests/_pricing_data/prices.csv')
    df = pd.read_csv(csv_file)
    linear_model = LinearRegressionSgd(df, 50)

    linear_model.normalize_values()
    linear_model.train_model(1200)
    linear_model.predict_values()