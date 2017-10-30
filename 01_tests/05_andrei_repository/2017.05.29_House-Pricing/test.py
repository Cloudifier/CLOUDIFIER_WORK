import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt

# Normalize all atributes
# v_norm = (v - v_min) / v_max
def normalize_values(dset):
	cols_to_norm = dset.columns.tolist()[:-1]

	dset[cols_to_norm] = dset[cols_to_norm].apply(lambda x: \
		(x - x.min()) / (x.max() - x.min()))


# Make a prediction with coefficients
# y = a0 + a1*x1 + a2*x2 + ... + an*xn
def make_prediction(dset_row, coefs):

	y_p = coefs[0]

	for i in range(len(dset_row) - 1):
		y_p += coefs[i + 1] * dset_row[i]

	return y_p


# Estimate linear regression coefficients using stochastic gradient descent
def get_coefs_sgd(train_dset, l_rate, n_epoch):

	coefs = [0.0 for i in range(train_dset.shape[0])]
	errors = []

	for epoch in range(n_epoch):
		sum_error = 0

		for row in train_dset:
			y_r = row[-1]
			y_p = make_prediction(row, coefs)

			# Sum (y_p - y_r)^2
			error = y_p - y_r
			sum_error += error ** 2

			# a0(t+1) = a0(t) - learning_rate * error(t)
			coefs[0] = coefs[0] - l_rate * error

			# ai(t+1) = ai(t) - learning_rate * error(t) * xi
			for i in range(len(row)-1):
				coefs[i + 1] = coefs[i + 1] - l_rate * error * row[i]

		errors.append(sum_error)
		sum_error /= train_dset.shape[0]

		if epoch % 10 == 0:
			print(">epoch={}, lrate={}, error={}".format( \
											  epoch, l_rate, abs(error)))

	plt.plot(np.arange(n_epoch), errors)
	plt.title('Convergence plot of gradient descendent')
	plt.xlabel('Iterations')
	plt.ylabel('Cost Function')
	plt.show()

	return coefs

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train_dset, test, l_rate, n_epoch):
	predictions = list()

	coefs = get_coefs_sgd(train_dset, l_rate, n_epoch)

	for row in test:
		y_p = make_prediction(row, coefs)
		predictions.append(y_p)

	return predictions

# Train model on all data except the last 50 and test it on the last 50
def train_and_predict(dset):

	train_df = dset[:-50]
	test_df  = dset[-50:]

	predictions = linear_regression_sgd(train_df.ix[:,1:].values, \
		test_df.ix[:,1:].values, 0.001, 100)

	for i in range(50):
		crt_idx = len(dset) - 50 + i
		print("Prediction is {0:0.2f} // Real value is {1:0.2f}\n".format( \
			predictions[i], dset.ix[crt_idx][-1]))

	return predictions

# Main
if __name__ == "__main__":

	dset = pd.read_csv("_data/train_simple.csv")
	normalize_values(dset)
	predictions = train_and_predict(dset)

