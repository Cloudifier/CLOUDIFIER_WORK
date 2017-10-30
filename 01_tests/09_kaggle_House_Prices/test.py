import pandas as pd
import math as mt

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

	coefs = [0.0 for i in range(len(train_dset[0]))]

	for epoch in range(n_epoch):
		sum_error = 0

		for row in train_dset:
			y_r = row[-1]
			y_p = make_prediction(row, coefs)

			# Sum (y_p - y_r)^2
			error = y_p - y_r
			sum_error += error**2

			# a0(t+1) = a0(t) - learning_rate * error(t)
			coefs[0] = coefs[0] - l_rate * error

			# ai(t+1) = ai(t) - learning_rate * error(t) * xi
			for i in range(len(row)-1):
				coefs[i + 1] = coefs[i + 1] - l_rate * error * row[i]


		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coefs

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train_dset, test, l_rate, n_epoch):
	predictions = list()

	coefs = get_coefs_sgd(train_dset, l_rate, n_epoch)
	for row in test:
		y_p = make_prediction(row, coefs)
		predictions.append(y_p)

	return(predictions)

# Train model on first 1300 entries and test it on the last 160
def train_and_predict(dset):
	predictions = linear_regression_sgd(dset.ix[:1299,1:].values, \
		dset.ix[1300:,1:].values, 0.001, 5000)

	for i in range(160):
		print("Prediction is {} // Real value is {}\n".format( \
			predictions[i], dset.ix[1300+i][-1]))

	return predictions

# Calculate root mean squared error
def get_rmse(actual, predicted):

	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[1300 + i]
		sum_error += (prediction_error ** 2)

	mean_error = sum_error / float(len(actual))
	return mt.sqrt(mean_error)

# Main
if __name__ == "__main__":

	dset = pd.read_csv("_data/train_simple.csv")
	normalize_values(dset)
	predictions = train_and_predict(dset)

	rmse = get_rmse(dset.ix[1300:]['SalePrice'], predictions)
	print("RMSE: {}".format(rmse))
