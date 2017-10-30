import numpy as np
from sklearn.svm import SVC
from logger import Logger
import os
from sklearn.model_selection import train_test_split

def minmax_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  simple = False

  if simple:
    from sklearn.datasets import load_digits
    digits = load_digits()

    X = digits.data
    y = digits.target

    X = minmax_scaler(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)

    data = { 'train': { 'X': X_train, 'y': y_train },
             'test': { 'X': X_test, 'y': y_test } }
  else:
    os_home = os.path.expanduser("~")
    data_home = os.path.join(os_home, 'Google Drive/_cloudifier_data/09_tests/_MNIST_data')
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    X = mnist.data
    y = mnist.target

    X = minmax_scaler(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                  test_size=0.5,
                                                                  random_state=42)

    data = { 'train': { 'X': X_train, 'y': y_train },
             'test': { 'X': X_test, 'y': y_test },
             'validation': { 'X': X_validation, 'y': y_validation } }

  return data


if __name__=="__main__":
  logger = Logger(show = True, verbosity_level = 2)
  data = fetch_data()
  clf = SVC(kernel='linear', verbose=True)
  clf.fit(data['train']['X'], data['train']['y'])