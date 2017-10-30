import numpy as np

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)
  div_val[div_val == 0] = 1
  return (X - min_val) / div_val

if __name__=='__main__':
  print("Library module. No main function")





