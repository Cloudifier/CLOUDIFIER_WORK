import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from scipy.stats import mode
import multiprocessing
import time

class KnnClassifier():

  def __init__(self, df, test_size, logger):
    self.df = df
    self.test_size = test_size
    self.logger = logger

  def preprocess(self):
    X = np.array(self.df.iloc[:, 1:].values)
    y = np.array(self.df.iloc[:, 0].values, dtype = int)

    self.normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = self.normalizer.fit_transform(X)

    self.logger.log("Finished normalizing data")

    self.train_X, self.test_X, self.train_y, self.test_y  = \
        train_test_split(X, y, test_size = self.test_size, random_state=42)

    self.logger.log("Finished spliting data into train ({}) and test ({})".
      format(1 - self.test_size, self.test_size))

  def predict(self, k = 1):

    start_time = time.time()
    self.logger.log("Starting prediction using KNN with k = {}".format(k))


    total_size = self.test_y.shape[0]
    self.total_size = total_size
    chunks = int(total_size / 1500) + 1 if total_size % 1500 != 0 else 0;

    self.logger.log("Splitting test data into {} chunks".format(chunks))

    self.logger.log("Making prediction for each chunk...")
    vals = []
    counts = []
    for chunk in tqdm(range(chunks)):
      sums = pairwise_distances(self.train_X, self.test_X[chunk * 1500 : chunk * 1500 + 1500], metric='manhattan',n_jobs=multiprocessing.cpu_count())
      top_k_indexes = np.argsort(sums, axis=0)[:k]
      top_k = self.train_y[top_k_indexes]
      crt_vals, crt_counts = mode(top_k, axis=0)

      crt_vals = crt_vals.flatten()
      crt_counts = crt_counts.flatten()

      vals.extend(crt_vals)
      counts.extend(crt_counts)

    self.logger.log("Finished prediction for all chunks in {:.2f}s".
      format(time.time() - start_time))

    return vals, counts

  def compute_accuracy(self, vals, counts):

    corrects = 0
    wrongs = 0

    self.logger.log("Evaluating model...")
    size = self.test_y.shape[0]
    for  i in tqdm(range(size)):
      if vals[i] == self.test_y[i]:
        corrects += 1
      else:
        wrongs += 1

    self.logger.log("Corrects/Wrongs:{}/{}".format(corrects, wrongs))
    self.logger.log("Accuracy: {:.2f}%".format((corrects*100)/size))

if __name__=='__main__':
  print("Library module. No main function")

