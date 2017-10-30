import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
from utils import min_max_scaler

class KnnClassifier():

  def __init__(self, df, split_params, logger):
    self.df = df
    self.split_params = split_params
    self.logger = logger

  def preprocess(self):
    X = np.array(self.df.iloc[:, 1:].values)
    y = np.array(self.df.iloc[:, 0].values, dtype = int)
    X, nX, y, ny = train_test_split(X, y, test_size = self.split_params[0],
        random_state = self.split_params[1])

    X = min_max_scaler(X)

    self.X = X
    self.y = y


  def predict(self, values_to_predict, positions, k = 1):

    self.logger.log("\tStart predicting values for {} samples"
                    .format(len(values_to_predict)), verbosity_level = 1)


    empty_img_threshold = 35
    positions_empty_im = np.where(values_to_predict.sum(axis=1) < empty_img_threshold)

    sums = pairwise_distances(self.X, values_to_predict, n_jobs = multiprocessing.cpu_count(),metric = "manhattan")

    top_k_sums = np.sort(sums,axis=0)[:k]
    top_k_indexes = np.argsort(sums, axis=0)[:k]
    top_k = self.y[top_k_indexes]
    vals, counts = mode(top_k, axis=0)

    not_confident_threshold = 60
    positions_not_confident = np.where(np.average(top_k_sums,axis=0) > not_confident_threshold)

    vals = vals.flatten()
    vals[positions_empty_im] = -1
    vals[positions_not_confident] = -2
    counts = counts.flatten()

    good_positions = np.where(vals >= 0)
    good_positions = good_positions[0]
    if good_positions.size == 0:
        return -1, (-1, -1), vals, counts, top_k_sums

    most_confident_position = np.average(top_k_sums[:,good_positions],axis=0).argmin()

    predicted_val = vals[good_positions[most_confident_position]]
    predicted_pos = positions[good_positions[most_confident_position]]

    self.logger.log("\tFinished predicting values", verbosity_level = 1)

    return predicted_val, predicted_pos, vals, counts, top_k_sums

if __name__=='__main__':
  print("Library module. No main function")