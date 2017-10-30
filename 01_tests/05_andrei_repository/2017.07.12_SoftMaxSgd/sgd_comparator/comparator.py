import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SgdComparator():

  def __init__(self, logger):
    self.logger = logger
    self.accuracies = []
    self.df = pd.DataFrame(columns = ['Alg', 'Reg', 'TestAcc', 'TrainAcc', 'TotalTrainTime'])
    self.contor = 0

  def softmax(self, z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

  def plot(self, n_epochs, cost_history, title_str):
    plt.plot(np.arange(0, n_epochs), cost_history)
    plt.title(title_str)
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Value')
    plt.show()

  def show_results(self, type, set, all_theta, test_X, test_y, sgd_time, \
    cost_history = None, epochs = None):

    probabilities = self.softmax(np.dot(test_X, all_theta))
    preds = np.argmax(probabilities, axis=1)
    accuracy = sum(preds == test_y)/(float(len(test_y)))
    self.logger.log("Results using {} solver -- {}".format(type, set), verbosity_level = 2)
    self.logger.log("General accuracy result: accuracy={:.2f}%"
      .format(accuracy*100), tabs = 1, verbosity_level = 2)

    if epochs is not None:
      self.plot(epochs, cost_history, "Loss values plotted by epoch")

    name = None
    is_reg = None
    if type == "simple without reg":
      name = 'SGD'
      is_reg = False
    elif type == "simple with reg":
      name = 'SGD'
      is_reg = True
    elif type == "momentum without reg":
      name = 'MSGD'
      is_reg = False
    elif type == "momentum with reg":
      name = 'MSGD'
      is_reg = True

    if set == "test":
      self.df = self.df.append(pd.Series([name, is_reg,\
              round(accuracy*100,2), round(sgd_time,3)],\
              index=['Alg', 'Reg', 'TestAcc', 'TotalTrainTime']), ignore_index=True)
    else:
      self.df.loc[self.contor, 'TrainAcc'] = round(accuracy*100, 2)
      self.contor += 1
