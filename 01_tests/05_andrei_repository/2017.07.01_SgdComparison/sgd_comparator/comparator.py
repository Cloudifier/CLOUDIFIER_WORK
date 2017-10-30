import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SgdComparator():

  def __init__(self, logger):
    self.logger = logger
    self.accuracies = []
    self.df = pd.DataFrame(columns = ['Alg', 'Reg', 'TestAcc', 'TrainAcc', 'BestTestAcc', \
      'WorstTestAcc', 'BestTrainAcc', 'WorstTrainAcc', 'TotalTrainTime'])
    self.contor = 0

  def sigmoid(self, z):
    return 1.0 / (1 + np.exp(-z))

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
    cost_histories = None, epochs = None):

    self.accuracies = []
    prediction = self.softmax(test_X.dot(all_theta.T))
    corrects = np.zeros(10)
    wrongs = np.zeros(10)

    for i in range(test_X.shape[0]):
      current_digit = int(test_y[i])
      res = current_digit - np.argmax(prediction[i, :])

      if res == 0:
        corrects[current_digit] += 1
      else:
        wrongs[current_digit] += 1

    t_correct = np.sum(corrects)
    t_wrong = np.sum(wrongs)
    accuracy = 100 * t_correct / float(t_correct + t_wrong)

    self.logger.log("Results using {} solver -- {}".format(type, set), verbosity_level = 2)
    self.logger.log("General accuracy results are: correct={}, wrong={}, accuracy={:.2f}%"
      .format(int(t_correct), int(t_wrong), accuracy), tabs = 1, verbosity_level = 2)

    for i in range(10):
      digit_accuracy =  (100 * corrects[i]) / float(corrects[i] + wrongs[i])
      self.accuracies.append(digit_accuracy)

      self.logger.log("Printing results for target {}: correct={}, wrong={}, accuracy={:.2f}%"
        .format(i, int(corrects[i]), int(wrongs[i]), digit_accuracy), tabs = 1, verbosity_level = 1)

    best_idx, best_val  = self.accuracies.index(max(self.accuracies)), max(self.accuracies)
    worst_idx, worst_val  = self.accuracies.index(min(self.accuracies)), min(self.accuracies)

    self.logger.log("Best accuracy is {:.2f}% for digit {}".format(best_val, best_idx), verbosity_level = 2)
    self.logger.log("Worst accuracy is {:.2f}% for digit {}".format(worst_val, worst_idx), verbosity_level = 2)

    '''
    plt.matshow(np.reshape(all_theta[best_idx, 1:], (28, 28)), cmap = 'gray')
    plt.title("Coefficients best digit " + type + " " + set)
    plt.colorbar()
    plt.show()


    if epochs is not None:
      self.plot(epochs[best_idx], cost_histories[best_idx], "Best digit " + type)

    plt.matshow(np.reshape(all_theta[worst_idx, 1:], (28, 28)), cmap = 'gray')
    plt.title("Coefficients worst digit " + type + " " + set)
    plt.colorbar()
    plt.show()

    if epochs is not None:
      self.plot(epochs[worst_idx], cost_histories[worst_idx], "Worst digit " + type)

    '''

    name = None
    is_reg = None
    if type == "simple without reg":
      name = 'SGD'
      is_reg = False
    elif type == "simple with reg":
      name = 'SGD'
      is_reg = True
    elif type == "momentun without reg":
      name = 'MSGD'
      is_reg = False
    elif type == "momentun with reg":
      name = 'MSGD'
      is_reg = True

    if set == "test":
      self.df = self.df.append(pd.Series([name, is_reg,\
              round(accuracy,2), round(best_val,2), round(worst_val,2), round(sgd_time,3)],\
              index=['Alg', 'Reg', 'TestAcc', 'BestTestAcc', 'WorstTestAcc', 'TotalTrainTime']), ignore_index=True)
    else:
      self.df.loc[self.contor, 'TrainAcc'] = round(accuracy, 2)
      self.df.loc[self.contor, 'BestTrainAcc'] = round(best_val,2)
      self.df.loc[self.contor, 'WorstTrainAcc'] = round(worst_val,2)
      self.df.loc[self.contor, 'WorstTrainAcc'] = round(worst_val,2)
      self.contor += 1

