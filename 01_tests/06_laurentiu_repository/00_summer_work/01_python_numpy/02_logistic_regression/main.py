import os
import numpy as np
import platform
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class dotdict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


def get_paths(current_platform, data_dir):

  if current_platform == "Windows":
    base_dir = os.path.join("D:/", "GoogleDrive/_cloudifier_data/09_tests")
  else:
    base_dir = os.path.join(os.path.expanduser("~"), "Google Drive/_cloudifier_data/09_tests")

  utils_path = os.path.join(base_dir, "Utils")
  data_path = os.path.join(base_dir, data_dir)

  return base_dir, utils_path, data_path

def min_max_scaler(X):
  min_val = np.min(X, axis=0)
  div_val = np.max(X, axis=0) - np.min(X, axis=0)

  div_val[div_val==0] = 1
  return (X - min_val) / div_val

def fetch_data():
  _, utils_path, mnist_path = get_paths(platform.system(), "_MNIST_data")
  logger_lib = SourceFileLoader("logger", os.path.join(utils_path, "base.py")).load_module()
  logger = logger_lib.Logger(lib='LOGREG')

  from sklearn.datasets import fetch_mldata
  mnist = fetch_mldata('MNIST original', data_home=mnist_path)

  X = mnist.data
  y = mnist.target

  X = min_max_scaler(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)
  X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test,
                                                                test_size=0.5,
                                                                random_state=42)

  return dotdict({'train': (X_train, y_train),
                  'test': (X_test, y_test),
                  'validation': (X_validation, y_validation)}), logger


def pretty_print_conf_matrix(y_true, y_pred,
                             classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.Blues):
    """
    Mostly stolen from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Normalization changed, classification_report stats added below plot
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.gca().set_position((.085, .44, .88, .525))

    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11)
        plt.autoscale(True)


    # Add Precision, Recall, F-1 Score as Captions Below Plot
    rpt = classification_report(y_true.astype(int), y_pred)
    rpt = rpt.replace('avg / total', '      avg')
    rpt = rpt.replace('support', 'N Obs')

    plt.figtext(.39, .0001, rpt)
    plt.savefig('Confuzie')
    plt.show()

def create_bar_plots(y_test, yhat):
  n_groups = len(np.unique(y_test))
  l1 = []
  l2 = []
  for i in range(n_groups):
    l1.append((y_test==i).sum())
    l2.append((yhat==i).sum())
  t1 = tuple(l1)
  t2 = tuple(l2)
  # create plot
  fig, ax = plt.subplots()
  index = np.arange(n_groups)
  bar_width = 0.3
  opacity = 0.8
  rects1 = plt.bar(index, t1, bar_width,
                   alpha=opacity,
                   color='b',
                   label='Real')
  rects2 = plt.bar(index + bar_width, t2, bar_width,
                   alpha=opacity,
                   color='g',
                   label='Pred')
  plt.xlabel('Target')
  plt.ylabel('Count')
  plt.title('Target real vs prezis')
  plt.xticks(index + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  data_sets, logger = fetch_data()
  VERBOSITY = 10

  lr = LogisticRegression(logger, VERBOSITY)
  lr.train(data_sets.train[0], data_sets.train[1],
           data_sets.validation[0], data_sets.validation[1],
           epochs=25,
           batch_size=10,
           learning_rate=0.001,
           beta=0.0005,
           momentum_speed=0.85,
           decay_factor=0.65)

  X_test, y_test = data_sets.test
  yhat = lr.predict(X_test,y_test)

  pretty_print_conf_matrix(y_test,yhat,classes=[str(i) for i in range(10)])

