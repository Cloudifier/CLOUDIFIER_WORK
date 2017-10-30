import numpy as np
from scipy.special import expit
from time import time
import matplotlib.pyplot as plt

valid_activations = ['','direct','sigmoid','relu','softmax']
valid_layers = ['','input','hidden','output']

class DNNLayer:
  def __init__(self, nr_units, layer_name='', prev_layer=None, next_layer=None, activation='', layer_type=''):
    self.layer_name = layer_name
    self.layer_id = -1
    self.layer_type = layer_type
    self.nr_units = nr_units

    if not (layer_type in valid_layers):
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown layer type: " + layer_type)
    if not (activation in valid_activations):
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown activation: " + activation)

    self.activation = activation
    self.prev_layer = prev_layer
    self.next_layer = next_layer
    self.theta = None
    self.a_array = None
    self.z_array = None
    self.delta = 0

    self.is_output = (layer_type == "output")
    self.gradient = None
    self.momentum = None
    self.step = -1

    self.y_ohm = None
    self.y_lbl = None
    self.keep_prob = 0.5

  def Describe(self):
    res = ' Layer:[{}]'.format(self.layer_id)
    res += ' Name:[{}]'.format(self.layer_name)
    res += ' Type:[{}]'.format(self.layer_type)
    res += ' Act:[{}]'.format(self.activation)
    res += ' Units:[{}]'.format(self.nr_units)
    return res


  def OneHotMatrix(self, y, classes):
    n_obs = y.shape[0]
    ohm = np.zeros((n_obs, classes))
    ohm[np.arange(n_obs), y.astype(int)] = 1
    return ohm


  def SetLabels(self, y):
    act = self.activation
    if act == 'softmax':
      self.y_ohm = self.OneHotMatrix(y, self.nr_units)
      self.y_lbl = y
    elif act == 'direct':
      self.y_lbl = y
    else:
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown output computation: " + act)
    return


  def GetOutputLabels(self):
    act = self.activation
    if act == 'softmax':
      yhat = np.argmax(self.a_array, axis=1)
    elif act == 'direct':
      yhat = self.a_array
    else:
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown output computation: " + act)
    return yhat


  def IsComputingOk(self, z):
    res = True
    nr = np.sum(np.isinf(z))
    nr += np.sum(np.isnan(z))
    if nr > 0:
      res = False
    return res


  def ThetaNoBias(self):
    return self.theta[1:, :]


  def sigmoid(self, z):
    return expit(z)


  def Dsigmoid(self, z):
    return self.sigmoid(z) * (1 - self.sigmoid(z))


  def relu(self, z):
    a = np.array(z)
    np.maximum(a, 0, a)
    return a


  def Drelu(self, z):
    a = (z > 0).astype(int)
    return a


  def softmax(self, z):
    z -= np.max(z)
    ez = np.exp(z)

    sm = (ez.T / np.sum(ez, axis=1)).T

    if not self.IsComputingOk(sm):
      print("z={}".format(z))
      print("ez={}".format(ez))
      print("p={}".format(sm))
      print("z nan ={}".format(np.isnan(z).sum()))
      print("ez nan ={}".format(np.isnan(ez).sum()))
      print("p nan ={}".format(np.isnan(sm).sum()))
      print("z inf ={}".format(np.isinf(z).sum()))
      print("ez inf ={}".format(np.isinf(ez).sum()))
      print("p inf ={}".format(np.isinf(sm).sum()))
      raise Exception('INF/NAN value in softmax step {}'.format(
        self.step))
    return sm


  def Activate(self):
    act = self.activation
    if np.count_nonzero(self.z_array) == 0:
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " zero input received for layer:")

    if act == 'sigmoid':
      self.a_array = self.sigmoid(self.z_array)
    elif act == 'relu':
      self.a_array = self.relu(self.z_array)
    elif act == 'softmax':
      self.a_array = self.softmax(self.z_array)
    elif act == 'direct':
      self.a_array = self.z_array
    else:
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown activation !")


  def log_loss(self, y, yhat):
    m = yhat.shape[0]
    J_matrix = y * np.log(yhat)

    if not self.IsComputingOk(J_matrix):
      raise Exception('INF/NAN value in log_loss step {}'.format(
        self.step))

    J = -np.sum(J_matrix)
    J /= m
    return J


  def Dlog_loss(self, y, yhat):
    return yhat - y


  def CostFunction(self, y_labels):
    return self.log_loss(y_labels, self.a_array)


  def J(self):
    if self.y_ohm is not None:
      return self.CostFunction(self.y_ohm)
    else:
      return self.CostFunction(self.y_lbl)


  def DCostFunction(self):
    if self.y_ohm is not None:
      return self.Dlog_loss(self.y_ohm, self.a_array)
    else:
      return self.Dlog_loss(self.y_lbl, self.a_array)


  def GetDerivative(self):
    act = self.activation
    if act == 'sigmoid':
      return self.Dsigmoid(self.z_array)
    elif act == 'relu':
      return self.Drelu(self.z_array)
    elif act == 'direct':
      return 1
    else:
      raise Exception("[DNNLayer:" + str(self.layer_id) + " ERROR]" +
                      " unknown activation !")


  def InitLayer(self, prev_layer):
    if prev_layer == None:
      return

    self.prev_layer = prev_layer
    nr_prev = self.prev_layer.nr_units
    nr_curr = self.nr_units

    # theta initialization (size = (InLayer + 1) x OutLayer)
    self.theta = np.random.randn(nr_prev + 1, nr_curr) * np.sqrt(2 / (nr_prev + 1))


  def FProp(self, inp_array):
    nr_rows = inp_array.shape[0]
    if self.prev_layer == None:
      self.a_array = inp_array
    else:
      self.z_array = self.prev_layer.a_array.dot(self.theta)
      self.Activate()

    if self.layer_type != 'output':
      # add bias if not output layer
      self.a_array = np.c_[np.ones((nr_rows, 1)), self.a_array]


  def BProp(self):
    prev_act = self.prev_layer.a_array
    m = prev_act.shape[0]
    if self.next_layer == None:
      self.delta = self.DCostFunction()
    else:
      deriv = self.GetDerivative()
      next_layer_delta = self.next_layer.delta
      next_layer_ThetaNoBias = self.next_layer.ThetaNoBias()
      self.delta = next_layer_delta.dot(next_layer_ThetaNoBias.T) * deriv

    self.gradient = prev_act.T.dot(self.delta) / m


class NeuralNetwork:
  def __init__(self, logger, hyper_parameters, VERBOSITY=10):
    self.logger = logger
    self.model_prepared = False
    self.layers = list()
    self.hyper_parameters = hyper_parameters
    self.step = 0
    self.VERBOSITY = VERBOSITY
    self.train_cost_history = list()
    self.validation_cost_history = list()


  def Describe(self):
    res = "Layers:"
    for i in range(self.nr_layers):
      res += "\n  " + self.layers[i].Describe()

    self.logger._log(res)
    return res


  def AddLayer(self, NewLayer):
    nr_layers = len(self.layers)
    if nr_layers == 0:
      NewLayer.layer_type = 'input'
    elif self.layers[nr_layers - 1].layer_type == 'output':
      raise Exception("[NeuralNetwork ERROR] Cannot add layer after output!")

    if NewLayer.layer_type == '':
      NewLayer.layer_type = 'hidden'
      if NewLayer.activation == 'softmax':
        NewLayer.layer_type = 'output'

    self.layers.append(NewLayer)


  def PrepareModel(self):
    nr_layers = len(self.layers)
    self.nr_layers = nr_layers
    if nr_layers == 0:
      raise Exception("[NeuralNetwork ERROR] Zero layers !")
    elif nr_layers < 3:
      raise Exception("[NeuralNetwork ERROR] Nr. layers < 3")

    self.nr_weights = 0
    # first check model capacity and generate best thetas
    for i in range(1, nr_layers):
      cunits = self.layers[i].nr_units
      punits = self.layers[i - 1].nr_units
      self.nr_weights += (punits + 1) * (cunits)

    model_size_MB = self.nr_weights * 4 / (1024 * 1024)
    self.logger._log("Model capacity: {:,} weights, {:,.2f}MB"
                    .format(self.nr_weights, model_size_MB))
    if (model_size_MB > 4000):
      self.logger._log("Model requires to much memory, please optimize!")
      return False

    PrevLayer = None
    for i in range(nr_layers):
      self.layers[i].layer_id = i
      self.layers[i].InitLayer(PrevLayer)
      PrevLayer = self.layers[i]
      if i < (nr_layers - 1):
        self.layers[i].next_layer = self.layers[i + 1]

    self.layers[nr_layers - 1].is_output = True
    self.layers[nr_layers - 1].layer_type = 'output'

    if self.VERBOSITY > 0:
      self.Describe()

    self.model_prepared = True
    return True


  def DebugInfo(self, Value, lvl=0):
    if lvl > self.VERBOSITY:
      return
    text = ""
    text += str(Value)
    if self.VERBOSITY >= 10:
      show = True
    else:
      show = False
    self.logger._log(text, show=show)


  def BackProp(self):
    nr_layers = len(self.layers)
    for i in range(nr_layers - 1, 0, -1):
      self.layers[i].BProp()


  def ForwProp(self, X_batch):
    nr_layers = len(self.layers)
    for i in range(nr_layers):
      self.layers[i].FProp(X_batch)


  def SGDStep(self, x_batch, y_batch):
    if not self.model_prepared:
      raise Exception("[NeuralNetwork ERROR] Model not prepared!")

    nr_layers = len(self.layers)
    learning_rate = self.hyper_parameters.learning_rate

    # forward propagation
    self.ForwProp(x_batch)

    # set training labels
    OutputLayer = self.layers[nr_layers - 1]
    OutputLayer.step = self.step
    OutputLayer.SetLabels(y_batch)

    y_preds = OutputLayer.GetOutputLabels()

    J = OutputLayer.J()
    acc = np.sum(y_batch == y_preds) / float(y_preds.shape[0])
    stp = self.step

    if (stp % 1000) == 0:
      self.logger._log('[TRAIN MiniBatch: {}] loss:{:.2f} - acc:{:.2f}'.format(stp, J, acc))
      if self.VERBOSITY >= 10:
        n_to_slice = self.hyper_parameters.batch_size
        if n_to_slice > 10:
          n_to_slice = 10
        d1_slice = y_batch.reshape(y_batch.size)[:n_to_slice]
        d2_slice = y_preds.reshape(y_preds.size)[:n_to_slice]
        self.logger._log('        yTrue:{}'.format(d1_slice.astype(int)))
        self.logger._log('        yPred:{}'.format(d2_slice))

    """
    if (len(self.cost_list) > 1) and self.best_theta:
      if J < min(self.cost_list):
        self.DebugInfo("Found best params so far!")
        self.SaveBestThetas()
    """

    #self.cost_list.append(J)

    self.BackProp()

    # update thetas
    for i in range(nr_layers - 1, 0, -1):
      grad = self.layers[i].gradient
      momentum = self.layers[i].momentum
      if not (momentum is None):
        momentum = self.layers[i].momentum * self.hyper_parameters.momentum_speed
        momentum = momentum + grad
      else:
        momentum = grad
      self.layers[i].theta -= learning_rate * momentum
      self.layers[i].momentum = momentum

    self.step += 1


  def Train(self, xi, yi, X_cross=None, y_cross=None):
    self.SGDStep(xi, yi)

  def train(self, X_train, y_train, X_validation=None, y_validation=None):
    total_train_time = 0
    learning_rate = self.hyper_parameters.learning_rate
    batch_size = self.hyper_parameters.batch_size
    beta = self.hyper_parameters.beta
    epochs = self.hyper_parameters.epochs
    momentum_speed = self.hyper_parameters.momentum_speed
    decay_factor = self.hyper_parameters.decay_factor

    self.logger._log("Training dnn model (randomly initialized)... epochs={}, alpha={:.2f}, batch_sz={}, beta={}, momentum={}, decay={}"
                     .format(epochs, learning_rate, batch_size, beta, momentum_speed, decay_factor))

    n_batches = X_train.shape[0] // batch_size
    lr_patience = 0
    lr_plateau = 5
    for epoch in range(epochs):
      self.logger._log('Epoch {}/{}'.format(epoch + 1, epochs))
      epoch_start_time = time()

      for i in range(n_batches):
        X_batch = X_train[(i * batch_size):((i + 1) * batch_size), :]
        y_batch = y_train[(i * batch_size):((i + 1) * batch_size)]

        self.SGDStep(X_batch, y_batch)

      self.step = 0
      epoch_time = time() - epoch_start_time
      total_train_time += epoch_time

      self.ForwProp(X_train)
      OutputLayer = self.layers[self.nr_layers - 1]
      OutputLayer.SetLabels(y_train)
      y_pred_train = OutputLayer.GetOutputLabels()
      J_train = OutputLayer.J()
      self.train_cost_history.append(J_train)
      acc_train = np.sum(y_train == y_pred_train) / float(y_train.shape[0])

      if (X_validation is not None) and (y_validation is not None):
        self.ForwProp(X_validation)
        OutputLayer = self.layers[self.nr_layers - 1]
        OutputLayer.SetLabels(y_validation)
        y_pred_validation = OutputLayer.GetOutputLabels()
        J_valid = OutputLayer.J()
        self.validation_cost_history.append(J_valid)
        acc_valid = np.sum(y_validation == y_pred_validation) / float(y_validation.shape[0])

        if (epoch > 0) and (decay_factor != 1):
          if J_valid >= self.validation_cost_history[-1]:
            lr_patience += 1
            if self.VERBOSITY >= 10:
              self.logger._log('curr_loss >= last_loss - Increase lr_patience to {}'.format(lr_patience))
          else:
            if (self.validation_cost_history[-1] - J_valid) <= 1e-4:
              lr_patience += 1
              if self.VERBOSITY >= 10:
                self.logger._log('loss decreased slowly - Increase lr_patience to: {}'.format(lr_patience))

          if lr_patience >= lr_plateau:
            lr_patience = 0
            learning_rate *= decay_factor
            if self.VERBOSITY >= 10:
              self.logger._log('lr_patience == {} - alpha: {:.3f}'.format(lr_plateau, learning_rate))

        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}% - val_loss: {:.2f} - val_acc: {:.2f}%\n'
                         .format(epoch_time, J_train, acc_train * 100, J_valid, acc_valid * 100))
        self.validation_cost_history.append(J_valid)

      else:
        self.logger._log('{:.2f}s - loss: {:.2f} - acc: {:.2f}%\n'.format(epoch_time, J_train, acc_train * 100))

    self.logger._log('Total TRAIN time: {:.2f}s'.format(total_train_time))



  def predict(self, X_test, y_test):
    self.ForwProp(X_test)
    OutputLayer = self.layers[self.nr_layers - 1]
    OutputLayer.SetLabels(y_test)
    y_preds = OutputLayer.GetOutputLabels()
    J_test = OutputLayer.J()
    accuracy = sum(y_preds == y_test) / (float(len(y_test)))

    self.logger._log("Predicting ... test_loss: {:.2f} - test_acc: {:.2f}%".format(J_test, accuracy * 100))

    return y_preds

  def plot_cost_history(self, cost_history):
    plt.plot(np.arange(0, len(cost_history)), cost_history)
    # plt.title('Convergence of the Linear Regression')
    plt.xlabel('Epoch #')
    plt.ylabel('Cost Function')
    plt.show()


if __name__ == "__main__":
  print("DNN engine - Cannot be run.")