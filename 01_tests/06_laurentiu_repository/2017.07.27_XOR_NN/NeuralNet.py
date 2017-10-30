import numpy as np
from scipy.special import expit

valid_layers = ["hidden", "output"]

class Layer:

  def __init__(self, tip, n_neurons, previous_layer=None, next_layer=None,
               layer_id=-1, activation='sigmoid'):
    self.tip = tip
    self.n_neurons = n_neurons
    self.previous_layer = previous_layer
    self.next_layer = next_layer
    self.layer_id = layer_id
    self.theta = None
    self.a_array = None
    self.z_array = None
    self.activation = activation
    self.gradient = None
    self.delta = None

  def insert_weights(self):
    if self.tip in valid_layers:
      for i in range(self.previous_layer.n_neurons):
        weights_str = input("[layer {}] Insert {} weights from prev_neuron #{}:\n"
                            .format(self.layer_id, self.n_neurons, i))
        weights = [float(elem) for elem in weights_str.split()]
        if len(weights) != self.n_neurons:
          raise Exception("You should insert {} weights".format(self.n_neurons))

        self.theta.append(weights)
      self.theta = np.array(self.theta)
    else:
      print("Could not insert weights for the input layer")

  def create_weights(self):
    if self.tip in valid_layers:
      nr_prev = self.previous_layer.n_neurons
      nr_curr = self.n_neurons
      self.theta = np.random.uniform(low=-0.5, high=0.5, size=(nr_prev,nr_curr))

  def DMSE(self, y, y_pred):
    m = y_pred.shape[0]

    J = (y_pred-y)
    J = J / (m)
    return J

  def my_activation(self, output):
    output[output<=0] = 0
    output[output>0] = 1
    return output

  def sigmoid(self, z):
    return expit(z)

  def Dsigmoid(self,z):
    return self.sigmoid(z) * (1 - self.sigmoid(z))

  def activate(self):
    self.a_array = self.sigmoid(self.z_array)

  def BProp(self, y_labels):
    prev_act = self.previous_layer.a_array
    deriv = self.Dsigmoid(self.z_array)

    if self.next_layer == None:
      self.delta = self.DMSE(y_labels, self.a_array) * deriv
    else:
      next_layer_delta = self.next_layer.delta
      next_layer_theta = self.next_layer.theta
      self.delta = next_layer_delta.dot(next_layer_theta.T) * deriv

    self.gradient = np.dot(prev_act.T, self.delta)

  def __repr__(self):
    return "Layer %d that has %d neurons" % (self.layer_id, self.n_neurons)

class NeuralNetwork:
  def __init__(self, architecture, inputs, y_labels, manual = False):
    self.inputs = inputs
    self.y_labels = y_labels
    self.n_layers = len(architecture)
    self.layers = list()

    print("Creating a neural network with {} layers".format(self.n_layers))

    self.layers.append(Layer(tip="input", n_neurons=architecture[0], layer_id=0))

    for i in range(self.n_layers-2):
      previous_layer = self.layers[-1]
      self.layers.append(Layer(tip="hidden", previous_layer=previous_layer, n_neurons=architecture[i+1], layer_id=i+1))

    previous_layer = self.layers[-1]
    self.layers.append(Layer(tip="output", previous_layer=previous_layer, n_neurons=architecture[-1], layer_id=self.n_layers-1))

    for i in range(self.n_layers-1):
      if manual:
        self.layers[i+1].insert_weights()
      else:
        self.layers[i+1].create_weights()
      self.layers[i].next_layer = self.layers[i+1]

  def MSE(self):
    m = self.layers[-1].a_array.shape[0]

    J = np.sum((self.layers[-1].a_array-y_labels)**2)
    J = J / (2 * m)
    return J

  def forward_propagation(self):
    self.layers[0].a_array = self.inputs

    for i in range(self.n_layers-1):
      self.layers[i+1].z_array = self.layers[i].a_array.dot(self.layers[i+1].theta)
      self.layers[i+1].activate()

  def backward_propagation(self, learning_rate = 0.5):
    for i in range(self.n_layers-1, 0, -1):
      self.layers[i].BProp(y_labels)

    for i in range(self.n_layers-1, 0, -1):
      self.layers[i].theta -= self.layers[i].gradient

if __name__=="__main__":
  inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
  y_labels = np.array([0,1,1,0])
  y_labels = np.reshape(y_labels, (y_labels.size, -1))
  architecture = [2, 2, 1]

  n = NeuralNetwork(architecture, inputs, y_labels, manual=False)

  for i in range(10000):
    n.forward_propagation()
    n.backward_propagation(learning_rate=1)

  print(n.layers[-1].a_array)
