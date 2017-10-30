from abc import ABC, abstractmethod
import numpy as np
import time
import pandas as pd
from utils import min_max_scaler, softmax, relu
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import cv2
import time
import scipy.misc

_DEBUG_ = False
_TEST_  = True

class Slider(ABC):

  def __init__(self, scene_files, scene_sizes, step_size, window_size, logger):
    self.scene_files = scene_files
    self.scene_sizes = scene_sizes
    self.window_size = window_size
    self.step_size = step_size
    self.logger = logger
    self.crt_df = None
    self.crt_idx = 0
    self.results = [[0, 0, 0] for i in range(len(self.scene_files))]
    self.epsilon = 2
    np.set_printoptions(precision = 2, suppress = True)

    self.scene_inference = {}

    if _TEST_ :
      self.scenes = []
      self.correct_windows = []
      self.windows = []
      self.over95_windows = []

      for i in range(15):
        self.over95_windows.append([])

  def sliding_window(self, image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
      for x in range(0, image.shape[1] - window_size[0], step_size):
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

  def check_position(self, i, predicted_pos):
    return (abs(self.img_pos[i][1] - predicted_pos[1]) < self.epsilon and
      abs(self.img_pos[i][0] - predicted_pos[0]) < self.epsilon)

  @abstractmethod
  def process_window(self, window):
    pass

  def process_results(self, results, idx):

    max_prob = 0
    max_tuple = (0, 0)
    for crt_t in results:

      if crt_t[1] > max_prob:
        max_prob = crt_t[1]
        max_tuple = crt_t

      if _TEST_ :
        if crt_t[1] > .95:
          self.over95_windows[self.crt_idx * 5 + idx].append(crt_t)

    return max_tuple


  def check_target_pos(self, pos1, pos2, epsilon = 15):
    return (abs(pos1[1] - pos2[1]) < epsilon and abs(pos1[0] - pos2[0]) < epsilon)




  def process_scene_inference(self):

    #[self.scene_inference[i]: [] for i in range(10)]
    for crt_t in self.scene_results:

      target, prob, pos = crt_t


      if prob < .9:
        continue

      if not (target in self.scene_inference.keys()):
        self.scene_inference[target] = []
        self.scene_inference[target].append( (pos, prob) )
      else:
        update_pos = False
        for i in range(len(self.scene_inference[target])):
          item = self.scene_inference[target][i]
          if self.check_target_pos(item[0], pos, epsilon = 27):
            update_pos = True
            if prob > item[1]:
              self.scene_inference[target][i] = (pos, prob)
            break
        if not update_pos:
          self.scene_inference[target].append( (pos, prob) )

  def slide_over_image(self, image, real_idx, scene_idx):

    self.scene_results = []
    self.all_windows = {}

    image = np.load('scene1_300x300.npy')
    image = min_max_scaler(image)

    for (x, y, window) in self.sliding_window(image, step_size = self.step_size,
      window_size = self.window_size):

      if np.sum(window) == 0:
        continue

      if sum(window[0, :] == 0) != self.window_size[1]:
        continue

      if sum(window[self.window_size[1] - 1, :] == 0) != self.window_size[1]:
        continue

      if sum(window[:, 0] == 0) != self.window_size[0]:
        continue


      if sum(window[:, self.window_size[1] - 1] == 0) != self.window_size[0]:
        continue

      self.all_windows[(x, y)] = window

      if _DEBUG_:
        self.logger.log("\tTaking window at pos = {}, {}...".format(y,x))

      predicted_val, probability = self.process_window(window)
      self.scene_results.append( (predicted_val[0], probability, (y, x)) )

      if _TEST_:
        if (x == self.img_pos[real_idx][0]) and (y == self.img_pos[real_idx][1]):
          self.correct_windows.append( (window, predicted_val, probability) )

      txt = "{} {:.2f}%".format(predicted_val[0], probability * 100, y, x)
      #if probability * 100 >= 90:
        #print(txt)


      clone = image.copy()

      cv2.rectangle(clone, (x, y), (x + self.window_size[0], y + self.window_size[1]), (69, 244, 66), 2)
      if y >= 50:
        cv2.putText(clone, txt, (x - 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)
      else:
        cv2.putText(clone, txt, (x - 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)

      cv2.imshow("Window", clone)
      cv2.waitKey(1)
      time.sleep(0.25)

    self.create_final_img(image)
    #final_val, final_max, final_position, final_window = self.process_results(results, scene_idx)

    #return (final_val, final_position, final_window, final_max)

  def read_df(self):
    self.crt_df = pd.read_pickle(self.scene_files[self.crt_idx])
    self.X = np.array(self.crt_df.iloc[:, 3:].values, dtype = float)
    self.X = min_max_scaler(self.X)

    self.y = np.array(self.crt_df.iloc[:, 0].values, dtype = int)
    self.img_pos = np.array(self.crt_df.iloc[:, 1:3].values, dtype = int)


  def create_final_img(self, image):
    self.process_scene_inference()
    keys = self.scene_inference.keys()

    clone = image.copy()

    for target in keys:
      for item in self.scene_inference[target]:
        x = item[0][1]
        y = item[0][0]
        prob = item[1]
        cv2.rectangle(clone, (x, y) , (x + self.window_size[0], y + self.window_size[1]), (69, 244, 66), 2)
        txt = "{} {:.2f}%".format(target, prob * 100)
        if y >= 50:
          cv2.putText(clone, txt, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)
        else:
          cv2.putText(clone, txt, (x - 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)

    cv2.imshow("Window", clone)
    cv2.waitKey(1)


  def slide_over_df(self, idx):

    self.crt_idx = idx
    self.read_df()

    num_scenes = self.crt_df.shape[0]

    self.logger.log("Sliding {} test scenes of size {}x{} with {}x{} windows and step_size={}".format(num_scenes, self.scene_sizes[self.crt_idx][0], self.scene_sizes[self.crt_idx][1],
      self.window_size[0], self.window_size[1], self.step_size))

    num_scenes = 1
    for i in range(num_scenes):
      saved_i = i
      if _TEST_:
        #i = random.randrange(0, self.crt_df.shape[0])
        #i = 1
        pass

      self.logger.log("Start sliding scene #{}; position of the image with target = {} in the scene = ({}, {})".format(i, self.y[i], self.img_pos[i][1], self.img_pos[i][0]), tabs = 1)

      start_time = time.time()
      image = self.X[i].reshape(self.scene_sizes[self.crt_idx][0],
        self.scene_sizes[self.crt_idx][1])
      #val, pos, window, prob =
      self.slide_over_image(image, i, saved_i)


      #cv2.waitKey(1)
      #time.sleep(10)



      """
      clone = image.copy()
      cv2.rectangle(clone, (pos[0], pos[1]), (pos[0] + self.window_size[0], pos[1] + self.window_size[1]), (69, 244, 66), 2)
      txt = "{} {:.2f}%".format(val[0], prob * 100)
      if pos[1] >= 50:
        cv2.putText(clone, txt, (pos[0] - 5, pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)
      else:
        cv2.putText(clone, txt, (pos[0] - 5, pos[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 250)
      cv2.imshow("Window", clone)
      cv2.waitKey(1)
      time.sleep(10)
      """

      '''
      self.logger.log("Find {} at position {} with probability {:.2f}".format(val, tuple(reversed(pos)), prob),
        tabs = 2)

      if val == self.y[i]:
        if self.check_position(i, pos):
          self.results[self.crt_idx][0] += 1
          self.logger.log("Correct target, correct position", tabs = 2)
        else:
          self.results[self.crt_idx][1] += 1
          self.logger.log("Correct target, wrong position", tabs = 2)
      else:
          self.results[self.crt_idx][2] += 1
          self.logger.log("Wrong target", tabs = 2)

      self.logger.log("Scene slided in {:.2f}s".format(time.time()-start_time), tabs = 2)

      if _TEST_:
        self.windows.append( (window, val, prob, pos) )
        self.scenes.append( (self.X[i], self.y[i], self.img_pos[i]) )
      '''

  def slide(self):

    start_time = time.time()
    num_dfs = len(self.scene_files)
    num_dfs = 1

    for i in range(num_dfs):
      start_time = time.time()
      self.slide_over_df(i)

      #self.logger.log("Test scenes of sz {}x{} in {:.2f}s; corrects={}, partially_wrongs={}, wrongs={}"
      #                 .format(self.scene_sizes[i][0], self.scene_sizes[i][1], time.time() - start_time, self.results[i][0], self.results[i][1], self.results[i][2]))
      #print()

    #for item in self.results:
      #self.logger.log(str(item[0]) + " " + str(item[1]) + " " + str(item[2]))



class LogRegSlider(Slider):

  def __init__(self, scene_files, scene_sizes, theta_file, step_size, window_size, logger):
    super().__init__(scene_files, scene_sizes, step_size, window_size, logger)
    self.model = np.load(theta_file)


  def process_window(self, window):

    window = window.flatten()
    window = np.insert(window, 0, 1)
    window = window.reshape(-1, window.shape[0])

    probabilities = softmax(np.dot(window, self.model))

    return np.argmax(probabilities), np.max(probabilities)


class KNNSlider(Slider):
  pass


class FCNSlider(Slider):

  def __init__(self, scene_files, scene_sizes, model_files, step_size, window_size, logger):
    super().__init__(scene_files, scene_sizes, step_size, window_size, logger)
    self.w0 = np.load(model_files[0])
    self.b0 = np.load(model_files[1])
    self.w1 = np.load(model_files[2])
    self.b1 = np.load(model_files[3])

  def process_window(self, window):

    window = window.flatten()
    window = window.reshape(-1, window.shape[0])
    z0 = window.dot(self.w0) + self.b0
    a0 = relu(z0)
    z1 = a0.dot(self.w1) + self.b1
    probabilities = softmax(z1)

    return np.argmax(probabilities), np.max(probabilities)


class CNNSlider(Slider):
  def __init__(self, scene_files, scene_sizes, model_files, step_size, window_size, logger):
    super().__init__(scene_files, scene_sizes, step_size, window_size, logger)
    self.saver = tf.train.import_meta_graph('cnn_64_256_none_v2' + '.meta')
    self.sess = tf.Session()
    self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
    self.graph = tf.get_default_graph()


    # [n.name for n in tf.get_default_graph().as_graph_def().node]
    self.tf_keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
    self.tf_X = self.graph.get_tensor_by_name('X_batch:0')
    self.tf_y = self.graph.get_tensor_by_name('y_batch:0')
    self.tf_loss = self.graph.get_tensor_by_name('loss:0')
    self.tf_accuracy = self.graph.get_tensor_by_name('accuracy:0')
    self.tf_y_pred = self.graph.get_tensor_by_name('predictions:0')
    self.tf_probs = self.graph.get_tensor_by_name('Softmax:0')


  def process_window(self, window):

    #window = window.flatten()
    #window = window.reshape(-1, window.shape[0])
    window = window.reshape(-1, self.window_size[0],self.window_size[1],1)
    y_pred, probs = self.sess.run([self.tf_y_pred, self.tf_probs], feed_dict = {
                              self.tf_X: window, self.tf_keep_prob: 1})

    return y_pred, np.max(probs)

if __name__=='__main__':
  print("Library module. No main function")

