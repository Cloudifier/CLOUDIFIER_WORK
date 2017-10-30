import numpy as np
import pandas as pd
from random import randint

class SlidingWindowPreprocessor:
  def __init__(self, test_X, test_y):
    self.test_X = np.array(test_X)
    self.test_y = np.array(test_y)

  def create_new_img(self, big_dim, small_dim, small_img):
    small_img = small_img.reshape(small_dim[0], small_dim[1])
    pos_X = randint(0, big_dim[1] - small_dim[1])
    pos_Y = randint(0, big_dim[0] - small_dim[0])
    position = (pos_X, pos_Y)

    big_img = np.zeros((big_dim[0], big_dim[1]))
    big_img[pos_Y:pos_Y+small_dim[1], pos_X:pos_X+small_dim[0]] = small_img
    return big_img.ravel(), position

  def create_nparrays(self):
    self.test1_X = np.zeros((2450,5000), dtype=int)
    self.pos1 = list()
    for i in range(2450):
      if i % 500 == 0:
        print("Iteration #{}".format(i))
      big_dim = (100, 50)
      big_img, position = self.create_new_img(big_dim, (28,28), self.test_X[i])
      self.test1_X[i] = big_img
      self.pos1.append(position)
    print("Done computing test1_X")

    self.test2_X = np.zeros((2450,2000), dtype=int)
    self.pos2 = list()
    for i in range(2450):
      if i % 500 == 0:
        print("Iteration #{}".format(i))
      big_dim = (50, 40)
      big_img, position = self.create_new_img(big_dim, (28,28), self.test_X[i+2450])
      self.test2_X[i] = big_img
      self.pos2.append(position)
    print("Done computing test2_X")

    self.test3_X = np.zeros((2450,70000), dtype=int)
    self.pos3 = list()
    for i in range(2450):
      if i % 500 == 0:
        print("Iteration #{}".format(i))
      big_dim = (200, 350)
      big_img, position = self.create_new_img(big_dim, (28,28), self.test_X[i+4900])
      self.test3_X[i] = big_img
      self.pos3.append(position)
    print("Done computing test3_X")

    self.test4_X = np.zeros((2450,3200), dtype=int)
    self.pos4 = list()
    for i in range(2450):
      if i % 500 == 0:
        print("Iteration #{}".format(i))
      big_dim = (40, 80)
      big_img, position = self.create_new_img(big_dim, (28,28), self.test_X[i+7350])
      self.test4_X[i] = big_img
      self.pos4.append(position)
    print("Done computing test4_X")

  def create_dataframes(self):
    labels = ["pixel_" + str(i) for i in range(5000)]
    self.df1 = pd.DataFrame(self.test1_X, columns = labels)
    self.df1['Digit_label'] = self.test_y[:2450]
    list1, list2 = zip(*self.pos1)
    self.df1['Position_x'] = np.array(list1)
    self.df1['Position_y'] = np.array(list2)
    cols = ['Digit_label', 'Position_x', 'Position_y']  +\
      [col for col in self.df1 if (col != 'Digit_label') and (col != 'Position_x') and (col != 'Position_y')]
    self.df1 = self.df1[cols]

    labels = ["pixel_" + str(i) for i in range(2000)]
    self.df2 = pd.DataFrame(self.test2_X, columns = labels)
    self.df2['Digit_label'] = self.test_y[2450:4900]
    list1, list2 = zip(*self.pos2)
    self.df2['Position_x'] = np.array(list1)
    self.df2['Position_y'] = np.array(list2)
    cols = ['Digit_label', 'Position_x', 'Position_y']  +\
      [col for col in self.df2 if (col != 'Digit_label') and (col != 'Position_x') and (col != 'Position_y')]
    self.df2 = self.df2[cols]

    labels = ["pixel_" + str(i) for i in range(70000)]
    self.df3 = pd.DataFrame(self.test3_X, columns = labels)
    self.df3['Digit_label'] = self.test_y[4900:7350]
    list1, list2 = zip(*self.pos3)
    self.df3['Position_x'] = np.array(list1)
    self.df3['Position_y'] = np.array(list2)
    cols = ['Digit_label', 'Position_x', 'Position_y']  +\
      [col for col in self.df3 if (col != 'Digit_label') and (col != 'Position_x') and (col != 'Position_y')]
    self.df3 = self.df3[cols]

    labels = ["pixel_" + str(i) for i in range(3200)]
    self.df4 = pd.DataFrame(self.test4_X, columns = labels)
    self.df4['Digit_label'] = self.test_y[7350:9800]
    list1, list2 = zip(*self.pos4)
    self.df4['Position_x'] = np.array(list1)
    self.df4['Position_y'] = np.array(list2)
    cols = ['Digit_label', 'Position_x', 'Position_y']  +\
      [col for col in self.df4 if (col != 'Digit_label') and (col != 'Position_x') and (col != 'Position_y')]
    self.df4 = self.df4[cols]

