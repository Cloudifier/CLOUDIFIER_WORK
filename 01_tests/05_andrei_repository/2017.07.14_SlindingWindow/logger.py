from datetime import datetime as dt
import os

class Logger():

  def __init__(self, log_folder = "tmp", show = False, verbosity_level = 0):

    print(self.get_time() + " Initialize the logger")
    self.log_folder = log_folder
    self.log_file = None
    self.show = show
    self.verbosity_level = verbosity_level

    if not os.path.exists(log_folder):
      os.makedirs(log_folder)

    print(self.get_time() + " Create log folder {}".format(log_folder))
    self.create_file()

  def get_time(self):
    return dt.strftime(dt.now(), '%Y.%m.%d-%H:%M:%S')

  def create_file(self):

    time_prefix = dt.strftime(dt.now(), '%Y-%m-%d_%H_%M_%S')

    i = 0
    while True:
      log_path = os.path.join(self.log_folder, time_prefix + "_" + str(i))
      i += 1

      if not os.path.exists(log_path):
        self.log_file = open(log_path, 'w+')
        print(self.get_time() + " Create log file {}\n".format(log_path))
        break

  def change_show(self, show):
    self.show = show

  def log(self, str_to_log, show = None, tabs = 0, verbosity_level = 0):

    if show is None:
      show = self.show

    if verbosity_level < self.verbosity_level:
      show = False

    time_prefix = dt.strftime(dt.now(), '[%Y.%m.%d-%H:%M:%S] ')

    if show:
      print(time_prefix + tabs * '\t' + str_to_log, flush=True)

    self.log_file.write(time_prefix + str_to_log + '\n')
