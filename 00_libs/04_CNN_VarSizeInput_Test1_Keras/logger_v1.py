# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:21:30 2017

@author:  High Tech Systems and Software

@project: Cloudifier.NET

@module:  Utility

@description: 
    utility module

@copyright: CLOUDIFIER SRL

"""
from datetime import datetime as dt
import matplotlib.pyplot as plt
import sys
import os
import socket

from scipy.misc import imsave

from io import TextIOWrapper, BytesIO

class Logger:
  def __init__(self, lib_name = "LOGR", base_folder = "", SHOW_TIME = True):
    self.app_log = list()
    self.results = list()
    self.printed = list()
    self.MACHINE_NAME = self.GetMachineName()
    self.__version__ = "1.0"
    self.__author__ = "HTSS"
    self.__project__ = "CLOUDIFIER.NET"
    self.SHOW_TIME = SHOW_TIME
    self.file_prefix = dt.now().strftime("%Y%m%d_%H%M%S") 
    self.log_file = self.file_prefix + '_log.txt'
    self.log_results_file = self.file_prefix + "_RESULTS.txt"
    self.__lib__= lib_name
    self._base_folder  = base_folder

    self._logs_dir = os.path.join(self._base_folder,"_logs")
    self._outp_dir = os.path.join(self._base_folder,"_output")

    self._setup_folders([self._outp_dir,self._logs_dir])
    
    self.log_file = os.path.join(self._logs_dir, self.log_file)
    self.log_results_file = os.path.join(self._logs_dir, self.log_results_file)
    
    self.VerboseLog("Library [{}] initialized on machine [{}]".format(
                    self.__lib__, self.MACHINE_NAME))
    
    return

  def _setup_folders(self,folder_list):
    for folder in folder_list:
      if not os.path.isdir(folder):
        print("Creating folder [{}]".format(folder))
        os.makedirs(folder)
    return

  def ShowNotPrinted(self):
    nr_log = len(self.app_log)
    for i in range(nr_log):
      if not self.printed[i]:
        print(self.app_log[i], flush = True)
        self.printed[i] = True
    return
  
  def ShowResults(self):
    for res in self.results:
      self._logger(res, show = True, noprefix = True)
    return
  
  def _logger(self, logstr, show = True, results = False, noprefix = False):
    """ 
    log processing method 
    """
    nowtime = dt.now()
    prefix = ""
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(self.__lib__))
    if self.SHOW_TIME and (not noprefix):
      prefix = strnowtime
    if logstr[0]=="\n":
      logstr = logstr[1:]
      prefix = "\n"+prefix
    logstr = prefix + logstr
    self.app_log.append(logstr)
    if show:
      print(logstr, flush = True)
      self.printed.append(True)
    else:
      self.printed.append(False)    
    if results:
      self.results.append(logstr)
    try:
      log_output = open(self.log_file, 'w')
      for log_item in self.app_log:
        log_output.write("%s\n" % log_item)
      log_output.close()
    except:
      print(strnowtime+"Log write error !", flush = True)
    return

  def OutputImage(self,arr, label=''):
    """
    saves array to a file as image
    """
    label = label.replace(">","_")
    file_prefix = dt.now().strftime("%Y%m%d_%H%M%S_") 
    file_name = os.path.join(self._outp_dir,file_prefix+label+".png")
    self.Log("Saving figure [{}]".format(file_name))
    if os.path.isfile(file_name):
      self.Log("Aborting image saving. File already exists.")
    else:
      imsave(file_name, arr)
    return

  def OutputPyplotImage(self, label=''):
    """
    saves current figure to a file
    """
    file_prefix = dt.now().strftime("%Y%m%d_%H%M%S") 
    file_name = os.path.join(self._outp_dir,file_prefix+label+".png")
    self.Log("Saving figure [{}]".format(file_name))
    plt.savefig(file_name)
  
  def VerboseLog(self,str_msg, results = False):
    self._logger(str_msg, show = True, results = results)
    return
  
  def Log(self,str_msg, show = False, results = False):
    self._logger(str_msg, show = show, results = results)
    return
    
  def GetKerasModelSummary(self, model, full_info = False):
    if not full_info:
      # setup the environment
      old_stdout = sys.stdout
      sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)      
      # write to stdout or stdout.buffer
      model.summary()      
      # get output
      sys.stdout.seek(0)      # jump to the start
      out = sys.stdout.read() # read output      
      # restore stdout
      sys.stdout.close()
      sys.stdout = old_stdout
    else:
      out = model.to_yaml()
    
    str_result = "Keras Neural Network Layout\n"+out
    return str_result    
  
  def GetMachineName(self):
    if socket.gethostname().find('.')>=0:
        name=socket.gethostname()
    else:
        name=socket.gethostbyaddr(socket.gethostname())[0]
    self.MACHINE_NAME = name
    return name
