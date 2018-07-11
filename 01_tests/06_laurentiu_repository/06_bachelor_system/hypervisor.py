from logger_helper import LoadLogger
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.python.client import device_lib
import psutil

valid_devices = ['gpu', 'cpu']

class Hypervisor:
  def __init__(self):
    self.logger = LoadLogger(lib_name='Bachelor', config_file='config.txt')
    self._log("Initializing tensor graph allocation&optimization system ...")
    self.div_gigabytes = 1024 * 1024 * 1024
    self.GPUMemLimit = dict()
    self.UsedGPUMem = dict()

    self._collect_environment_info()
    
    self._log("DAG allocator&optimizer initialized.", show_time=True)
    return


  def _log(self, str_log, show_time=False):
    self.logger.VerboseLog(str_log, show_time=show_time)

  def _check_path(self, path):
    if not os.path.exists(path):
      return False
    return True    

  def _collect_environment_info(self):
    local_device_protos = device_lib.list_local_devices()
    cpus = [x for x in local_device_protos if x.device_type == 'CPU']
    gpus = [x for x in local_device_protos if x.device_type == 'GPU']
    gpu_names = [x.name for x in gpus]

    self._log(" Found {} available CPU device(s): {}."
              .format(len(cpus), [x.name for x in cpus]))
    vmem_stats = psutil.virtual_memory()
    total_vmem = vmem_stats.total
    available_vmem = vmem_stats.available
    percent_vmem = vmem_stats.percent
    used_vmem = vmem_stats.used
    
    self._log(" System memory stats:")
    self._log(" Total memory: {:.2f}GB".format(total_vmem / self.div_gigabytes))
    self._log(" Available memory: {:.2f}GB [{}%]"
              .format(available_vmem / self.div_gigabytes, 100.0 - percent_vmem))
    self._log(" Used memory: {:.2f}GB [{}%]"
              .format(used_vmem / self.div_gigabytes, percent_vmem))
    
    
    self._log(" Found {} available GPU device(s): {}."
              .format(len(gpus), gpu_names))

    self._log(" GPU device(s) description:")
    for x in gpus:
      self.GPUMemLimit[x.name] = x.memory_limit
      self.UsedGPUMem[x.name] = 0
      try:
        self._log("  " + x.physical_device_desc)
      except:
        pass
    #endfor
    
    self._log(" GPU memory limit:")
    for dev,mem_lim in self.GPUMemLimit.items():
      self._log("  " + dev + ' - ' + '{:.2f}GB'.format(mem_lim / self.div_gigabytes))
    
    return
    

  def _check_memory_requirement(self, graph_def, device):
    pass
  
  def CreateJob(self, graph, device):
    if device not in valid_devices:
      raise Exception("ERROR! Device '{}' not recognized. Please try one of these: {}."
                      .format(device, valid_devices))

    self._log("Creating job on device '{}' ...".format(device))

    if type(graph) is str:
      if not self._check_path(graph):
        raise Exception("ERROR! '{}' does not exist.".format(graph))

      with tf.gfile.GFile(graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
      self._log("Imported GraphDef from '{}'.".format(graph_def))
    elif type(graph_def) is tf.GraphDef:
      graph_def = graph
    else:
      raise Exception("ERROR! graph_def should represent path to a " +\
                      "ProtoBuffer file or should be a tf.GraphDef effectively.")
      
    
    


class Job:
  def __init__(self):
    pass