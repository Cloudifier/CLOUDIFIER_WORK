import tensorflow as tf
import numpy as np
from threading import Thread, Lock
from time import sleep, time
from datetime import datetime as dt



class ThreadedTFSession(Thread):
  def __init__(self, name, threadID=-1, threadLock=None):
    self.size = 10
    self.threadLock = threadLock
    Thread.__init__(self)

    if threadID == -1:
      threadID = str(time()).replace(".","")
    self.threadID = str(threadID)
    
    self.name = name + '_' + self.threadID
    
    self.run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    #with tf.device("/cpu:0"):
    self.create_graph()
    self.create_session()
    
    return
  
  def log(self, _str):
    if self.threadLock is not None:
      self.threadLock.acquire()
    
    print(_str, flush=True)
    
    if self.threadLock is not None:
      self.threadLock.release()
    
    return


  def create_graph(self):
    print("Creating graph {} .. ".format(self.name))
    rs = np.random.RandomState(seed=23)
    np_var1 = rs.normal(size=[1,self.size]).astype(np.float32)
    np_var2 = rs.normal(size=[self.size,self.size]).astype(np.float32)
    np_var3 = rs.normal(size=[self.size,1]).astype(np.float32)


    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.variable_scope(self.name):
        self.tf_input = tf.placeholder(tf.float32, shape=[1,1], name = "input")

        #for i in range(4):
        tf_w1 = tf.Variable(tf.truncated_normal(shape=(268435456,2)), dtype=tf.float32, name='w1')
        tf_w2 = tf.Variable(tf.truncated_normal(shape=(268435456,1)), dtype=tf.float32, name='w2')
        tf_w3 = tf.Variable(tf.truncated_normal(shape=(268435456,3)), dtype=tf.float32, name='w3')
        
        #tf_w = tf.Variable(tf.truncated_normal(shape=(16384,16384)), dtype=tf.float32, name='w')


        matrix1 = tf.Variable(np_var1, name = 'matrix1')
        matrix2 = tf.Variable(np_var2, name = 'matrix2')
        matrix4 = tf.Variable(np_var3, name = 'matrix4')

        

        matmul1 = tf.matmul(self.tf_input, matrix1, name = 'matmul1')
        matmul2 = tf.matmul(matmul1, matrix2, name = 'matmul2')
        self.matmul4 = tf.matmul(matmul2, matrix4, name = "matmul4")
        self.tf_init = tf.global_variables_initializer()
        
        #self.mem_stats_max_bytes = tf.contrib.memory_stats.MaxBytesInUse()
        #self.mem_stats_use_bytes = tf.contrib.memory_stats.BytesInUse()
    
    #l = [n.name for n in self.graph.as_graph_def().node]
    #tensors = [self.graph.get_tensor_by_name(n+':0') for n in l]
    #print(tensors)
    return

  
  def create_session(self):
    self.log("Creating session {} ..".format(self.name))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options,
                                                                      #device_count = {'GPU' : 0},
                                                                      #allow_soft_placement=True,
                                                                      log_device_placement=True))

  def run(self):
    self.log("Starting " + self.name)
    self.session.run(self.tf_init, options=self.run_options)

    for i in range(5):
      v = self.session.run(self.matmul4, feed_dict={self.tf_input: np.ones((1,1))}, options=self.run_options)
      strnowtime = dt.now().strftime("[%Y-%m-%d %H:%M:%S] ")
      sleep(0.1)
      self.log(strnowtime + self.name + "  " + str(v.sum()))

    return


class ThreadBasic(Thread):
  def __init__(self, threadLock, threadID=1):
    Thread.__init__(self)
    self.threadLock = threadLock
    self.threadID = threadID
    return
  
  def log(self, _str):
    if self.threadLock is not None:
      self.threadLock.acquire()
    
    print(_str, flush=True)
    
    if self.threadLock is not None:
      self.threadLock.release()
    
    return
  
  def run(self):
    l = np.arange(0, 1000)
    for i in l:
      self.log("Thread {} - {}".format(self.threadID, i))
    return

if __name__ == '__main__':
  thrLock = Lock()
  threads = []

  threads.append(ThreadedTFSession(name='a', threadID=1, threadLock=thrLock))
  #threads.append(ThreadedTFSession(name='b', threadID=2, threadLock=thrLock))   
  #threads.append(ThreadedTFSession(name='c', threadID=3, threadLock=thrLock))
  
  

  """
  thread_1 = ThreadBasic(thrLock, 1)
  thread_2 = ThreadBasic(thrLock, 2)
  """
  for th in threads:
    th.start()
    
  for th in threads:
    th.join()
  
  
"""
tf.train.write_graph(t.session.graph_def, 'aici', 'train.pb', False)
with tf.gfile.GFile('aici/train.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
graph_def
with tf.Graph().as_default() as grphȘ
with tf.Graph().as_default() as grph:
    tf.import_graph_def(graph_def, name='')
"""