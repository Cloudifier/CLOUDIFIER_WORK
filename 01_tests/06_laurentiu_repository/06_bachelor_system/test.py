import tensorflow as tf
import numpy as np

rs = np.random.RandomState(seed=23)

np_var1 = np.zeros(shape=[16380*16380,2]).astype(np.float32)
graph = tf.Graph()

with graph.as_default(), tf.device('/gpu:0'):
  tf_w = tf.Variable(np_var1, dtype=tf.float32, name='w')
  tf_w0 = tf.Variable(np_var1, dtype=tf.float32, name='w1')
  tf_w2 = tf.Variable(np_var1, dtype=tf.float32, name='w2')
  init= tf.global_variables_initializer()


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options,
                                                     log_device_placement=True))


sess.run(init, options=run_options)