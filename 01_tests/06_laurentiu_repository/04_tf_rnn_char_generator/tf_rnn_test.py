import tensorflow as tf
import numpy as np

class SimpleLogger:
  def __init__(self):
    return
  def P(self, _str):
    print(_str, flush=True)


def rnd_p2s(preds, i2c):
  tt = ''
  for p in preds:
    ch = i2c[np.random.choice(range(preds.shape[1]), p=p)]
    tt += ch
  return tt


if __name__ == '__main__':
  text_file = __file__
  logger = SimpleLogger()
  logger.P("Loading data file '{}'".format(text_file))
  data = open(text_file).read()
  
  seq_len = 15
  hidden_size = 64
  epochs = 2500
  sampling_size = 250
  full_step = True
  random_sampling = True
  
  start_text = "    while True:"[:seq_len]
  
  vocabulary = sorted(list(set(data)))
  vocabulary_size = len(vocabulary)
  chr_to_idx = {c:i for i,c in enumerate(vocabulary)}
  idx_to_chr = {i:c for i,c in enumerate(vocabulary)}
  oh_mat = np.eye(vocabulary_size)
  str_to_oh = lambda txt: oh_mat[[chr_to_idx[c] for c in txt]]
  pred_to_str = lambda pre: "".join(idx_to_chr[i] for i in np.argmax(pre, axis=1))
  
  
  test_txt = "Ana are mere"
  identity = pred_to_str(str_to_oh(test_txt))
  logger.P("Test: '{}' == '{}'".format(test_txt, identity))
  
  logger.P("Creating graph ...")
  g = tf.Graph()
  with g.as_default():
    tf_x_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocabulary_size], name='x_seq')
    tf_y_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocabulary_size], name='y_seq')
    tf_h_ini = tf.placeholder(dtype=tf.float32, shape=[1, hidden_size], name='h_init')
    
    tf_wxh = tf.Variable(np.random.randn(vocabulary_size, hidden_size) * 0.01, dtype=tf.float32)
    tf_why = tf.Variable(np.random.randn(hidden_size, vocabulary_size) * 0.01, dtype=tf.float32)
    tf_whh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01, dtype=tf.float32)
    tf_h_bias = tf.Variable(np.zeros((1, hidden_size)), dtype=tf.float32)
    tf_y_bias = tf.Variable(np.zeros((1, vocabulary_size)), dtype=tf.float32)
    
    tf_h = tf_h_ini
    output_list = []
    seq_list = tf.split(tf_x_seq, seq_len)
    for unroll_step, tf_x_input in enumerate(seq_list):
      tf_h = tf.add(tf.matmul(tf_x_input, tf_wxh) + tf.matmul(tf_h,tf_whh), tf_h_bias,
                    name='h_'+str(unroll_step))
      tf_h = tf.nn.tanh(tf_h)
      if unroll_step == 1:
        tf_h_second = tf_h
      
      tf_y = tf.add(tf.matmul(tf_h, tf_why), tf_y_bias, name='y_' + str(unroll_step))
      output_list.append(tf_y)
    
    tf_h_out = tf_h if full_step else tf_h_second
    
    tf_y_full_seq = tf.concat(output_list, axis=0)
    tf_y_preds = tf.nn.softmax(tf_y_full_seq)
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_y_seq,
        logits=tf_y_full_seq))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(tf_loss)
    tf_init_op = tf.global_variables_initializer()
    
  start_idx = 0
  sess = tf.Session(graph=g)
  sess.run(tf_init_op)
  step_size = seq_len if full_step else 1
  text_range = range(0, len(data) - 1) #step_size)
  np_h_init = np.zeros((1, hidden_size))


  checkpoints = (0.1 * np.arange(1,11) * epochs).astype(np.int16)
  for epoch in range(epochs):
    if epoch % 100 == 0:
      logger.P("Running epoch {} ..".format(epoch))
    h_start = np_h_init.copy()
    
    for start_idx in text_range:
      x_seq = data[start_idx : start_idx+seq_len]
      x_seq += ' ' * (seq_len - len(x_seq))
      
      y_seq = data[start_idx+1 : start_idx+seq_len+1]
      y_seq += ' ' * (seq_len - len(y_seq))
      
      x_seq_oh = str_to_oh(x_seq)
      y_seq_oh = str_to_oh(y_seq)

      feed_dict = {
          tf_x_seq: x_seq_oh,
          tf_y_seq: y_seq_oh,
          tf_h_ini: h_start
      }
      _, loss, h_start = sess.run([tf_train_op, tf_loss, tf_h_out], feed_dict=feed_dict)
    
    if epoch in checkpoints:
      logger.P("Predicting with start: '{}'".format(start_text))
      final_output = start_text
      input_text = start_text
      for s in range(sampling_size):
        x_test = str_to_oh(input_text)
        
        feed_dict = {
            tf_x_seq: x_test,
            tf_h_ini: h_start
        }
        preds, h_start = sess.run([tf_y_preds, tf_h_out], feed_dict=feed_dict)
        if random_sampling:
          out_text = rnd_p2s(preds, idx_to_chr)
        else:
          out_text = pred_to_str(preds)
        ch = out_text[-1]
        input_text = input_text[1:] + ch
        final_output += ch
      logger.P("Predicted:\n   {}".format(final_output))
