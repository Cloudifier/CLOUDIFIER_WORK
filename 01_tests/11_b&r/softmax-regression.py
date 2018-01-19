import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np


import tensorflow as tf

def softmax(z):
  z = z.copy()
  z -= z.max(axis = 1, keepdims = True)
  return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims = True)


def accuracy(y_hat, y_test):
  pred = np.argmax(y_hat, axis = 1) == np.argmax(y_test, axis = 1)
  return (np.sum(pred) / y_hat.shape[0]) * 100
  
  

mnist = fetch_mldata('mnist-original')
X = mnist.data
y = mnist.target
n_classes = np.unique(y).shape[0]
Y = np.zeros((y.shape[0], n_classes))
for i in range(y.shape[0]):
    Y[i, int(y[i])] = 1

m, n = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state = 1234)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

m_train, n_features = X_train.shape

batch_size = 512
epochs = 3
alpha = 0.05
lmbd = 0.05

if False:
  g = tf.Graph()
  with g.as_default():
    tf_X_input = tf.placeholder(shape = (None, 784), dtype = tf.float32, name = "X_input")
    tf_Y_input = tf.placeholder(shape = (None, 10), dtype = tf.float32, name = "y_input")
    
    tf_theta = tf.Variable(initial_value=np.zeros((n_features, n_classes)), 
                           dtype = tf.float32,
                           name = "Theta")
    tf_batch_size = tf.Variable(initial_value = batch_size, name = "BS", dtype = tf.float32)
    
    tf_z = tf.matmul(tf_X_input, tf_theta, name = "Z")
    tf_yhat = tf.nn.softmax(tf_z, name = "SM")
    
    tf_loss_v1 = tf.divide( - tf.reduce_sum(tf_Y_input * tf.log(tf_yhat)) + lmbd / 2 * tf.reduce_sum(tf.pow(tf_theta, 2)),
                          tf_batch_size, name = "LOSS") 
    
    tf_loss_v2 = tf.losses.softmax_cross_entropy(onehot_labels = tf_Y_input, logits = tf_z) 
    tf_loss_v2 += lmbd * tf.nn.l2_loss(tf_theta)
    
    
    tf_g_v1 = tf.divide(tf.matmul(tf.transpose(tf_X_input), tf_yhat - tf_Y_input) + lmbd * tf_theta,
                        tf_batch_size, name = "Gv1")
    
    tf_g_v2 = tf.gradients(tf_loss_v1, tf_theta, name = "Gv2")[0]
  
    tf_g_v3 = tf.gradients(tf_loss_v2, tf_theta, name = "Gv3")[0]
    
    tf_update_v1 = tf.assign(tf_theta, tf_theta - alpha * tf_g_v1 , name = "UpdateOP_v1_grad_calc")
  
    tf_update_v2 = tf.assign(tf_theta, tf_theta - alpha * tf_g_v2 , name = "UpdateOP_v2_grad_loss")
  
    tf_update_v3 = tf.assign(tf_theta, tf_theta - alpha * tf_g_v3 , name = "UpdateOP_v3_grad_cros")
  
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha)
    tf_update_v4 = optimizer.minimize(tf_loss_v2, name = 'UpdateOP_v4_MinimizeAdam')
  
    
    tf_init1 = tf.global_variables_initializer()
  
  #tf_init2 = tf.global_variables_initializer()
  
  all_ops = [tf_update_v1, tf_update_v2, tf_update_v3, tf_update_v4]
  all_losses = [tf_loss_v1, tf_loss_v1, tf_loss_v2, tf_loss_v2]
  for i, opt_op in enumerate(all_ops):
    loss_tensor = all_losses[i]
    loss_list = []
    sess = tf.Session(graph = g)
    sess.run(tf_init1)
    print("Optimization with {} for {} epochs".format(opt_op.name, epochs))
    for epoch in range(epochs):
        n_iter = m_train // batch_size
        for i in range(n_iter):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            X_batch = X_train[batch_start: batch_end]
            y_batch = y_train[batch_start: batch_end]
            _, loss, theta_tf = sess.run([opt_op, loss_tensor, tf_theta], feed_dict = {
                tf_X_input : X_batch,
                tf_Y_input : y_batch})
            loss_list.append(loss)
        print(" Epoch {} loss {:.2f}".format(epoch, loss))
        acc_train_tf = accuracy(softmax(X_train.dot(theta_tf)),y_train)
        z_test_tf = X_test.dot(theta_tf)
        yhat_test_tf = softmax(z_test_tf)
        acc_test_tf = accuracy(yhat_test_tf, y_test)
        print("  Train accuracy {:.2f}%".format(acc_train_tf))
        print("  Test accuracy  {:.2f}% diff: {:.2f}".format(acc_test_tf, acc_train_tf-acc_test_tf))
    sess.close()

  
print("\n\nNP:")
  
  
#epochs *=3
loss_list = []
theta = np.zeros((n_features, n_classes))
for epoch in range(epochs):
    n_iter = m_train // batch_size
    for i in range(n_iter):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start: batch_end]
        y_batch = y_train[batch_start: batch_end]
        z = X_batch.dot(theta)
        yhat = softmax(z)
        loss = - np.sum(y_batch * np.log(yhat)) / batch_size
        loss_list.append(loss)
        D_loss_wrt_z = yhat - y_batch # aoleo e matrice !
        D_loss_wrt_theta = (X_batch.T.dot(D_loss_wrt_z) + lmbd * theta) / batch_size
        theta = theta - alpha * D_loss_wrt_theta
    print("Epoch {} loss {:.2f}".format(epoch, loss))
    acc_train = accuracy(softmax(X_train.dot(theta)),y_train)
    print("Train accuracy {:.2f}%".format(acc_train))
    z_test = X_test.dot(theta)
    yhat_test = softmax(z_test)
    acc_test = accuracy(yhat_test, y_test)
    print("Test accuracy  {:.2f}% diff: {:.2f}".format(acc_test, acc_train-acc_test))

plt.plot(range(len(loss_list)), loss_list)
plt.show()
  
  
  


