# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 12:28:54 2018

@author: Mihai.Cristea
"""
import CNN
from CNN import accuracy, minmaxNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('mnist-original')
X = mnist.data
y = mnist.target.astype(int)
#y = y.reshape(y.shape[0],1)
label_to_oh=lambda x: np.eye(10)[x]
y = label_to_oh(y)

m, n = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

#print (y_train)
#input()

nn = CNN.NeuralNetwork(X.shape[1])   
nn.addLayer(64, 'relu') 
nn.addLayer(32, 'relu')
nn.addLayer(10, 'softmax')


#nn.fit(X_train, y_train, epochs=6, batchSize=16, alpha=0.03, lmbd=0)
nn.fit(X_train, y_train, batch_dim=16, epochs=100, learning_rate=0.01, lmbd=0.01, cost_function_type = 'cross_entropySoftmax')




#
#learningStatus = nn.getLearningStatus()
#yhat = nn.predict(X_test)
#m, n = yhat.shape
#predict = (np.argmax(yhat, axis=1))
#print("Test acc: {:.2f}%".format(np.mean((predict == y_test) * 1) * 100))
#
#yhat_train = nn.predict(X_train)
#print("Train acc: {:.2f}%".format(100 * ((np.argmax(yhat_train, axis=1) == y_train.ravel()).sum() / y_train.shape[0])))
#
##plot cost
#x_pos = [idx for idx, x in enumerate(learningStatus)]
#y_pos = [x.error for idx, x in enumerate(learningStatus)]
#plt.plot(x_pos, y_pos, c='red')
## plt.show()
#
#x_pos = [idx for idx, x in enumerate(learningStatus)]
#y_pos = [x.accuracy for idx, x in enumerate(learningStatus)]
#plt.plot(x_pos, y_pos, c='blue')
## plt.show()






yhat = nn.predict(X_test)
print('Train accuracy: ' + str(nn.acc[-1]))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))
plt.plot(range(len(nn.Elist)),nn.Elist )
plt.show()



#http://danielfrg.com/blog/2013/07/27/not-so-basic-neural-network-python/



    
