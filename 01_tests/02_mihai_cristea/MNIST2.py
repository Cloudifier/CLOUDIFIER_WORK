# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:46:09 2017

@author: Mihai.Cristea
"""

import numpy as np
#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=29)
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

print("1")
mnist = fetch_mldata('MNIST original')
print("2")
X = mnist.data.astype('float64')
#X = np.array(mnist.data)
y = mnist.target
#y_unic=np.unique(mnist.target)
#print (y)
#X2=np.reshape(X,(70000,28,28))
#print (mnist.data.shape)

#print (X[10382],y[10382])


#Splitting the dataset into the Training set and Test set

X /= 255

print("3")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30,random_state = 0)



#Feature Scaling

# Fitting logistic regression to the Training set
print("4")
classifier = LogisticRegression(random_state = 0)
print("5")
classifier.fit(X_train, y_train)

# Predicting the Test results
print("6")
y_pred = classifier.predict(X_test)
#print(y_pred)
print("Done")


# The coefficients
print('Coefficients: \n', classifier.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score r2: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()
#http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#http://connor-johnson.com/2014/02/18/linear-regression-with-python/
print('acuratete: ', 1-np.count_nonzero(y_test-y_pred)/y_test.shape[0])
print('acuratete bool: ', np.sum(y_pred==y_test)/len(y_test))
plt.imshow(X_test[7].reshape(28,28), cmap='gray')
plt.show()
plt.imshow(X_train[7].reshape(28,28), cmap='gray')
plt.show()
print(y_test[7])
print(y_train[7])

for i in range(0, 10):
    plt.imshow(classifier.coef_[i].reshape(28,28), cmap = 'gray')
    plt.show()





