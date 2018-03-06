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


titanic_df = pd.read_excel('titanic3.xls')
#titanic_df = pd.read_csv('titanic3.csv')

titanic_df['title'] = titanic_df['name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
titanic_df['age_mean'] = titanic_df.groupby('title')['age'].transform('mean').round(0)
titanic_df['age'].fillna(titanic_df['age_mean'], inplace=True)
titanic_df.drop('age_mean', axis=1, inplace=True)
titanic_df['fare_01'] = titanic_df.groupby(['pclass','title','age'])['fare'].transform('mean')
titanic_df['fare'].fillna(titanic_df['fare_01'], inplace=True)
titanic_df.drop('fare_01', axis=1, inplace=True)

titanic_df.drop('name', axis=1, inplace=True)
titanic_df.drop('embarked', axis=1, inplace=True)
titanic_df.drop('cabin', axis=1, inplace=True)
titanic_df.drop('home.dest', axis=1, inplace=True)
titanic_df.drop('ticket', axis=1, inplace=True)
titanic_df.drop('title', axis=1, inplace=True)
titanic_df.drop('body', axis=1, inplace=True)

titanic_df = titanic_df.fillna(0)
titanic_df.loc[titanic_df.boat != 0, 'boat'] = 1
titanic_df.loc[titanic_df.sibsp != 0, 'sibsp'] = 1
titanic_df.loc[titanic_df.parch != 0, 'parch'] = 1
titanic_df = pd.get_dummies(titanic_df, columns=["pclass","sex"], prefix=["class", "sex"])
#print(titanic_df.isnull().values.any())
#input()
y = np.array(titanic_df.iloc[0:,0])
y = y.reshape(y.shape[0],1)
X = np.array(titanic_df.iloc[:,1:])
X = X.astype(float)
m = X.shape[0]
X = minmaxNorm(X)

#X = normalize(X)
#X = np.c_[np.ones((m,1)),X]
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)
   

nn = CNN.NeuralNetwork(X.shape[1])   
nn.addLayer(6, 'relu') 
nn.addLayer(3, 'relu')
nn.addLayer(1, 'sigmoid')



### X shape (n,m) - m = sample numbers in rows, n = features number
#nn.fit(X_train.T, y_train.T, 8, 1000, 0.05,lmbd=0.01)
#nn.fit(X_train, y_train, 8, 2000, 0.01, 0.01)
nn.fit(X_train, y_train, 8, 300, 0.01, 0.01,'cross_entropy')



yhat = nn.predict(X_test)
print('Train accuracy: ' + str(nn.acc[-1]))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))
plt.plot(range(len(nn.Elist)),nn.Elist )
plt.show()


### K-NN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.ravel())
yhat = knn.predict(X_test)
yhat = yhat.reshape(yhat.shape[0],1)
print ("accuracy : " + str(accuracy(yhat, y_test)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, yhat)


### K-NN perform 10-fold cross validation
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X_train, y_train.ravel(), cv=10, scoring='accuracy')
score = np.mean(scores)
print('score k = 3:', score)

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbor = filter(lambda x: x % 2 != 0, myList)
neighbors = list(neighbor)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train.ravel(), cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = MSE.index(min(MSE))
optimal_k = neighbors[optimal_k]

optimal_acc = max(cv_scores)

print ("The optimal number of neighbors is %d" % optimal_k)
print ("Accuracy for the optimal number of neighbors is :",  optimal_acc)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()





    
