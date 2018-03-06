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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
   

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







    
