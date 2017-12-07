# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:28:21 2017

@author: Mihai.Cristea
"""
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidPrime(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g

def relu(z):
    return np.maximum(0, z)

def reluPrime(z):
    return (relu(z) > 0).astype(int)

def accuracy(yhat, y):
    m, n = y.shape
    acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
    return round(acc * 100, 2)

def predict(X,theta1,theta2,theta3):
    z1 = X.dot(theta1)
    a1 = relu(z1)

    z2 = a1.dot(theta2)
    a2 = relu(z2)

    z3 = a2.dot(theta3)
    a3 = sigmoid(z3)
    yhat = a3

    return yhat
#    return z1, a1, z2, a2, z3, a3, yhat

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())
    

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

titanic_df = pd.read_excel('titanic3.xls')
#titanic_df = pd.read_csv('titanic3.csv')

#print(titanic_df[(titanic_df['name'].str.contains('miss')) & (titanic_df['age'].notnull())] )
#a = titanic_df[titanic_df['name'].str.contains('Miss')]
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

#titanic_df['boat'].fillna(0, inplace=True)
titanic_df = titanic_df.fillna(0)
titanic_df.loc[titanic_df.boat != 0, 'boat'] = 1
titanic_df.loc[titanic_df.sibsp != 0, 'sibsp'] = 1
titanic_df.loc[titanic_df.parch != 0, 'parch'] = 1
#for i in range(10,80,10):
#    titanic_df.loc[(titanic_df.age > i-10) & (titanic_df.age <= i), 'age'] = i
#for i in range(50,550,50):
#    titanic_df.loc[(titanic_df.fare > i-50) & (titanic_df.fare <= i), 'fare'] = i
#titanic_df = pd.get_dummies(titanic_df, columns=["pclass","sex","age","fare"], prefix=["class", "sex","age","fare"])
titanic_df = pd.get_dummies(titanic_df, columns=["pclass","sex"], prefix=["class", "sex"])

y = np.array(titanic_df.iloc[0:,0])
y = y.reshape(y.shape[0],1)
X = np.array(titanic_df.iloc[:,1:])
X = X.astype(float)
m = X.shape[0]
X = normalize(X)
X = np.c_[np.ones((m,1)),X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



#X_train = np.c_[np.ones((m,1)),X_train]
m, n = X_train.shape
alpha = 0.03
epochs = 3500
batch_dim = 8
#mse bias
max_cross_bias_relu = []
i = 0
print("Training", flush=True)

theta1 = np.random.rand(n,6)
theta2 = np.random.rand(6,3)
theta3 = np.random.rand(3,1)

Elist= list()
steps = epochs // 10 
for epoch in range (epochs):
    for j in range(int(m / batch_dim)):
        begin = j * batch_dim
        end = m if ((j + 1) * batch_dim  > m) else (j + 1) * batch_dim
        X_batch = X_train[begin : end]
        y_batch = y_train[begin : end]   

        z1 = X_batch.dot(theta1)
        a1 = relu(z1)
        z2 = a1.dot(theta2)
        a2 = relu(z2)
        z3 = a2.dot(theta3)
        a3 = sigmoid(z3)
        
        yhat = a3
        
        delta3 = yhat - y_batch
        #delta3 = np.mean(yhat - y_batch)
        grd3 = 1/batch_dim * a2.T.dot(delta3)
        delta2 = delta3.dot(theta3.T) * reluPrime(z2)
        grd2 = 1/batch_dim * a1.T.dot(delta2)
        delta1 = delta2.dot(theta2.T) * reluPrime(z1)
        grd1 = 1/batch_dim * X_batch.T.dot(delta1)
#        print(delta3.shape)
#        print(delta2.shape)
#        print(delta1.shape)
#        input()
    
        theta1 = theta1 - alpha * grd1
        theta2 = theta2 - alpha * grd2
        theta3 = theta3 - alpha * grd3   
    
    yhat = predict(X_train, theta1, theta2, theta3)
    E = np.mean(-(y_train * np.log(yhat)+ (1-y_train) * np.log (1-yhat)))
    acc = accuracy(yhat, y_train)
    if epoch % steps == 0 :
        print(f'#{epoch} epochs. Train accuracy: {acc}%, E: {E}')
    Elist.append(E)   

yhat = predict(X_test, theta1, theta2, theta3)
print('Train accuracy: ' + str(acc))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))
plt.plot(range(len(Elist)),Elist )
plt.show()