# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:18:40 2017

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

def normalize(X):
    return (X - X.mean()) / np.std(X, axis = 1, keepdims = True)

activ = {
    'sigmoid': sigmoid,
    'relu':   relu,
    'sigmoidPrime': sigmoidPrime,
    'reluPrime': reluPrime,
}

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
#X = np.c_[np.ones((m,1)),X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#z=-1
#interest = activ['sigmoid'](z)
#print (interest)


class NeuralNetwork:
    
#    def __init__ (self, input_nodes, activ_f):
    def __init__ (self, input_nodes, activ_f):

        '''activation function (1-sigmoid, 2-relu, 0-none)'''
#        self.input_nodes = input_nodes
#        self.activ_f = activ_f
#        self.output_nodes = output_nodes
#        self.layer_type = layer.type

        self.layers = [input_nodes]
        self.theta = [] #= np.empty()
        self.activation = [activ_f]

            
    def addLayer(self,nodes_nr, activ_f):
        '''add new layer, specify numbers of nodes and activation function (1-sigmoid, 2-relu, 0-none) '''
        self.layers.append(nodes_nr)
        self.activation.append(activ_f)      
        self.theta.append(np.random.rand(self.layers[-2],self.layers[-1]))

    def removeLayer(self, index):
        del self.layers[index]
        del self.activation[index]
        self.initializeTheta()

    
    def initializeTheta(self):
            self.theta.append(np.random.rand(self.layers[i-1],self.layers[i]))
    
    def fit(self, X, y, epochs, batch_dim):
        '''specify predict activation function (sigmoid, relu, none) '''
        theta = np.array(self.theta)
        layers = np.array(self.layers)
        activation = np.array(self.activation)
        print(X.shape)
        input
        m, n = X.shape
        X = np.c_[np.ones((m,1)),X]
        m, n = X.shape
        X = normalize(X)
        

#        batch_dim=8
#        epochs = 300
        Elist= list()
        steps = epochs // 10 
        for epoch in range (epochs):
            for j in range(int(m / batch_dim)):
                begin = j * batch_dim
                end = m if ((j + 1) * batch_dim  > m) else (j + 1) * batch_dim
                X_batch = X[begin : end]
                y_batch = y[begin : end]
#                z = np.array(X_batch.shape[0],layers.shape)
#                a = np.array(X_batch.shape[0],layers.shape)
#                delta = np.array(X_batch.shape[0],layers.shape)
#                grd = np.array(X_batch.shape[0],layers.shape)
                z = dict()
                a = dict()
                print('X_batch: \n', X_batch.shape, '\n', 'theta: \n', theta[0].shape, '\n', 'z: \n', z.shape  )
                input()
                #z(0) = np.dot(X_batch,theta[0])
                print(z = np.dot(X_batch,theta[0]))
                a(0)= activ[activation[0]](z)
                for k in range(0,layers.shape[0]):
#                    print('X \n', X_batch.shape)
#                    print('Theta : \n', theta[k].shape)
#                    print('k : \n', k)
#                    input()
                    z[k] = a[k-1].dot(theta[k])
                    a[k]= activ[activation[k]](z[k])
#
                yhat = a[-1]
#back propagation
#                delta[self.layers.shape[0]-1] = np.mean(yhat - y_batch)
#                for b in  range(self.layers.shape[0]-2, -1, -1 ):
#                    delta[b] = delta[b-1].dot(theta[b].T) * activ[self.activation[b](z[b])
#                    #grd[b] = (1/batch_dim) * a[b].T.dot(delta[b])
#                
#                grd3 = 1/batch_dim * a2.T.dot(delta3)
#                delta2 = delta3.dot(theta3.T) * reluPrime(z2)
#                grd2 = 1/batch_dim * a1.T.dot(delta2)
#                delta1 = delta2.dot(theta2.T) * reluPrime(z1)
#                grd1 = 1/batch_dim * X_batch.T.dot(delta1)
#            
#                theta1 = theta1 - alpha * grd1
#                theta2 = theta2 - alpha * grd2
#                theta3 = theta3 - alpha * grd3   
#    
#        yhat = predict(X_train, theta1, theta2, theta3)
#        E = np.mean(-(y_train * np.log(yhat)+ (1-y_train) * np.log (1-yhat)))
#        acc = accuracy(yhat, y_train)
#        if epoch % steps == 0 :
#            print(f'#{epoch} epochs. Train accuracy: {acc}%, E: {E}')
#        Elist.append(E)  
#    return z1, a1, z2, a2, z3, a3, yhat


#    
#    def fit(self):
        
        
nn = NeuralNetwork()
nn.addLayer(11,'sigmoid')  
nn.addLayer(6,'sigmoid')   
nn.addLayer(3,'sigmoid')
nn.addLayer(1,'sigmoid') 

for i in range (3,0,-1):
    print(i)
    
y = np.array(titanic_df.iloc[0:,0])
y = y.reshape(y.shape[0],1)
X = np.array(titanic_df.iloc[:,1:])
X = X.astype(float)

nn.fit(X_train, y_train, 300, 8)

print(np.percentile(X,10,1))