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

def predict(X,theta1,theta2,theta3):
    z1 = X.dot(theta1)
    a1 = relu(z1)

    z2 = a1.dot(theta2)
    a2 = relu(z2)

    z3 = a2.dot(theta3)
    a3 = sigmoid(z3)
    
    z4
    yhat = a3


activ = {
    'sigmoid': sigmoid,
    'relu':   relu,
    'sigmoidPrime': sigmoidPrime,
    'reluPrime': reluPrime,
}

class Layer:
    nextIdNum = 0 # next ID number to assign
    def __init__(self,  inputs, nodes,  activ_func):
        self.nodes = nodes
        self.inputs = inputs
        self.theta = np.random.rand(self.inputs, self.nodes)
        self.activ_f = activ_func
        self.idNum = Layer.nextIdNum # Layer attribute: unique ID
        Layer.nextIdNum += 1
        #self.z = np.dot(a,self.theta)
        #self.a = activ[activ_f](self.z)
        self.a = np.zeros([self.nodes,1])
        #self.a = list(self.nodes) if a == None else activ[self.activ_f](np.dot(a,self.theta))
        #self.a = activ[self.activ_f](np.dot(a,self.theta))
        
#    def activate(self, K):
#        self.a = activ[self.activ_f](np.dot(K,self.theta)) 
        
class NeuralNetwork:
    
    def __init__ (self):
        '''activation function (1-sigmoid, 2-relu, 0-none)'''
        self.layers = []
            
    def addLayer(self,inputs, nodes_nr, activ_f):
        '''add new layer, specify numbers of nodes and activation function (1-sigmoid, 2-relu, 0-none) '''
        self.layers.append(Layer(inputs, nodes_nr, activ_f))
    
    
    def forwardProp(self,X):
        m, n = X.shape
        
        self.layers[0].a = np.c_[np.ones((m,1)),X]
        #self.layers[0].a = activ[self.layers[0].activ_f](np.dot(self.layers[0].a,self.layers[0].theta))  
        
      
        i=1
        for i in range(self.layers[-1].idNum):
            print('a{ww} : \n'.format(ww=i), self.layers[i].a)
            print('theta{ww} : \n'.format(ww=i), self.layers[i].theta)
            input()
            self.layers[i].a = activ[self.layers[i].activ_f](np.dot(self.layers[i].a,self.layers[i].theta))           
            print('a{ww} : \n'.format(ww=i), self.layers[i].a)
            print('theta{ww} : \n'.format(ww=i), self.layers[i].theta)
            input()
        
        
        
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
        

nn = NeuralNetwork()  
nn.addLayer(11, 6,'relu')   
nn.addLayer(6, 3, 'relu')
nn.addLayer(3, 1,'sigmoid') 

nn.forwardProp(X_train)




    
