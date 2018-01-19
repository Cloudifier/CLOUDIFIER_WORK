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

class Layer:
    nextIdNum = 0 # next ID number to assign
    def __init__(self, nodes,  activ_func):
        '''activation function (1-sigmoid, 2-relu, 0-none)'''
        self.nodes = nodes

        self.activ_f = activ_func
        self.activ_f_prime = 'sigmoidPrime' if self.activ_f == 'sigmoid' else 'reluPrime'
        self.idNum = Layer.nextIdNum # Layer attribute: unique ID
        Layer.nextIdNum += 1

        self.w =[]
        self.b = []
        self.z = []
        self.a = []
        self.dw = []
        self.db = []
        self.dz = []
        self.da = []
        

        
#        self.deltas = np.zeros([self.inputs, self.nodes])


        
class NeuralNetwork:
    
    def __init__ (self, nodes,  activ_func = 'none'):
        '''activation function (1-sigmoid, 2-relu, 0-none)'''
        self.layers = []
        self.Elist = list()
        self.acc = None
        self.addLayer(nodes,  activ_func)
  
        
    def addLayer(self,nodes, activ_f):
        '''add new layer, specify numbers of nodes and activation function (1-sigmoid, 2-relu, 0-none) '''
        self.layers.append(Layer(nodes, activ_f))
    
    def init_param(self):
        #print(self.layers[0].nodes)
        #input()
        for i in range(1,len(self.layers)):
            self.layers[i].w = np.random.randn(self.layers[i].nodes, self.layers[i-1].nodes) * 0.01
            self.layers[i].b = np.zeros((self.layers[i].nodes,1))
    
    def cost_function(self, y, yhat, activ_f):
        
        #m = np.shape(y)
        if activ_f == 'cross_entropy':
            cost =  np.mean(-(y * np.log (yhat) + (1 - y) * np.log(1 - yhat)))
        elif activ_f == 'MSE':
            cost = np.mean((yhat - y)**2)
        self.Elist.append(cost)
        return cost
    
    def forwardProp(self,X):

        m, n = X.shape  
        print(X.T.shape)
        input
        self.layers[0].a = X.T

        for i in range(1,len(self.layers)):
            self.layers[i].z = np.dot(self.layers[i].w,self.layers[i-1].a) + self.layers[i].b 
            self.layers[i].a = activ[self.layers[i].activ_f](self.layers[i].z)  
           

    
    def backwardProp(self, cost, y, batch_dim ):
        
         L = len(self.layers)
         for l in reversed(range(L)):
             self.layers[l].dz = self.layers[l].da * activ[self.layers[l].activ_f_prime](self.layers[l].a)
             self.layers[l].dw = 1/batch_dim * np.dot(self.layers[l].dz,self.layers[l-1].a.T)
             self.layers[l].b = 1/batch_dim * np.sum(self.layers[l].dz)
             self.layers[l-1].da = np.dot(self.layers[l].w.T, self.layers[l].dz)
             
         print(error)
         input()



    def fit(self, X_train, y_train, batch_dim, epochs, learning_rate, lmbd):
        steps = epochs // 10 
        for epoch in range (epochs):
            for j in range(int(m / batch_dim)):
                begin = j * batch_dim
                end = m if ((j + 1) * batch_dim  > m) else (j + 1) * batch_dim
                X_batch = X_train[begin : end]
                y_batch = y_train[begin : end]   
        
                self.forwardProp(X_batch)
                
                cost = self.cost_function(y_batch, self.layers[len(self.layers)].a, batch_dim )
                
                self.backwardProp(X_batch, cost, y_batch, learning_rate)
                
                self.upgrade_param()
                            

#            E = np.mean(-(y_train * np.log(yhat) + (1-y_train) * np.log (1-yhat)))
##            print (E)
##            input()
#            acc = accuracy(yhat, y_train)
#            if epoch % steps == 0 :
#                print(f'#{epoch} epochs. Train accuracy: {acc}%, E: {E}')
#            self.Elist.append(E)   
        
           
        
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
X = normalize(X)
#X = np.c_[np.ones((m,1)),X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#print(X_train.shape)
#input() 
    

nn = NeuralNetwork(10)   
nn.addLayer(6, 'sigmoid') 
nn.addLayer(3, 'sigmoid')
nn.addLayer(1, 'sigmoid')
nn.init_param()
nn.forwardProp(X_train)

#nn.fit(X_train, y_train, 8, 1000, 0.0001) 
#
#yhat = nn.forwardProp(X_test)
#print('Train accuracy: ' + str(nn.acc))
#print('Test accuracy: ' + str(accuracy(yhat, y_test)))
#plt.plot(range(len(nn.Elist)),nn.Elist )
#plt.show()




#nn.forwardProp(X_train)
#nn.backwardProp(X_train, y_train)

#
#for layer in nn.layers:
#    print (layer.a.shape)
#    
#for layer in nn.layers:
#    print (layer.theta.shape)
#    
##for layer in nn.layers:
##    print (layer.deltas.shape)
#    
#for layer in nn.layers:
#    print (layer.theta)


    
