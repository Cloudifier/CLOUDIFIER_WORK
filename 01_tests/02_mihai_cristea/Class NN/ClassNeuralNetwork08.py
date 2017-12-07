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
    def __init__(self,  inputs, nodes,  activ_func):
        self.nodes = nodes
        self.inputs = inputs
        self.theta = np.random.rand(self.inputs, self.nodes)
        self.activ_f = activ_func
        self.activ_f_prime = 'sigmoidPrime' if self.activ_f == 'sigmoid' else 'reluPrime'
        self.idNum = Layer.nextIdNum # Layer attribute: unique ID
        Layer.nextIdNum += 1
        self.a = np.zeros([self.nodes,1])
#        self.deltas = np.zeros([self.inputs, self.nodes])


        
class NeuralNetwork:
    
    def __init__ (self):
        '''activation function (1-sigmoid, 2-relu, 0-none)'''
        self.layers = []
        self.Elist = list()
        self.acc = None
            
    def addLayer(self,inputs, nodes_nr, activ_f):
        '''add new layer, specify numbers of nodes and activation function (1-sigmoid, 2-relu, 0-none) '''
        self.layers.append(Layer(inputs, nodes_nr, activ_f))
    
    
    def forwardProp(self,X):

        m, n = X.shape        
#        self.layers[0].a = np.c_[np.ones((m,1)),X]
        #self.layers[0].a = X
        #z = np.dot(self.layers[0].a,self.layers[0].theta)
#        print(X)
#        print(self.layers[0].theta)
#        input()
        z = X.dot(self.layers[0].theta)
#        print(z)
#        input()
        self.layers[0].a = activ[self.layers[0].activ_f](z) 
#        print(self.layers[0].a)
#        input()
#        self.layers[0].a = activ[self.layers[0].activ_f](np.dot(self.layers[0].a,self.layers[0].theta))        
        for i in range(1,len(self.layers)):
            self.layers[i].a = activ[self.layers[i].activ_f](np.dot(self.layers[i-1].a,self.layers[i].theta))           
#            print(f'# i = {i}  a : {self.layers[i].a }, a.shape: {self.layers[i].a.shape}')
#            input()
#        print(f'# -1  a : {self.layers[-1].a }, a.shape: {self.layers[-1].a.shape}')
#        input()
        return self.layers[-1].a
    
    def backwardProp(self, X, y, learning_rate):
 
         error = y - self.layers[-1].a
         print(error)
         input()
    #     delta = [error * activ[self.layers[-1].activ_f_prime](self.layers[-1].a)]
         delta = [error]
         print(delta)
         input()
         for l in range(len(self.layers) - 2, -1,  -1): 
                 print(f'# l = {l}  theta : {self.layers[l-1].a.shape }, theta.shape: {self.layers[l].theta.T.shape}')
                 input()
                 ddd = delta[-1].dot(self.layers[l+1].theta.T)
                 dd = activ[self.layers[l].activ_f_prime](self.layers[l-1].a.dot(self.layers[l].theta))
                 print(f'# l = {l}  dd : {dd.shape }, ddd: {ddd.shape}')
                 input()
                 d = ddd * dd
                 print('d :',d)
                 input()
                 delta.append(d)
         #delta[-1].dot(self.layers[1].theta.T) * activ[self.layers[l].activ_f_prime](self.layers[l].a.dot(self.layers[l+1].theta))
         delta.reverse()
         print(delta)
         input()
         m,n = X.shape
         #X = np.c_[np.ones((m,1)),X]
#         if m == 0:
#             print('m', m)
#             input()
#         grd = 1/m * X.T.dot(delta[0])
         grd = np.mean(X.T.dot(delta[0]))
         self.layers[0].theta = self.layers[0].theta - learning_rate * grd
         
         for i in range(1,len(self.layers)):           
#                grd = 1/m * self.layers[i-1].a.T.dot(delta[i])
                grd = np.mean(self.layers[i-1].a.T.dot(delta[i]))
                self.layers[i].theta = self.layers[i].theta - learning_rate * grd 

    def fit(self, X_train, y_train, batch_dim, epochs, learning_rate):
        steps = epochs // 10 
        for epoch in range (epochs):
            for j in range(int(m / batch_dim)):
                begin = j * batch_dim
                end = m if ((j + 1) * batch_dim  > m) else (j + 1) * batch_dim
                X_batch = X_train[begin : end]
                y_batch = y_train[begin : end]   
        
                self.forwardProp(X_batch)
                
                self.backwardProp(X_batch, y_batch, learning_rate)
                            
            yhat = self.forwardProp(X_train)
#            print(yhat.shape)
#            print(y_train.shape)
#            input()
#            print(y_train * np.log(yhat))
#            print((1-y_train) * np.log (1-yhat))            
#            input()
            E = np.mean(-(y_train * np.log(yhat) + (1-y_train) * np.log (1-yhat)))
#            print (E)
#            input()
            acc = accuracy(yhat, y_train)
            if epoch % steps == 0 :
                print(f'#{epoch} epochs. Train accuracy: {acc}%, E: {E}')
            self.Elist.append(E)   
        
           
        
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
X = np.c_[np.ones((m,1)),X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
 
        

nn = NeuralNetwork()  
nn.addLayer(11, 6,'sigmoid')   
nn.addLayer(6, 3, 'sigmoid')
nn.addLayer(3, 1,'sigmoid')

nn.fit(X_train, y_train, 8, 1000, 0.0001) 

yhat = nn.forwardProp(X_test)
print('Train accuracy: ' + str(nn.acc))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))
plt.plot(range(len(nn.Elist)),nn.Elist )
plt.show()




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


    
