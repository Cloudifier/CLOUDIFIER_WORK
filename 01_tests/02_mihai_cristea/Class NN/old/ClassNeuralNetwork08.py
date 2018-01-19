# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:18:40 2017
vertical
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
    return (X - X.mean(axis=0)) / np.std(X, axis = 0)

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)
    return norm #, mean, max, min

def minmaxNorm(X):
    _min = X.min(axis = 0)
    _max = X.max(axis = 0)    
    return (X - _min) / (_max-_min)

def activation(g, function):
    if function == 'sigmoid':
        activ = sigmoid(g)
    elif function == 'relu':
        activ = relu(g)
    elif function == 'sigmoidPrime':
        activ = sigmoidPrime(g)
    elif function == 'reluPrime':
        activ = reluPrime(g)    
#    elif function == 'softmax':
#        activ = softmax(g)
    return activ


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
        
        
class NeuralNetwork:
    
    def __init__ (self, nodes,  activ_func = 'none'):
        '''activation function (1-sigmoid, 2-relu, 0-none)'''
        self.layers = []
        self.Elist = list()
        self.acc = list()
        self.addLayer(nodes,  activ_func)
  
        
    def addLayer(self,nodes, activ_f):
        '''add new layer, specify numbers of nodes and activation function (1-sigmoid, 2-relu, 0-none) '''
        self.layers.append(Layer(nodes, activ_f))
    
#    def init_param(self):        
#        np.random.seed(100)
#        for i in range(1,len(self.layers)):
#            self.layers[i].w = np.random.randn(self.layers[i].nodes, self.layers[i-1].nodes) * 0.1
#            self.layers[i].b = np.zeros((self.layers[i].nodes,1))

    def init_param(self):        
        np.random.seed(100)
        for i in range(1,len(self.layers)):
            self.layers[i].w = np.random.randn( self.layers[i-1].nodes, self.layers[i].nodes) * 0.1
            self.layers[i].b = np.zeros((1, self.layers[i].nodes))
   
    def cost_function(self, y, yhat, activ_f):
        m, n = np.shape(y)
        if activ_f == 'cross_entropy':
            #cost =  np.mean(-(y * np.log (yhat) + (1 - y) * np.log(1 - yhat)))
            cost =  1/m * np.sum(-(y * np.log (yhat) + (1 - y) * np.log(1 - yhat)))
        elif activ_f == 'MSE':
            cost = np.mean(2 * (yhat - y))
        return cost
    
    def upgrade_param(self, learning_rate, lmbd):
        L = len(self.layers) - 1
        for l in reversed(range(1,L)):
            self.layers[l].w = self.layers[l].w - learning_rate * self.layers[l].dw 
            self.layers[l].b = self.layers[l].b - learning_rate * self.layers[l].db
    
    
    def forwardProp(self,X):
        self.layers[0].a = X
        for i in range(1,len(self.layers)):
            self.layers[i].z = np.dot(self.layers[i-1].a,self.layers[i].w) + self.layers[i].b 
            self.layers[i].a = activation(self.layers[i].z, self.layers[i].activ_f)  
        yhat = self.layers[-1].a
        return yhat
    
    def predict(self,X):
        return self.forwardProp(X)
    
    def backwardProp(self, cost, y, batch_dim, learning_rate, lmbd ):         
         L = len(self.layers) - 1
         self.layers[L].dz = np.mean(self.layers[L].a - y)
         self.layers[L].dw = 1/batch_dim * np.dot(self.layers[L-1].a.T,self.layers[L].dz)
         self.layers[L].db = np.sum(self.layers[L].dz)
#         print(np.shape(self.layers[L].a))
#         print(np.shape(y))
#         print('a-y', np.mean(self.layers[L].a - y))
#         print(np.shape(self.layers[L-1].a))
#         print(self.layers[L].dz)
#         input()        
         for l in reversed(range(1,L)):
#             print(self.layers[l].dz)
#             print(self.layers[l+1].dz)
#             print(np.shape(self.layers[l-1].a))
#             print(np.shape(self.layers[l].w))
#             input()
             self.layers[l].dz = np.dot(self.layers[l+1].w, self.layers[l+1].dz) * activation(self.layers[l].a.T,self.layers[l].activ_f_prime)
#             self.layers[l].dw = 1/batch_dim * np.dot(self.layers[l].dz,self.layers[l-1].a.T) 
#             print(np.shape(self.layers[l].dz))
#             print(np.shape(self.layers[l-1].a))
#             print(np.shape(self.layers[l].w))
#             input()
             self.layers[l].dw = 1/batch_dim * np.dot(self.layers[l-1].a.T,self.layers[l].dz.T) #+ lmbd / (2 * batch_dim) * self.layers[l].w
             self.layers[l].db = 1/batch_dim * np.sum(self.layers[l].dz)
 
          
    def fit(self, X_train, y_train, batch_dim, epochs, learning_rate, lmbd):
        self.init_param()
        m = y_train.shape[0]
        steps = epochs // 10 

        for epoch in tqdm(range(epochs)):
            for j in range(int(m / batch_dim)):
                begin = j * batch_dim
                end = m if ((j + 1) * batch_dim  > m) else (j + 1) * batch_dim
                X_batch = X_train[begin : end,:]
                y_batch = y_train[begin : end,:]

                self.forwardProp(X_batch)
                cost = self.cost_function(y_batch, self.layers[-1].a, 'cross_entropy' )
                self.backwardProp( cost, y_batch, batch_dim, learning_rate, lmbd)
                self.upgrade_param(learning_rate,lmbd)
                


            yhat = self.predict(X_train)                     
            self.forwardProp(X_train)
            self.Elist.append(self.cost_function(y_train, yhat, 'cross_entropy' ))
            self.acc.append(accuracy(yhat, y_train))            
            if epoch % steps == 0 :
                tqdm.write(f'#{epoch} epochs. Train accuracy: {self.acc[-1]}%, E: {self.Elist[-1]}')
#                print(f'\n #{epoch} epochs. Train accuracy: {self.acc[-1]}%, E: {self.Elist[-1]}')  
        
           
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progressbar import ProgressBar, Bar, ETA, Percentage
from tqdm import tqdm
from sklearn.model_selection import train_test_split
pbar = ProgressBar(maxval=80) 

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
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#print(X_train.shape)
#print(y_train.shape)
#input() 
   


nn = NeuralNetwork(X.shape[1])   
nn.addLayer(6, 'relu') 
nn.addLayer(3, 'relu')
nn.addLayer(1, 'sigmoid')

#nn = NeuralNetwork(X.shape[1])   
#nn.addLayer(6, 'sigmoid') 
#nn.addLayer(3, 'sigmoid')
#nn.addLayer(1, 'sigmoid')

#nn = NeuralNetwork(10)   
#nn.addLayer(8, 'sigmoid') 
#nn.addLayer(5, 'sigmoid')
#nn.addLayer(3, 'sigmoid')
#nn.addLayer(1, 'sigmoid')

#nn = NeuralNetwork(X.shape[1])  
#nn.addLayer(8, 'relu') 
#nn.addLayer(5, 'relu')
#nn.addLayer(3, 'relu')
#nn.addLayer(1, 'sigmoid')

#nn.init_param()
#nn.forwardProp(X_train.T)


### X shape (n,m) - m = sample numbers in rows, n = features number
#nn.fit(X_train.T, y_train.T, 8, 1000, 0.05,lmbd=0.01)
nn.fit(X_train, y_train, 8, 2000, 0.01,lmbd= 0.01)



yhat = nn.predict(X_test)
print('Train accuracy: ' + str(nn.acc[-1]))
print('Test accuracy: ' + str(accuracy(yhat, y_test)))
plt.plot(range(len(nn.Elist)),nn.Elist )
plt.show()




#for layer in nn.layers:
#    print (layer.a.shape)
##for layer in nn.layers:
##    print (layer.w)
#for layer in nn.layers:
#    print (np.shape(layer.w))  
#for layer in nn.layers:
#    print (np.shape(layer.z))
#for layer in nn.layers:
#    print (np.shape(layer.dz))  
#for layer in nn.layers:
#    print (np.shape(layer.dw))        

#    
##for layer in nn.layers:
##    print (layer.deltas.shape)
#    



#for i in reversed(range(1,len(nn.layers))):
#     print (i)
#     print ('dz \n', nn.layers[i].dz)
#     print ('dw \n', nn.layers[i].dw)
#     print ('db \n', nn.layers[i].db)

#np.random.seed(0) 
#print(np.random.rand(4))
#print(np.random.randn(4))



    
