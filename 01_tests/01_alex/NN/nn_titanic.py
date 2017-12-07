import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle
from enum import Enum
from typing import List
import os
import matplotlib.pyplot as plt
from NN.nn import *

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)

    return norm, mean, max, min

def meanNormalizationWithParams(X, mean, max, min):
    return (X - mean) / (max - min)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def cleanData(df):
    m, n = df.shape
    #fill missing dates
    for i in range(m):
        name = str(df.ix[i, 'Name']).upper()
        age = df.ix[i, 'Age']
        if not age or math.isnan(age):
            if 'MISS' in name:
                df.ix[i, 'Age'] = 20
            elif 'MASTER' in name:
                df.ix[i, 'Age'] = 15
            elif 'MR' in name:
                df.ix[i, 'Age'] = 40
            elif 'MRS' in name:
                df.ix[i, 'Age'] = 35
            elif 'DR' in name:
                df.ix[i, 'Age'] = 55
            else:
                df.ix[i, 'Age'] = 35

    #remove string columns. for now
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    #fill some nan on specific columns
    df['Embarked'] = df['Embarked'].fillna('S')

    #fill nan with 0
    df = df.fillna(0)

    #get_dummies
    df = pd.get_dummies(df)

    return df

def accuracy(yhat, y):
    m, n = y.shape
    acc = float(np.sum((yhat >= 0.5 * 1) == y) / m)
    return round(acc * 100, 2)

df = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')
dfTestPassIdCol = dfTest.loc[:, 'PassengerId']
df = cleanData(df)
dfTest = cleanData(dfTest)

pred = list()
for i in range(5):
    print(f'Iteration #{i}')
    df = shuffle(df)

    X_train = df.head(800)
    X_train = X_train.drop('Survived', axis=1)
    X_train, mean, max, min = meanNormalization(X_train)
    m_train, n_train = X_train.shape
    y_train = df.head(800).ix[:, 'Survived'].values.reshape(m_train, 1)

    X_test = df.tail(91)
    m_test, n_test= X_test.shape
    X_test = X_test.drop('Survived', axis=1)
    X_test = meanNormalizationWithParams(X_test, mean, max, min)
    y_test = df.tail(91).ix[:, 'Survived'].values.reshape(m_test, 1)

    layers = []
    layers.append(Layer(nrNeurons=n_train))
    layers.append(Layer(nrNeurons=15, activationFunc=ActivationFuncEnum.RELU))
    layers.append(Layer(nrNeurons=8, activationFunc=ActivationFuncEnum.RELU))
    layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.RELU))
    layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.SIGMOID))
    nn = NN(layers, costFunction=CostFuncEnum.CROSSENTROPY, useBias=True)
    nn.fit(X_train, y_train, epochs=100, batchSize=8, alpha=0.03, lmbd=1)

    learningStatus: List[LearningStatus] = nn.getLearningStatus()
    yhat = nn.predict(X_test)
    acc = accuracy(yhat, y_test)
    print(f'Acc with last theta: {acc}')

    X_titanic_test = meanNormalizationWithParams(dfTest, mean, max, min)
    yhat_titanic = (nn.predict(X_titanic_test) >= 0.5) * 1

    pred.append((acc, yhat_titanic))

    best = sorted(learningStatus, key=lambda x: x.accuracy, reverse=True)
    print(best[0].accuracy, best[1].accuracy)
    yhat = nn.predictWithBestTheta(X_test)
    acc = accuracy(yhat, y_test)
    print(f'Acc with best theta: {acc}')
    learningStatus: List[LearningStatus] = nn.getLearningStatus()
    best = sorted(learningStatus, key=lambda x: x.accuracy, reverse=True)
    print(best[0].accuracy, best[1].accuracy)

    # plot cost
    x_pos = [idx for idx, x in enumerate(learningStatus)]
    y_pos = [x.error[0] for idx, x in enumerate(learningStatus)]
    plt.plot(x_pos, y_pos)

max = sorted(pred, key=lambda x: x[0], reverse=True)[0]

#accuracy per iteration
print('-----accuracy per iteration------')
for i in range(len(pred)):
    print(pred[i][0])
print('-----accuracy per iteration------')

print(f'Max accuracy: {max[0]}')
dirPath = os.path.dirname(os.path.realpath(__file__))
submissionDf = pd.DataFrame()
submissionDf.insert(0, 'PassengerId', dfTestPassIdCol)
submissionDf.insert(1, 'Survived', max[1])
submissionDf.to_csv(dirPath + r'\submission_titanic.csv', index=False)
plt.show()
