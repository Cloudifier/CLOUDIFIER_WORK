import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from Finals.algo.classification import Classification
from Finals.algo.nn import Layer, ActivationFuncEnum, CostFuncEnum

np.random.seed(1234)

def meanNormalization(X):
    mean = X.mean()
    max = X.max()
    min = X.min()
    norm = (X - mean) / (max - min)

    return norm, mean, max, min

def meanNormalizationWithParams(X, mean, max, min):
    return (X - mean) / (max - min)

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

df = pd.read_csv('_data/train.csv')
dfTest = pd.read_csv('_data/test.csv')
df = cleanData(df)
dfTest = cleanData(dfTest)

df = shuffle(df)

X_train = df.head(800)
X_train = X_train.drop('Survived', axis=1)
X_train, mean, max, min = meanNormalization(X_train)
m_train, n_train = X_train.shape
y_train = df.head(800).ix[:, 'Survived'].values.reshape(800, 1)

X_test = df.tail(91)
m_test, n_test= X_test.shape
X_test = X_test.drop('Survived', axis=1)
X_test = meanNormalizationWithParams(X_test, mean, max, min)
y_test = df.tail(91).ix[:, 'Survived'].values.reshape(91, 1)

stats = list()
classif = Classification(name='Titanic', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
stats.append(classif.kmeans(n_clusters=2))
stats.append(classif.gaussian_naive_bayes())
stats.append(classif.knn())
stats.append(classif.decission_tree())
stats.append(classif.logistic_regression())


layers = []
layers.append(Layer(nrNeurons=n_train))
layers.append(Layer(nrNeurons=15, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=8, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=3, activationFunc=ActivationFuncEnum.RELU))
layers.append(Layer(nrNeurons=1, activationFunc=ActivationFuncEnum.SIGMOID))
stats.append(classif.NN(epochs=100, layers=layers, costFunction=CostFuncEnum.CROSSENTROPY, useBias=False, early_stopping=False, theta_init=(-0.5, 0.5)))

df = pd.DataFrame.from_records([x.to_dict() for x in stats])
df.to_csv('_dataframes/titanic.csv', index=False)