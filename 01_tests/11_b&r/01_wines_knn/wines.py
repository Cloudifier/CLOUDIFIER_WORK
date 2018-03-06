import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv("winequality-red.csv", sep=';')

#print(df)

x = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

x_min = x.min(axis=0)
x_max = x.max(axis=0)
x_norm = (x - x_min)/(x_max - x_min)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.2)

def knn(x, xt, yt, k, return_topk = True):
    
    x_dif = xt - x
    dist = (x_dif**2).sum(axis=1)
    best_k = np.argsort(dist)[0:k]
    best_preds = yt[best_k]
    pred = Counter(best_preds).most_common(1)[0][0]
    if return_topk:
        return pred, best_preds
    else:
        return pred
    
preds = []
for i in range(x_test.shape[0]):
    preds.append(knn(x_test[i], x_train, y_train, 5, False))
    
print("accuracy = {:.2f}".format((preds == y_test).sum()/y_test.shape[0]))

#inmultire
#transpunere
#inversare