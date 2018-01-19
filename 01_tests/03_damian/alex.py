# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:11:26 2018

@author: Andrei Ionut Damian
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_mldata
#from sklearn.model_selection import train_test_split

print("loading...", flush=True)
mnist = fetch_mldata('mnist-original')
X = mnist.data / 255.
y = mnist.target
#X_train, X_test, y_train, y_test = train_test_split(X, mnist.target, test_size=0.3)

model = KMeans(n_clusters=10, n_jobs=-1)

print("Training...", flush=True)
model.fit(X)


