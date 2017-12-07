# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:18:01 2017

@author: Mihai.Cristea
"""

layers = [1,2,3,4]
for layer in reversed(list(enumerate(layers))):
    print (layer[1])

for layer in reversed(list(enumerate(layers[:-1]))):
    print (layer)

for layer in reversed(list(enumerate(layers))):
    print (layer[1])   
    
for layer in reversed(layers):
    print (layers.index(layer))
    
for layer in reversed(layers[:-1]):
    print (layer)

for layer in reversed(list(layers)):
    print (layer) 