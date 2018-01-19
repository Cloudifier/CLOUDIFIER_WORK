# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 09:19:34 2018

@author: Mihai.Cristea
"""
from tqdm import trange
from time import sleep

for i in trange(10, desc='1st loop'):
    for j in trange(5, desc='2nd loop', leave=False):
        sleep(0.01)
        for k in trange(100, desc='3nd loop'):
            sleep(0.01)
            
from tqdm import tqdm
import time
for i1 in tqdm(range(5)):
    for i2 in tqdm(range(300)):
        # do something, e.g. sleep
        time.sleep(0.01)