# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:28:21 2017

@author: Mihai.Cristea
"""

import numpy as np
import pandas as pd

titanic_df = pd.read_excel('titanic3.xls')

#print(titanic_df[(titanic_df['name'].str.contains('miss')) & (titanic_df['age'].notnull())] )
a = titanic_df[titanic_df['name'].str.contains('Miss')]
titanic_df['title'] = titanic_df['name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
titanic_df['age_mean'] = titanic_df.groupby('title')['age'].transform('mean').round(0)
titanic_df['age'].fillna(titanic_df['age_mean'], inplace=True)
titanic_df.drop('age_mean', axis=1, inplace=True)
print(titanic_df)