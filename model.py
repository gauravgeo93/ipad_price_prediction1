# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:14:07 2020

@author: Gaurav Verma
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Data.csv')
df.head()

x = df.iloc[:,1:]
y = df.iloc[:,0]

def screen_to_int(word):
    word_dict = {'Mini': 1,'Air':2,'Pro':3}
    return word_dict[word]

def capacity_to_int(word):
    word_dict = {'16GB': 1,'32GB':2,'64GB':3,'128GB':4}
    return word_dict[word]

def connectivity_to_int(word):
    word_dict = {'Wifi': 1,'Cellular':2,'wifi':1}
    return word_dict[word]

def generation_to_int(word):
    word_dict = {'Previous': 1,'current':2,'Current':2}
    return word_dict[word]

x['Screen'] = x['Screen'].apply(lambda x:screen_to_int(x))
x['Capacity'] = x['Capacity'].apply(lambda x:capacity_to_int(x))
x['Connectivity'] = x['Connectivity'].apply(lambda x:connectivity_to_int(x))
x['Gen'] = x['Gen'].apply(lambda x:generation_to_int(x))

reg = LinearRegression()
reg.fit(x,y)

pickle.dump(reg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,2,1,2]]))