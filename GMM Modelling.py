# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:19:22 2020

@author: katra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

k = np.random.randint(5,10) # Number of Gaussians
d = np.random.randint(2,8) # Number of Features
N = np.random.randint(1000,1500) # Samples
dataset = np.zeros(N*d).reshape(d,N)
label = np.zeros(N)

m = np.random.randint(4,8)*np.random.randn(k)
v = 0.1*np.random.randn(k)


for i in range(d):
    for j in range(N):
        dataset[i][j] = np.random.normal(m[j%k],abs(v[j%k]),1)

dataset = dataset.T

d1 = pd.DataFrame({'F1': dataset[:,0],'F2': dataset[:,1],'F3': dataset[:,2],'F4': dataset[:,3]})

d1.to_csv('Class1.csv')

d = pd.read_csv('Class1.csv')
