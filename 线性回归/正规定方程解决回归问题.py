# -*- coding: utf-8 -*-
# Author      : Add_on
# Created Time: 2022/1/3 23:48
# Desc        :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', names=['Population', 'Profit'])
# data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
# plt.show()
data.insert(0,'ones',1)
X = data.iloc[:,:-1]
y = data.iloc[:,2:3]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros((2,1))

def normal_equation(X, y):
    theta = np.linalg.inv(X.T*X)*X.T*y
    return theta

theta = normal_equation(X,y)
print(theta)

"""
[[-3.89578088]
 [ 1.19303364]]
"""