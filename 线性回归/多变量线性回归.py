# -*- coding: utf-8 -*-
# Author      : Add_on
# Created Time: 2022/1/2 23:49
# Desc        :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])

def normalize_feature(data):
    return (data-data.mean()) / data.std()

data = normalize_feature(data)
data.insert(0,'ones',1)
X = data.iloc[:,:-1]
X = X.values
y = data.iloc[:,-1:]
y = y.values

def cost_func(X,y,theta):
    inner = np.power(X @ theta - y,2)
    return np.sum(inner) / (2 * len(X))
theta = np.zeros((3,1))
cost_init = cost_func(X,y,theta)
print(cost_init)

def gradient_descent(X, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = cost_func(X, y, theta)
        # print('theta', theta)
        # print('cost', cost)
        costs.append(cost)

        if i % 200 == 0:
            print(cost)

    return theta, costs

candidate_alphas = [0.0003,0.003,0.0001,0.01]
iters = 2000
frg,ax = plt.subplots()
for alpha in candidate_alphas:
    _,costs = gradient_descent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters),costs,label=alpha)
    ax.legend()
ax.set(xlabel='iters',ylabel='cost',title='cost vs iters')
plt.show()