# -*- coding: utf-8 -*-
# Author      : Add_on
# Created Time: 2022/1/1 17:15
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
# theta.shape
def cost_func(X,y,theta):
    inner = np.power(X @ theta - y,2)
    return np.sum(inner) / (2 * len(X))

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
alpha = 0.02
iters = 2000
theta,costs = gradient_descent(X,y,theta,alpha,iters)

# fla,ax = plt.subplots()
# print(type(ax))
# ax.plot(np.arange(iters),costs)
# ax.set(xlabel='iters',ylabel='cost',title='cost vs iters')
# plt.show()

x = np.linspace(y.min(),y.max(),100)
print(theta[0,0])
print(theta[1,0])
print('*'*20)
print(theta)
print('*'*20)
y_ = theta[0,0] + theta[1,0] * x
fla,ax = plt.subplots()
# X,y卫列表中套了一层列表，不添加tolist会报错
ax.scatter(X[:,1].tolist(),y.tolist(),label='train data')
ax.plot(x,y_,'r',label='predict')
ax.legend()
ax.set(xlabel='population',ylabel='profit')
plt.show()

# # <class 'numpy.matrix'>
# print(type(X[:,1]))
# # <class 'list'>
# print(type(X[:,1].tolist()))
