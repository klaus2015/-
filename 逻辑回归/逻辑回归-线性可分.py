# -*- coding: utf-8 -*-
# Author      : Add_on
# Created Time: 2022/1/4 23:15
# Desc        :
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'ex2data1.txt'
data = pd.read_csv(path,names=['Exam 1','Exam 2','Accepted'])
data.head()

# fig,ax = plt.subplots()
# ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
# ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
# ax.legend()
# ax.set(xlabel='exam1',ylabel='exam2')
# plt.show()

def get_Xy(data):
    data.insert(0,'ones',1)
    X_ = data.iloc[:,0:-1]
    X = X_.values
    y = data.iloc[:,-1:]
    y = y.values
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_func(X,y,theta):
    A = sigmoid(X@theta)
    first = y*np.log(A)
    second = (1-y)*np.log(1-A)
    return -np.sum((first + second)) / len(X)

X,y = get_Xy(data)
theta = np.zeros((3,1))
cost_init = cost_func(X,y,theta)
print(cost_init)

def gradientDescent(X,y,theta,iters,alpha):
    m = len(X)
    costs = []
    for i in range(iters):
        A = sigmoid(X@theta)
        theta = theta - (alpha/m) * X.T@(A-y)
        cost = cost_func(X,y,theta)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs,theta

alpha = 0.004
iters = 200000
costs,fina_theta = gradientDescent(X,y,theta,iters,alpha)
coef1 = -fina_theta[0,0] / fina_theta[2,0]
coef2 = -fina_theta[1,0] / fina_theta[2,0]

x = np.linspace(20,100,100)
y = coef1 + coef2*x
fig,ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='exam1',ylabel='exam2')
ax.plot(x,y,c='g')
plt.show()