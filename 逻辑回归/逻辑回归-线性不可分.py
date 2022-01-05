# -*- coding: utf-8 -*-

# @Author  : Zxm
# @Time    : 2022/1/5 - 16:58
# @Dsce    :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex2data2.txt'
data = pd.read_csv(path,names=['Test 1','Test 2','Accepted'])

# fig,ax = plt.subplots()
# ax.scatter(data[data['Accepted']==0]['Test 1'],data[data['Accepted']==0]['Test 2'],c='r',marker='x',label='y=0')
# ax.scatter(data[data['Accepted']==1]['Test 1'],data[data['Accepted']==1]['Test 2'],c='b',marker='o',label='y=1')
# ax.legend()
# ax.set(xlabel='Test 1',ylabel='Test 2')
# plt.show()

def feature_mapping(x1,x2,power):
    data = {}
    for i in np.arange(power+1):
        for j in np.arange(i+1):
            data['F{}{}'.format(i-j,j)] = np.power(x1,i-j)*np.power(x2,j)
    return pd.DataFrame(data)

x1 = data['Test 1']
x2 = data['Test 2']
data2 = feature_mapping(x1,x2,6)
X = data2.values
y = data.iloc[:,-1:].values
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 损失函数
def cost_func(X, y, theta, lr):
    A = sigmoid(X@theta)

    first = y*np.log(A)
    second = (1 - y) * np.log(1-A)

    reg = np.sum(np.power(theta[1:], 2)) * (lr / (2 * len(X)))
    return -np.sum(first + second) / len(X) + reg
lr = 1
theta = np.zeros((28,1))
cost_init = cost_func(X,y,theta,lr)
print(cost_init)

# 梯度下降函数
def gradientDescent(X, y, theta, iters, alpha, lr):
    m = len(X)
    costs = []
    for i in range(iters):

        reg = theta[1:] * (lr / m) * alpha
        reg = np.insert(reg, 0, values=0, axis=0)

        A = sigmoid(X @ theta)
        theta = theta - (alpha / m) * X.T @ (A - y) - reg
        cost = cost_func(X, y, theta, lr)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs, theta

alpha = 0.001
iters = 200000
lr = 0.001
costs,fina_theta =  gradientDescent(X,y,theta,iters,alpha,lr)
def predict(X,y,theta):
    prob = sigmoid(X@theta)
    return [1 if x>=0.5 else 0 for x in prob]
y_  = np.array(predict(X,y,fina_theta))
y_pre = y_.reshape(len(y_),1)

acc = np.mean(y_pre == y)
print(acc)

x = np.linspace(-1.2,1.2,200)
# 生成网格状矩阵，200*200
xx,yy = np.meshgrid(x,x)

# 特征映射
z = feature_mapping(xx.ravel(),yy.ravel(),6).values

# 类似X @ theta =0的线
zz = z @ fina_theta
zz = zz.reshape(xx.shape)

fig,ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Test 1'],data[data['Accepted']==0]['Test 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Test 1'],data[data['Accepted']==1]['Test 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='Test 1',ylabel='Test 2')
plt.contour(xx,yy,zz,0)
plt.show()