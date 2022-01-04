# -*- coding: utf-8 -*-

# @Author  : Zxm
# @Time    : 2021/12/21 - 17:51 
# @Dsce    :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
# path路径，sep分隔符，默认 , headers：类型整数，指定第几行作为列名，，如果没有指定列名，默认headers等于0，names:类型列表。指定列名
data = pd.read_csv(path, sep=',',header=None, names=['Population', 'Profit'])
# print(data.head())

# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()

def computeCost(X, y, theta):
    # * and @ both represent the multiplication of matrices.
    inner = np.power(((X * theta) - y), 2)
    # inner = np.power(((X @ theta.T) - y), 2)
    # print('inner', inner)
    return np.sum(inner) / (2 * len(X))

# 索引为0的列插入列名为ones的数值为1的一列
data.insert(0,'ones',1)

# 初始化X和y
cols = data.shape[1]
X = data.iloc[:,:-1]#X是data里的除最后列
y = data.iloc[:,cols-1:cols]#y是data最后一列
# 三种方法效果一样
X = np.matrix(X.values)
# X = X.values
# X = np.array(X)
# y = np.matrix(y.values)

y = y.values
# print(y.shape)
# theta = np.matrix(np.array([0,0]))
theta = np.zeros((2,1))
print(theta)
print(computeCost(X, y, theta))
print()

# def gradient_descent(X, y, theta, alpha, iters):
#
#     costs_list = []
#     for i in range(iters):
#         theta = theta - (X.T@(X@theta - y)) * alpha / len(X)
#         cost = computeCost(X, y, theta)
#         costs_list.append(cost)
#
#         if i % 100 == 0:
#             print(cost)
#
#     return theta, costs_list


# def gradient_descent1(X, y, theta, alpha, iters):
#     costs = []
#     for i in range(iters):
#         # print(theta)
#         theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
#         cost = computeCost(X, y, theta)
#         # print('theta', theta)
#         # print('cost', cost)
#         costs.append(cost)
#
#         if i % 100 == 0:
#             print(cost)
#
#     return theta, costs
# alpha = 0.02
# iters = 2000
# theta, costs = gradient_descent1(X,y,theta,alpha,iters)