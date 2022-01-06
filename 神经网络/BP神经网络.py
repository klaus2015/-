# -*- coding: utf-8 -*-

# @Author  : Zxm
# @Time    : 2022/1/6 - 18:51
# @Dsce    :
"""
神经网络解决多分类问题
手写数字识别
数据集：ex4data1.mat
初始参数：ex4weights.mat

"""
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = sio.loadmat('ex4data1.mat')
raw_X = data['X']
# (5000, 1)
raw_y = data['y']
# (5000, 401)
X = np.insert(raw_X,0,values=1,axis=1)

# 1. 对y进行独热编码处理：one-hot编码
def one_hot_encoder(raw_y):
    result = []
    for i in raw_y:  # 1-10
        y_temp = np.zeros(10)
        y_temp[i - 1] = 1
        result.append(y_temp)

    return np.array(result)

# (5000, 10)
y = one_hot_encoder(raw_y)

theta = sio.loadmat('ex4weights.mat')
# ((25, 401), (10, 26))
theta1 = theta['Theta1']
theta2 = theta['Theta2']