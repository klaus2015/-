# -*- coding: utf-8 -*-

# @Author  : Zxm
# @Time    : 2022/1/6 - 17:38
# @Dsce    :
import numpy as np
import scipy.io as sio

data = sio.loadmat('ex3data1.mat')
raw_X = data['X']
raw_y = data['y']

# (5000, 401)
X = np.insert(raw_X,0,values=1,axis=1)

# (5000,1) --> (5000,)
y = raw_y.flatten()

# 读取已经训练好的theta参数
theta = sio.loadmat('ex3weights.mat')

# (25, 401)
theta1 = theta['Theta1']
# (10, 26)
theta2 = theta['Theta2']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 输入层 (5000,401)
a1 = X

z2 = X @ theta1.T  # (5000,401) (401,25) => (5000,25)
# (5000, 25)
a2 = sigmoid(z2)

# (5000, 26)
a2 = np.insert(a2,0,values=1,axis=1)

z3 = a2 @ theta2.T  # (5000,26) (26,10) => (5000,10)
# (5000, 10)
a3 = sigmoid(z3)

p_pred = np.argmax(a3,axis=1)
y_pred = p_pred + 1

acc = np.mean(y_pred == y)
print(acc) # 0.9752