# -*- coding: utf-8 -*-

# @Author  : Zxm
# @Time    : 2022/1/6 - 15:41
# @Dsce    :

"""
逻辑回归解决多分类问题
案列：手写数字识别
数据集ex3data1.mat
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('ex3data1.mat')
raw_X = data['X']
raw_y = data['y']
print(type(raw_X[1,:]))
print(type(raw_X[1]))


def plot_an_image(X):
    pick_one = np.random.randint(5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(image.reshape(20, 20).T, cmap='gray_r')

    # 去掉刻度
    plt.xticks([])
    plt.yticks([])

plot_an_image(raw_X)

def plot_100_image(X):
    # 显示100张图片
    sample_index = np.random.choice(len(X),100)
    images = X[sample_index,:]
    print(images.shape)
    fig,ax = plt.subplots(ncols=10,nrows=10,figsize=(8,8),sharex=True,sharey=True)
    for r in range(10):
        for c in range(10):
            ax[r,c].imshow(images[10*r+c].reshape(20,20).T,cmap='gray_r')
    # 去掉刻度
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_100_image(raw_X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_func(theta, X, y, lr):
    A = sigmoid(X @ theta)

    first = y * np.log(A)
    second = (1 - y) * np.log(1 - A)

    # reg = np.sum(np.power(theta[1:],2)) * (lr / (2 * len(X)))
    # scipy要求theta为一维
    reg = theta[1:] @ theta[1:] * (lr / (2 * len(X)))
    return -np.sum(first + second) / len(X) + reg


def gradient_reg(theta, X, y, lr):
    m = len(X)
    reg = theta[1:] * (lr / m)
    reg = np.insert(reg, 0, values=0, axis=0)

    A = sigmoid(X @ theta)
    first = X.T @ (A - y) / len(X)
    # 返回梯度向量
    return first + reg

# (5000, 401)
X = np.insert(raw_X,0,values=1,axis=1)

# (5000,)
y = raw_y.flatten()

from scipy.optimize import minimize


def one_vs_all(X, y, lr, K):
    n = X.shape[1]
    theta_all = np.zeros((K, n))

    for i in range(1, K + 1):
        theta_i = np.zeros(n, )

        res = minimize(fun=cost_func,
                       x0=theta_i,
                       args=(X, y == i, lr),
                       method='TNC',
                       jac=gradient_reg)
        theta_all[i - 1, :] = res.x
    return theta_all

lr = 1
K = 10
# (10, 401)
theta_final = one_vs_all(X,y,lr,K)


def predict(X, theta_final):
    h = sigmoid(X @ theta_final.T)  # (5000,401) (10,401) => (5000,10)
    # axis=1 沿着列比较，每一行所有列的值进行比较，选出最大值的索引
    h_argmax = np.argmax(h, axis=1)

    return h_argmax + 1  # 0索引代表的数字1，为了保证索引和数字对齐，所以索引加一

y_pred = predict(X,theta_final)

acc = np.mean(y_pred==y)

# 0.9446
print(acc)