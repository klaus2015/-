# -*- coding: utf-8 -*-
# Author      : Add_on
# Created Time: 2021/11/30 23:29
# Desc        :
import numpy as np

A = np.array([1, 1, 1, 2,1,3]).reshape((3, 2))
b = np.array([1, 2,2]).reshape((3, 1))

A_T = A.T

inv = np.linalg.inv(A_T.dot(A))
inv = np.linalg.pinv(A_T.dot(A))
# print(inv)
x = inv.dot((A_T.dot(b)))
print(x)




# print(np.linalg.inv(kernel))