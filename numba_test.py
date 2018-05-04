# @File  : numba_test.py
# @Author: 沈昌力
# @Date  : 2018/4/2
# @Desc  :
import numpy as np
from numba import jit
from numba import vectorize
import time

@jit
def mattest2(m1:np.ndarray, m2:np.ndarray):
    l = len(m1)
    m3 = np.zeros(shape=m1.shape)
    for i in range(l):
        for j in range(l):
            a = m1[i][j]
            b = m2[i][j]
            c = np.sqrt(a ** 2 + b ** 2)
            m3[i][j] = c
    return m3

# @jit
def compu(a, b):
    return np.sqrt(a**2 + b**2)

@jit
def mattest(m1, m2):
    l = len(m1)
    m3 = []
    for i in range(l):
        tmp = []
        for j in range(l):
            a = m1[i][j]
            b = m2[i][j]
            c = compu(a, b)
            tmp.append(c)
        m3.append(tmp)
    return m3


sz = 2000
a = np.random.random((sz, sz))
A = a.tolist()
b = np.random.random((sz, sz))
B = b.tolist()
start = time.time()
# mattest2(a, b)
mattest(A, B)
end = time.time()
print(end - start)