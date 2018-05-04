# @File  : numba_test2.py
# @Author: 沈昌力
# @Date  : 2018/4/8
# @Desc  :
import numba as nb
import numpy as np
import datetime
from timeit import timeit


def mytime(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        func(*args, **kwargs)
        end = datetime.datetime.now()
        print("%s消耗时间%f" % (func.__name__, (end - start).total_seconds()))
    return wrapper


# 普通的 for
@mytime
def add1(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs


# list comprehension
@mytime
def add2(x, c):
    return [xx + c for xx in x]


# 使用 jit 加速后的 for
@mytime
@nb.jit(nopython=True)
def add_with_jit(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs


# @mytime
@nb.vectorize("float32(float32, float32)", nopython=True)
def add_with_vec(x, c):
    return x + c


y = np.random.random(10**5).astype(np.float32)
# print(y.shape)
x = y.tolist()

# add1(x, 1)
# add2(x, 1)
# add_with_jit(x, 1)
# add_with_vec(x, 1.0)

# assert np.allclose(add1(x, 1), add2(x, 1), add_with_jit(x, 1))
# %timeit add2(x, 1)
# %timeit add_with_jit(x, 1)
# print(np.allclose(wrong_add(x, 1), 1))


import math
from concurrent.futures import ThreadPoolExecutor

# 计算类似于 Sigmoid 的函数


def np_func(a, b):
    return 1 / (a + np.exp(-b))

# 参数中的 result 代表的即是我们想要的结果，后同
# 第一个 kernel，nogil 参数设为了 False


@nb.jit(nopython=True, nogil=False)
def kernel1(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))

# 第二个 kernel，nogil 参数设为了 True


@nb.jit(nopython=True, nogil=True)
def kernel2(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))


def make_single_task(kernel):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        kernel(result, *args)
        return result
    return func


def make_multi_task(kernel, n_thread):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        args = (result,) + args
        # 将每个线程接受的参数定义好
        chunk_size = (length + n_thread - 1) // n_thread
        chunks = [[arg[i * chunk_size:(i + 1) * chunk_size] for i in range(n_thread)] for arg in args]
        # 利用 ThreadPoolExecutor 进行并发
        with ThreadPoolExecutor(max_workers=n_thread) as e:
            for _ in e.map(kernel, *chunks):
                pass
        return result
    return func


length = 10 ** 9
a = np.random.rand(length).astype(np.float32)
b = np.random.rand(length).astype(np.float32)


nb_func1 = make_single_task(kernel1)
nb_func2 = make_multi_task(kernel1, 8)
nb_func3 = make_single_task(kernel2)
nb_func4 = make_multi_task(kernel2, 8)

start = datetime.datetime.now()
rs_np = np_func(a, b)
end = datetime.datetime.now()
print("消耗时间%f" % ((end - start).total_seconds()))

start = datetime.datetime.now()
rs_nb1 = nb_func1(length, a, b)
end = datetime.datetime.now()
print("消耗时间%f" % ((end - start).total_seconds()))

start = datetime.datetime.now()
rs_nb2 = nb_func2(length, a, b)
end = datetime.datetime.now()
print("消耗时间%f" % ((end - start).total_seconds()))

start = datetime.datetime.now()
rs_nb3 = nb_func3(length, a, b)
end = datetime.datetime.now()
print("消耗时间%f" % ((end - start).total_seconds()))

start = datetime.datetime.now()
rs_nb4 = nb_func4(length, a, b)
end = datetime.datetime.now()
print("消耗时间%f" % ((end - start).total_seconds()))
