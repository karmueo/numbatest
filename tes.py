# @File  : tes.py
# @Author: 沈昌力
# @Date  : 2018/4/11
# @Desc  :
import time
from functools import wraps
import pandas as pd
import numpy as np

def fn_timer(function):
    @wraps(function)  #functools.wraps 的作用是将原函数对象的指定属性复制给包装函数对象
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.__name__, str(t1-t0))
               )
        return result
    return function_timer


@fn_timer
def fun1(n):
    l = []
    for i in range(n):
        l.append(i)
    return l

@fn_timer
def fun2(n):
    nd = np.zeros(n)
    for i in range(n):
        nd[i] = i
    return nd.tolist()

def fun3(n):
    df = pd.DataFrame([])
    for i in range(n):
        df = df.append(pd.DataFrame([i]))
    return df

if __name__ == "__main__":
    # fun1(10000000)
    # fun2(10000000)
    fun3(10000000)