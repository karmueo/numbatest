# @File  : numba_test3.py
# @Author: 沈昌力
# @Date  : 2018/4/8
# @Desc  :
import json
import numpy as np
import numba
import datetime


def mytime(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        func(*args, **kwargs)
        end = datetime.datetime.now()
        print("%s消耗时间%f" % (func.__name__, (end - start).total_seconds()))
    return wrapper

@numba.jit(numba.float32(numba.float32, numba.float32, numba.float32, numba.float32))
def disfunc(x1, y1, x2, y2):
    a = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    b = np.sqrt(abs(x1 - x2) ** 3 + abs(y1 - y2) ** 3)
    c = np.sqrt((x1 - x2) ** 4 + (y1 - y2) ** 4)
    return a + b + c


def dis(pt):
    start = pt['start']
    end = pt['end']
    res = disfunc(start['x'], start['y'], end['x'], end['y'])
    return res

# @numba.jit
def clusters_test(clusters):
    for c in clusters:
        for pt in c:
            dis(pt)

    # r = [dis(pt) for c in clusters for pt in c]

    # for c in clusters:
    #     r = list(map(dis, c))


@mytime
def main(clusters_output_file_name='Data/Clusters.txt'):
    with open(clusters_output_file_name, 'r') as clusters_stream:
        clusters_input = json.loads(clusters_stream.read())
        clusters_test(clusters_input)


if __name__ == '__main__':
    main()
