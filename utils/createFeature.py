import numpy as np
import math as mt


def direction(points, m = 1):

    return points

def angle(points, inc=1):

    size = int(len(points)/inc) # last point not taken into consideration
    if not len(points)%inc:
        size = size - 1
    # if len(points) < 21 :
    #   while len(dat) < 21:
    #       dat.append(dat[-1])\n",
    # print len(dat)\n",
    i = int(1)
    d = []
    while i < size:
        a2 = np.array(points[i*inc-inc] - points[i*inc])
        a1 = np.array(points[i*inc+inc] - points[i*inc])
        # print a2 - a1
        if (a1[0] ==  0 and a1[1] ==  0) or (a2[0] == 0 and a2[1] == 0):
            a = 0
        else:
            a = np.arccos(np.sum(a1 * a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)))
        if mt.isnan(a):
            a = 0
        d.append(a)
        # d.append(dat[i*inc][1])
        i += 1
    return d

def direction(points, m = 1):
    d = []
    for j in range(m,len(points)):
        if (points[j][0] - points[j-m][0]) == 0:
            a = 0
        else:
            a = (points[j][1] - points[j-m][1]) / (points[j][0] - points[j-m][0])
        d.append(a)

    return d