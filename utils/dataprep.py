import numpy as np
import os
import createFeature as cF
import math as mt
from scipy.interpolate import interp1d
from sklearn.cross_validation import StratifiedKFold

def remove_redundantdata(points, threshold = 5):
    index = []

    j = 1
    while j < len(points):
        if mt.sqrt(mt.pow(points[j][0] - points[j-1][0],2) + mt.pow(points[j][1] - points[j-1][1],2)) < 5:
            index.append(j)
            j += 2
        else:
            j += 1
    points = np.delete(points, index, axis = 0)

    # index = []
    # apply again for initial points and ending points
    # for j in range(1, len(points)):
    #     if mt.sqrt(mt.pow(points[j][0] - points[j - 1][0], 2) + mt.pow(points[j][1] - points[j - 1][1], 2)) < 5:
    #         index.append(j)
    #
    # points = np.delete(points, index, axis=0)
    if len(points) < 10:
        print points
    # np.savetxt("nonredundant_points.txt",points, delimiter=' ')
    return points

def interpolatedata(points, size = 100):
    x = np.asarray(range(0, len(points)),dtype=float)
    # normalize x
    x = x/(len(x)-1)
    f2 = interp1d(x, points, kind='linear', axis=0)
    m = np.asarray(range(0, size), dtype=float)
    m = m/(len(m)-1)
    points = f2(m)
    return points

def smoothdata(points, n = 2):

    for j in range(n,len(points)- n):
        pointx = points[j][0]
        pointy = points[j][1]
        for k in range(1,n+1):
            pointx += points[j-k][0] + points[j+k][0]
            pointy += points[j-k][1] + points[j+k][1]
        pointx = pointx/(2*n+1)
        pointy = pointy/(2*n+1)
        points[j] = [pointx,pointy]
    # np.savetxt("aftersmooth_points.txt", points, delimiter=' ')
    return points

def normalizedata(points):
    max = np.amax(points, axis=0)
    min = np.amin(points, axis=0)

    for j in range(0,len(points)):
        pointx = (points[j][0] - min[0])/(max[0] - min[0])
        pointy = (points[j][1] - min[1])/(max[1] - min[1])
        pointx = 2*pointx - 1
        pointy = 2*pointy - 1

        points[j] = [pointx,pointy]
    # np.savetxt("afternormalize_points.txt", points, delimiter=' ')
    return points

def loadPoints(fname="data/0_0_amit.txt"):
    points = np.loadtxt(fname, dtype='string')
    pointsx = points[:,1]
    pointsy = points[:,3]

    pointsx = np.array(pointsx).astype(float)

    pointsy = np.array(pointsy).astype(float)

    points = zip(pointsx, pointsy)

    for i,point in enumerate(points):
        points[i] = np.array(point)

    points = smoothdata(points)
    points = remove_redundantdata(points)
    points = normalizedata(points)
    points = interpolatedata(points)
    # np.savetxt("C:/Users/Amit/Documents/AirScript/preprocessed_data/afterpreprocess_points.txt", points, delimiter=' ')
    return points

def loadData(rootdir):
    training_class_dirs = os.walk(rootdir)
    labeldirs = []
    labels = []
    skip = True
    data_x = []
    data_y = []

    for trclass in training_class_dirs:
        #print(trclass)
        if skip is True:
            labels = trclass[1]
            skip = False
            continue
        labeldirs.append((trclass[0],trclass[2]))

    j = -1
    for i,labeldir in enumerate(labeldirs):
        saveDirPath = ""
        dirPath = labeldir[0]
        filelist = labeldir[1]
        if not bool(filelist):
            j += 1
            continue

        for file in filelist:
            fname = os.path.join(dirPath, file)
            points = loadPoints(fname)    # load point data
            # feat = cF.angle(points)
            feat = cF.direction(points)

            data_x.append(feat)
            data_y.append(labels[j])

    return data_x, data_y

def splitDataset(train,test,target,data):
    train_x = []
    train_y = []

    for i in train:
        train_x.append(data[i])
        train_y.append(target[i])

    val_x = []
    val_y = []

    for i in test:
        val_x.append(data[i])
        val_y.append(target[i])

    return train_x, train_y, val_x, val_y

def createFolds(data_x, data_y, n_folds = 5):
    skf = StratifiedKFold(data_y, n_folds)
    kFolds = []

    for train, test in skf:
        train_x, train_y, val_x, val_y = splitDataset(train, test, data_y, data_x)
        train = (train_x, train_y)
        validation = (val_x, val_y)
        kFolds.append((train, validation))
    return kFolds
