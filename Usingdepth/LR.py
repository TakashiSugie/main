import sys
import os

# import matplotlib.pyplot as plt
import numpy as np

# import glob
from libs.variable import imgName1, imgName2, saveName
from sklearn import preprocessing
import statistics
import math

error = 0
count = 0


def read3Dnp(npPath):
    FPnp = np.load(npPath)
    FPDict = {}
    for idx in range(FPnp.shape[0]):
        Z = float(FPnp[idx][2])
        # FPnp[idx] = FPnp[idx]/Z
        FPDict[str(idx)] = [
            float(FPnp[idx][0]),
            float(FPnp[idx][1]),
            float(FPnp[idx][2]),
        ]

    return FPDict


def LR_(X, Y):
    from sklearn import linear_model

    global error, count
    clf = linear_model.LinearRegression()
    X = np.array(X)
    Y = np.array(Y)
    clf.fit(X, Y)
    coef = list(clf.coef_)
    coef.append(clf.intercept_)
    predict = 0
    for sample_index in range(X.shape[0]):

        no_intercept = 0
        for i in range(3):
            no_intercept += clf.coef_[i] * X[sample_index][i]
        predict = clf.intercept_ + no_intercept
        # print("predict:", predict, "  GT", Y[sample_index])
        error += np.abs(predict - Y[sample_index])
        count += 1
    return coef


def LR(X, Y):
    global error, count
    X = np.array(X)
    Y = np.array(Y)
    mm = preprocessing.MinMaxScaler()
    # X = mm.fit_transform(X)
    # Y = mm.fit_transform(Y)

    # print(X_mm)
    # print(Y)
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, ones), axis=1)
    a = np.dot(np.dot((np.linalg.inv(np.dot(X.T, X))), X.T), Y)
    predict = 0
    # error, count = 0, 0
    for sample_index in range(X.shape[0]):
        predict = 0
        for i in range(4):
            predict += a[i] * X[sample_index][i]
        # predict = +no_intercept
        # print("predict:", predict, "  GT", Y[sample_index])
        error += np.abs(predict - Y[sample_index])
        count += 1
    return a

def calcSzMedian(point1List,point2List):
    Z1List,Z2List,SzList=[],[],[]
    zeroCount=0
    for idx in range(len(point1List)):
        Z1List.append(point1List[idx][2])
        Z2List.append(point2List[idx][2])
    for num1 in range(len(Z1List)):
        for num2 in range(len(Z1List)):
            if num1<num2:
                diff1=Z1List[num1]-Z1List[num2]
                diff2 =Z2List[num1]-Z2List[num2]
                if diff2==0:
                    # print("0000000")
                    zeroCount+=1
                else:
                    SzList.append(float(diff1)/float(diff2))
    print("zeroNum",zeroCount)

    medianSz=statistics.median(SzList)
    print("medianSz",medianSz)
    meanSz=statistics.mean(SzList)
    print("meanSz",meanSz)
    
    # print(Z1List,Z2List)
    # print(np.array(point2List).shape)
    # print(point2List)
    pass


def createData():
    M = []
    FPDict1 = read3Dnp("./FP_3d/" + imgName1 + ".npy")  # ("key": (x1,y1,z1))
    FPDict2 = read3Dnp("./FP_3d/" + imgName2 + ".npy")  # ("key": (x1',y1',z1'))
    X_train, y_train = [], []

    for key, value1 in FPDict1.items():
        value2 = FPDict2[key]
        X_train.append(value1)  # (x1,y1,z1)
        y_train.append(value2)  # (x1',y1',z1')
    # [[x1'],[y1'],[z1']]なんでかというとx1'=f(x1,y1,z1にしたかったから)
    calcSzMedian(X_train,y_train)

    y_train = list(np.array(y_train).T)

    for i in range(3):
        M.append(LR(X_train, y_train[i]))
    return M


def main():
    M = np.array(createData())
    print("M:", M)
    print("error_per_sample", error / float(count))
    # Path = "M_" + imgName1 + "_" + imgName2
    npyPath = "./M/" + saveName
    np.save(npyPath, M)


main()
