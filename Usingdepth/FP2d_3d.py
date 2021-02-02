import numpy as np
import sys
import os

# import scipy.io as sio
# import re
from libs.libs import pix2m_disp
from libs.variable import imgName1, imgName2, setFPAuto


def readCVMatching(npyPath):
    featurePointList = []
    FP_data = np.load(npyPath)
    print("FPset's number is %d" % FP_data.shape[0])
    if FP_data.shape[0] < 10:
        raise Exception("FPsets are too few")
    for y in range(FP_data.shape[0]):
        FP = (int(FP_data[y][1]), int(FP_data[y][0]))
        featurePointList.append(FP)
    # print(FP_data.shape[0])
    return featurePointList


def FP2d_3d(imgIdx, FP_2d):
    FP_3d = []
    for FP in FP_2d:
        FP_3d.append(pix2m_disp(FP[0], FP[1], imgIdx))
    return np.array(FP_3d)


if __name__ == "__main__":
    if setFPAuto:
        FP_2d_1 = readCVMatching("./FP_2d/FP_" + imgName1 + ".npy")
    else:
        FP_2d_1 = readCVMatching("./FPManual/npy/" + imgName1 + "_FP.npy")
    FP_3d_1 = FP2d_3d(1, FP_2d_1)
    np.save("./FP_3d/" + imgName1, FP_3d_1)

    # FP_2d_2 = readCVMatching("./FP_2d/npy/" + imgName2 + "_FP.npy")
    if setFPAuto:
        FP_2d_2 = readCVMatching("./FP_2d/FP_" + imgName2 + ".npy")
    else:
        FP_2d_2 = readCVMatching("./FPManual/npy/" + imgName2 + "_FP.npy")
    FP_3d_2 = FP2d_3d(2, FP_2d_2)

    np.save("./FP_3d/" + imgName2, FP_3d_2)
    print(FP_3d_2.shape)
    print(FP_3d_2[0])
