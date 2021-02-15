import numpy as np
import cv2
import os
import scipy.io as sio
import re
from libs.variable import paraDict, LFName, depthImg1, depthImg2

f_mm = paraDict["focal_length_mm"]
s_mm = paraDict["sensor_size_mm"]
b_mm = paraDict["baseline_mm"]
fd_mm = paraDict["focus_distance_m"] * 1000.0
longerSide = max(depthImg1.shape[0], depthImg1.shape[1])
beta = b_mm * f_mm * longerSide
f_pix = (f_mm * longerSide) / s_mm

print("f_pix", f_pix)


def pix2m_disp(x, y, imgIdx):
    # print(dispImg1.shape)
    if imgIdx == 1:
        Z = depthImg1[x][y]
        # print("idx: 1")
    elif imgIdx == 2:
        # print("idx: 2")
        Z = depthImg2[x][y]
    X = (float(x) - float(depthImg1.shape[1] / 2.0)) * Z / f_pix
    Y = (float(y) - float(depthImg1.shape[0] / 2.0)) * Z / f_pix

    # return X, Y, Z * (1.0 / 170.0)  # 単位はmm
    return X, Y, Z  # 単位はmm

