import numpy as np
import cv2
import os
import scipy.io as sio
import re
from libs.variable import paraDict, LFName, depthImg1, depthImg2

f_mm = paraDict["focal_length_mm"]
s_mm = paraDict["sensor_size_mm"]
b_mm = paraDict["baseline_mm"]
longerSide = max(depthImg1.shape[0], depthImg1.shape[1])
beta = b_mm * f_mm * longerSide
f_pix = (f_mm * longerSide) / s_mm


def pix2m_disp(x, y, imgIdx):
    # print(dispImg1.shape)
    if imgIdx == 1:
        Z = depthImg1[x][y]
    elif imgIdx == 2:
        Z = depthImg2[x][y]
    X = (float(x) - float(depthImg1.shape[1] / 2.0)) * Z / f_pix
    Y = (float(y) - float(depthImg1.shape[0] / 2.0)) * Z / f_pix

    # X = float(x) * Z / f_pix
    # Y = float(y) * Z / f_pix
    return X, Y, -Z  # 単位はmm


# def pix2m_disp_(x, y, imgIdx):
#     # print(dispImg1.shape)
#     if imgIdx == 1 and dispImg1[x][y]:
#         Z = float(beta * f_mm) / float((dispImg1[x][y] * f_mm * s_mm + beta))
#     elif imgIdx == 2 and dispImg2[x][y]:
#         Z = float(beta * f_mm) / float((dispImg2[x][y] * f_mm * s_mm + beta))
#     else:
#         print("zero!!")
#         Z = 0
#     X = (float(x) - float(dispImg1.shape[1] / 2.0)) * Z / f_pix
#     Y = (float(y) - float(dispImg1.shape[0] / 2.0)) * Z / f_pix
#     # X = float(x) * Z / f_pix
#     # Y = float(y) * Z / f_pix
#     return X, Y, -Z  # 単位はmm
