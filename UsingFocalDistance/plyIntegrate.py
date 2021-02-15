# read np.dot write
import numpy as np
import os
import sys

# import glob
import copy
from libs.plyClass import Ply
from libs.variable import imgName1, imgName2, saveName, splitRate


def meshChange(mesh, r=255, g=255, b=255, sigma=1.0, alpha=255):
    mesh.changeColor(sigma=sigma)
    if not alpha == 255:
        mesh.changeAlpha(alpha=alpha)
        print("alpha change")
    # print("mesh")
    return mesh


mesh1_fi = "./mesh/" + imgName1 + ".ply"
mesh2_fi = "./mesh/" + imgName2 + ".ply"
meshMiddle_fi = "./mesh/" + saveName + "middle.ply"
dotM_save_fi = "./mesh/" + saveName + ".ply"

mesh1 = Ply(mesh1_fi)
mesh2 = Ply(mesh2_fi)
meshTemp1 = Ply(mesh1_fi)
meshTemp2 = Ply(mesh2_fi)
# mesh1=meshChange(mesh1,r=255,b=255,g=255,sigma=1.0)
# mesh1.changeColor(r=255,b=255,g=0,sigma=1)
# meshMiddle = Ply(meshMiddle_fi)
npyMiddlePath = "./M/" + "%s_middleM.npy" % saveName
npyMiddleInvPath = "./M/" + "%s_middleMInv.npy" % saveName
# npyMiddleInvPath = "./M/" + "GradM.npy"
# npyMiddlePath = "./M/" + "GradM.npy"

npyMPath = "./M/" + saveName + ".npy"
save_fi = "./mesh/" + saveName + "_integrated.ply"


def integrate(meshList):
    dstMesh = meshList[0]
    # dstMesh = meshChange(dstMesh, r=0, alpha=128)
    # dstMesh = meshChange(dstMesh, sigma=255, alpha=255)
    for idx in range(len(meshList) - 1):
        print(idx)
        srcMesh = meshList[1 + idx]
        # srcMesh = meshChange(srcMesh, sigma=255, alpha=255)

        dstMesh.integrate([srcMesh.v_infos, srcMesh.num_vertex])
    dstMesh.ClassWritePly(save_fi)

    return dstMesh


def dotMesh(mesh, npyPath):
    meshDotM = mesh.dotsM(npyPath)
    return meshDotM


def main():
    meshMiddle = dotMesh(meshTemp1, npyMiddlePath)
    meshMiddleInv = dotMesh(meshTemp2, npyMiddleInvPath)
    # meshMiddleInv = meshChange(meshMiddleInv, r=0, alpha=128)
    # meshMiddle = meshChange(meshMiddle, g=0, alpha=128)
    # mesh1.changeColor(b=0)

    # mesh1_2 = integrate([mesh1, mesh2])
    if splitRate == 2:
        meshMiddle_MiddleInv = integrate([meshMiddle, meshMiddleInv])  # splitRate=2
    elif splitRate == 1:
        mesh12_2 = integrate([meshMiddle, mesh2])  # splitRate=1
        # mesh21_1 = integrate([meshMiddleInv, mesh1])  # splitRate=1

    # mesh1_2.ClassWritePly(save_fi)
    # meshMiddle_MiddleInv.ClassWritePly(save_fi)


main()

# threeIntegrate()
