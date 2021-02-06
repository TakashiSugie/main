# read np.dot write
import numpy as np
import os
import sys

# import glob
import copy
from libs.plyClass import Ply
from libs.variable import imgName1, imgName2, saveName


def meshChange(mesh,r=255,g=255,b=255,sigma=1.0):
    mesh.changeColor(r=r,g=g,b=b,sigma=sigma)
    # print("mesh")
    return mesh

mesh1_fi = "./mesh/" + imgName1 + ".ply"
mesh2_fi = "./mesh/" + imgName2 + ".ply"
meshMiddle_fi="./mesh/" + saveName + "middle.ply"
dotM_save_fi = "./mesh/" + saveName + ".ply"

mesh1 = Ply(mesh1_fi)
mesh2 = Ply(mesh2_fi)
meshTemp1 = Ply(mesh1_fi)
meshTemp2 = Ply(mesh2_fi)
# mesh1=meshChange(mesh1,r=255,b=255,g=255,sigma=1.0)
# mesh1.changeColor(r=255,b=255,g=0,sigma=1)
# meshMiddle = Ply(meshMiddle_fi)
npyMiddlePath = "./M/" + "middleM.npy"
npyMiddleInvPath = "./M/" + "middleMInv.npy"
npyMPath = "./M/" + saveName + ".npy"
save_fi = "./mesh/" + saveName + "_integrated.ply"


def integrate(meshList):
    dstMesh=meshList[0]
    # for i in range(len(meshList)):
    #     print(meshList[i].v_infos[:3])
    #     print(meshList[i].colors_np[:3])

    # print(len(dstMesh.v_infos))


    for idx in range(len(meshList)-1):
        print(idx)
        srcMesh=meshList[1+idx]

        dstMesh.integrate([srcMesh.v_infos, srcMesh.num_vertex])
    return dstMesh

def dotMesh(mesh,npyPath):
    meshDotM = mesh.dotsM(npyPath)
    return meshDotM


def main():
    # mesh1_fi = "./mesh/" + imgName1 + ".ply"
    # mesh2_fi = "./mesh/" + imgName2 + ".ply"
    # dotM_save_fi = "./mesh/" + saveName + ".ply"
    # save_fi = "./mesh/" + saveName + "_integrated.ply"
    # middleName = "middleM"
    # npyPath = "./M/" + saveName + ".npy"
    # npyPath = "./M/" + middleName + ".npy"
    # mesh1 = Ply(mesh1_fi)
    # mesh2 = Ply(mesh2_fi)
    # mesh1.changeColor(r=0, g=0, b=255, sigma=1.0)
    # mesh2.changeColor(r=0, g=255, b=0)
    # mesh1_2_fi = "./mesh/" + saveName + ".ply"
    # mesh1_2 = Ply(mesh1_2_fi)
    # mesh1が7 mesh2が9 mesh1_2つまり、1を2の座標系、つまり7を9に、integratedにはmesh1+mesh1_2
    meshDotM = mesh1.dotsM(npyMPath)
    # mesh1_2 = copy.deepcopy(mesh1)
    meshDotM.ClassWritePly(dotM_save_fi)
    meshDotM.integrate([mesh2.v_infos, mesh2.num_vertex])
    meshDotM.ClassWritePly(save_fi)

def main2():
    meshMiddle=dotMesh(meshTemp1,npyMiddlePath)
    meshMiddleInv=dotMesh(meshTemp2,npyMiddleInvPath)
    meshMiddleInv=meshChange(meshMiddleInv,sigma=0.5)
    meshMiddle=meshChange(meshMiddle,sigma=0.5)
    # mesh1.changeColor(b=0)

    # mesh1_2=integrate([mesh1,mesh2,meshMiddle,meshMiddleInv])
    meshMiddle_MiddleInv=integrate([meshMiddle,meshMiddleInv])
    # mesh1_2.ClassWritePly(save_fi)
    meshMiddle_MiddleInv.ClassWritePly(save_fi)



main2()

# threeIntegrate()
