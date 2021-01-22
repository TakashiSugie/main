# read np.dot write
import numpy as np
import os
import sys

# import glob
# import copy
from libs.plyClass import Ply
from libs.variable import imgName1, imgName2, saveName


def main():
    mesh1_fi = "./mesh/" + imgName1 + ".ply"
    mesh2_fi = "./mesh/" + imgName2 + ".ply"
    mono_save_fi = "./mesh/" + saveName + ".ply"
    save_fi = "./mesh/" + saveName + "_integrated.ply"
    npyPath = "./M/" + saveName + ".npy"
    mesh1 = Ply(mesh1_fi)
    mesh2 = Ply(mesh2_fi)
    # mesh1.changeColor(r=0, g=0, b=255, sigma=1.0)
    # mesh2.changeColor(r=0, g=255, b=0)
    # mesh1_2_fi = "./mesh/" + saveName + ".ply"
    # mesh1_2 = Ply(mesh1_2_fi)
    # mesh1が7 mesh2が9 mesh1_2つまり、1を2の座標系、つまり7を9に、integratedにはmesh1+mesh1_2
    mesh1_2 = mesh1.dotsM(npyPath)
    # mesh1_2 = copy.deepcopy(mesh1)
    mesh1_2.ClassWritePly(mono_save_fi)
    mesh1_2.integrate([mesh2.v_infos, mesh2.num_vertex])
    mesh1_2.ClassWritePly(save_fi)

    for key in mesh2.__dict__.keys():
        # print(key)
        pass


main()
