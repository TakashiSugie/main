# create ply from npy

import numpy as np

# import os
from libs.variable import imgName1, imgName2, img1, img2
from libs.plyClass import Ply

if __name__ == "__main__":
    mesh1_fi = "./mesh/" + imgName1 + ".ply"
    mesh1 = Ply(mesh_fi=None, img=img1, imgIdx=1)
    mesh1.ClassWritePly(mesh1_fi)
    mesh2_fi = "./mesh/" + imgName2 + ".ply"
    mesh2 = Ply(mesh_fi=None, img=img2, imgIdx=2)
    mesh2.ClassWritePly(mesh2_fi)
