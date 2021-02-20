from libs.plyClass import Ply
import numpy as np
import cv2

F = 1462.857142857143
C_X = 256.0
C_Y = 256.0


def main():
    mesh_fi = "./mesh/input_Cam000.ply"
    mesh1 = Ply(mesh_fi=mesh_fi)
    img = np.zeros((512, 512, 3))
    intrinsic = np.array([[F, 0, C_X], [0, F, C_Y], [0, 0, 1]])
    for idx in range(mesh1.verts_np.shape[0]):
        Point_3d = mesh1.verts_np[idx]
        Point_2d = np.dot(intrinsic, Point_3d) / Point_3d[2]
        img[int(Point_2d[0])][int(Point_2d[1])] = mesh1.colors_np[idx][:3]
        # 恐らく穴が開くのはここの丸めのときに同じ値に入ってしまうところがあるから？
    cv2.imwrite("projection.png", img)


main()
