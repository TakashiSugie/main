from libs.plyClass2 import Ply
from libs.variable import imgName1, imgName2, saveName


def main():
    mesh_fi = "./mesh/IMG_4652.ply"
    npyPath = "./M/" + saveName + ".npy"
    mesh = Ply(mesh_fi=mesh_fi)
    mesh.setInfos()
    mesh.np2infos()
    # mesh.changeColor2()

    # mesh.()
    # print(type(mesh.v_line))
    # print(mesh.v_line[0:100])
    # mesh.changeRound2()
    print(mesh.colors_np[0:3])
    print(mesh.verts_np[0:3])

    # mesh1_2 = mesh.dotsM(npyPath)
    # print(mesh1_2.verts_np[0])


if __name__ == "__main__":
    main()
