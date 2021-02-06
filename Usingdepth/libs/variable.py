import numpy as np
import cv2
import os
import scipy.io as sio
import re
import glob

# from libs import matLoad, readCg


def readCg(cgPath):
    patternList = ["focal_length_mm", "sensor_size_mm", "baseline_mm"]
    paraDict = {}
    if cgPath:
        with open(cgPath) as f:
            s = f.read()
            sLines = s.split("\n")
            for sLine in sLines:
                for pattern in patternList:
                    if re.match(pattern, sLine):
                        sList = sLine.split()
                        paraDict[pattern] = float(sList[2])
    else:
        paraDict = {
            "focal_length_mm": 100.0,
            "sensor_size_mm": 35.0,
            "baseline_mm": 90.0,
        }
    return paraDict


def matLoad(u, v):
    mat = sio.loadmat(
        "/home/takashi/Desktop/dataset/from_iwatsuki/mat_file/additional_disp_mat/%s.mat"
        # "../../for_mac/mat_file/additional_disp_mat/%s.mat"
        % LFName
    )
    disp_gt = mat["depth"]
    # print(type(disp_gt[u][v]))
    return disp_gt[u][v]


def longerResize(img, longerSideLen=640):
    # print(img.shape[0])
    longer = max(img.shape[0], img.shape[1])
    fraq = float(longerSideLen) / float(longer)
    if fraq < 1:
        img = cv2.resize(img, (int(img.shape[1] * fraq), int(img.shape[0] * fraq)))
    return img


def disp2depth(dispImg):
    depthImg = np.zeros(dispImg.shape)
    # print(paraDict)

    f_mm = paraDict["focal_length_mm"]
    s_mm = paraDict["sensor_size_mm"]
    b_mm = paraDict["baseline_mm"]
    longerSide = max(dispImg.shape[0], dispImg.shape[1])
    beta = b_mm * f_mm * longerSide
    # f_pix = (f_mm * longerSide) / s_mm
    for x in range(dispImg.shape[1]):
        for y in range(dispImg.shape[0]):
            depthImg[x][y] = float(beta * f_mm) / float(
                (-dispImg[x][y] * f_mm * s_mm + beta)
            )
    return depthImg
    Min, Max = np.min(dispImg), np.max(dispImg)
    dispImg = (dispImg - Min) / (Max - Min) * 0.4 + 99.7
    return dispImg


u1, v1 = 0, 0
u2, v2 = 8, 8  # 0~8(uが→方向　vが下方向)
# u2, v2 = 0, 1  # 0~8(uが→方向　vが下方向)
camNum1 = u1 * 9 + v1
camNum2 = u2 * 9 + v2
cgPath = True
setFPAuto = True
useManualFP = False
require_midas = False
# longerSideLen = 160
# longerSideLen = 1008
longerSideLen = 640
renderingPly = {
    1: "mesh1",
    2: "mesh2",
    3: "mesh2_1",
    4: "mesh1+mesh2_1",
}
renderingMode = 1
content = "additional"
# content = "lf"
# content = "ori"

if content == "ori":
    basePath = "/home/takashi/Desktop/dataset/image"
    LFName = "chairDesk23"
    dirPath = os.path.join(basePath, LFName)
    imgPathList = glob.glob(dirPath + "/*")
    imgName1 = os.path.splitext(os.path.basename(imgPathList[0]))[0]
    imgName2 = os.path.splitext(os.path.basename(imgPathList[1]))[0]
    threshold = False


else:
    basePath = os.path.join("/home/takashi/Desktop/dataset/lf_dataset", content)
    LFName = "antinous"
    if content == "additional":
        imgName1 = "input_Cam{:03}".format(camNum1)
        imgName2 = "input_Cam{:03}".format(camNum2)
        cfgName = "parameters.cfg"
        cgPath = os.path.join(basePath, LFName, cfgName)
    elif content == "lf":
        imgName1 = "%02d_%02d" % (u1, v1)
        imgName2 = "%02d_%02d" % (u2, v2)
    threshold = True
paraDict = readCg(cgPath)


imgPath1 = os.path.join(basePath, LFName, imgName1 + ".png")
imgPath2 = os.path.join(basePath, LFName, imgName2 + ".png")

img1 = cv2.imread(imgPath1)
img2 = cv2.imread(imgPath2)
img1 = longerResize(img1, longerSideLen=longerSideLen)
img2 = longerResize(img2, longerSideLen=longerSideLen)
# print(img1.shape)
# dispImg2, dispImg1 = None, None
if require_midas:
    if os.path.isfile("./depth/" + imgName1 + ".npy") and os.path.isfile(
        "./depth/" + imgName2 + ".npy"
    ):
        depth1 = np.load("./depth/" + imgName1 + ".npy")
        depth2 = np.load("./depth/" + imgName2 + ".npy")

    elif os.path.isfile("./depth/" + imgName1 + ".png") and os.path.isfile(
        "./depth/" + imgName2 + ".png"
    ):
        depth1 = cv2.imread("./depth/" + imgName1 + ".png", 0)
        depth2 = cv2.imread("./depth/" + imgName2 + ".png", 0)
        # print("\n\nimg:", dispImg1)
    # if os.path.isfile("./depth/" + imgName1 + ".npy") and os.path.isfile(
    #     "./depth/" + imgName2 + ".npy"
    # ):
    #     dispImg1 = np.load("./depth/" + imgName1 + ".npy")
    #     dispImg2 = np.load("./depth/" + imgName2 + ".npy")

    # elif os.path.isfile("./depth/" + imgName1 + ".png") and os.path.isfile(
    #     "./depth/" + imgName2 + ".png"
    # ):
    #     dispImg1 = cv2.imread("./depth/" + imgName1 + ".png", 0)
    #     dispImg2 = cv2.imread("./depth/" + imgName2 + ".png", 0)
    #     # print("\n\nimg:", dispImg1)

    if "depth1" in locals():
        Min, Max = np.min(depth1), np.max(depth1)
        depthImg1 = (depth1 - Min) / (Max - Min) * 0.4 + 99.7
        # dispImg1 = (dispImg1 - Min) / (Max - Min) * 0.4 + 99.7
        Min, Max = np.min(depth2), np.max(depth2)
        depthImg2 = (depth2 - Min) / (Max - Min) * 0.4 + 99.7
        # dispImg2 = (dispImg2 - Min) / (Max - Min) * 0.4 + 99.7
        # print(dispImg1)
        print("dispMax:", np.max(depthImg1), "dispMin:", np.min(depthImg1))
        Max, Min = np.max(depthImg1), np.min(depthImg1)
        cv2.imwrite("ESTantinous.png", (depthImg1 - Min) / (Max - Min) * 255)


else:
    dispImg1 = matLoad(u1, v1)
    dispImg2 = matLoad(u2, v2)
    depthImg1 = disp2depth(dispImg1)
    depthImg2 = disp2depth(dispImg2)

    Max, Min = np.max(dispImg1), np.min(dispImg1)  #
    print("dispMax:", np.max(dispImg1), "dispMin:", np.min(dispImg1))

    Max, Min = np.max(depthImg1), np.min(depthImg1)  #
    print("depthMax:", np.max(depthImg1), "depthMin:", np.min(depthImg1))
    # diff=dispImg1-dispImg2
    
    del dispImg1
    del dispImg2
    # cv2.imwrite("GTantinous.png", (depthImg1 - Min) / (Max - Min) * 255)

# ここをMidasOnlyから出てきたNpyに書き換える
# Depthとかは正直おかしいかもしれないが、そこに関してはスルー
# Depthか視差かも正直怪しい、disp→Depth変換をつけたりなしにする必要があるかも

width = img1.shape[1]
height = img1.shape[0]
saveName = LFName + "_" + str(camNum1) + "_" + str(camNum2)
