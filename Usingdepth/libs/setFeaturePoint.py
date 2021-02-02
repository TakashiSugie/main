import sys
import os
import numpy as np
import cv2
from libs.variable import img1, img2, saveName, imgName1, imgName2

# mouse callback function
index1 = 0
index2 = 0
FPList1 = []
FPList2 = []


def draw_circle1(event, x, y, flags, param):
    global index1, FPList1
    # print(max(img1.shape))
    circleSize = int(max(img1.shape) * 0.03)
    textSize = int(max(img1.shape) * 0.005)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img1, (x, y), circleSize, (255, 0, 0), -1)
        cv2.putText(
            img1,
            str(index1),
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            textSize,
            (255, 255, 255),
            textSize,
            cv2.LINE_AA,
        )
        index1 += 1
        FPList1.append(np.array([x, y]))


def draw_circle2(event, x, y, flags, param):
    global index2, FPList2
    circleSize = int(max(img1.shape) * 0.03)
    textSize = int(max(img1.shape) * 0.005)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2, (x, y), circleSize, (255, 0, 0), -1)
        cv2.putText(
            img2,
            str(index2),
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            textSize,
            (255, 255, 255),
            textSize,
            cv2.LINE_AA,
        )
        index2 += 1
        FPList2.append(np.array([x, y]))


def saveNpy(FP1, FP2):
    file1_data = np.array(FP1)
    file2_data = np.array(FP2)
    # print(file1_data.shape, file1_data.shape)
    if file1_data.shape[0] > 9 and file1_data.shape == file2_data.shape:
        np.save("./FPManual/npy/" + imgName1 + "_FP", file1_data)
        np.save("./FPManual/npy/" + imgName2 + "_FP", file2_data)
    else:
        print(file1_data.shape)
        raise Exception("FP shape is wrong")


# if __name__ == "__main__":
def setFPManual():
    cv2.namedWindow("image1", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image1", draw_circle1)
    cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image2", draw_circle2)

    while 1:
        cv2.imshow("image1", img1)
        cv2.imshow("image2", img2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    saveNpy(FPList1, FPList2)
    cv2.imwrite("./FPManual/img/" + imgName1 + ".png", img1)
    cv2.imwrite("./FPManual/img/" + imgName2 + ".png", img2)


if __name__ == "__main__":
    setFPManual()
