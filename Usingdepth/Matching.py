# OpenCVのFlannMatchingを用いて特徴点抽出を行い、FP_2dにnpy形式で保存する

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy
import sys


from libs.variable import (
    imgName1,
    imgName2,
    imgPath1,
    imgPath2,
    saveName,
    setFPAuto,
    threshold,
    useManualFP,
)
from libs.setFeaturePoint import setFPManual

# https: // code-graffiti.com/opencv-feature-matching-in-python/
# https://python-debut.blogspot.com/2020/02/csv.html
df = pd.DataFrame(columns=["x_query", "y_query", "x_train", "y_train", "Distance"])


def display(img, cmap="gray"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")
    plt.show()


def flannMatching(hacker, items):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(hacker, None)
    kp2, des2 = sift.detectAndCompute(items, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    saveMatches = []
    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7 * match2.distance:
            good.append([match1])
            saveMatches.append([match1, match2])

    saveCsv(saveMatches, kp1, kp2)

    flann_matches = cv2.drawMatchesKnn(hacker, kp1, items, kp2, good, None, flags=0)
    cv2.imwrite("./FPImg/" + saveName + ".png", flann_matches)


def saveCsv(matches, kp_train, kp_query):
    for i in range(len(matches)):
        xyDistance = np.square(
            kp_train[matches[i][0].queryIdx].pt[0]
            - kp_query[matches[i][0].trainIdx].pt[0]
        ) + np.square(
            kp_train[matches[i][0].queryIdx].pt[1]
            - kp_query[matches[i][0].trainIdx].pt[1]
        )
        if xyDistance > 100 and threshold:
            pass
        else:
            df.loc["Matches" + str(i)] = [
                kp_train[matches[i][0].queryIdx].pt[0],
                kp_train[matches[i][0].queryIdx].pt[1],
                kp_query[matches[i][0].trainIdx].pt[0],
                kp_query[matches[i][0].trainIdx].pt[1],
                matches[i][0].distance,
            ]

    df.to_csv("./FP_2d/" + saveName + ".csv")


def saveNpy():
    file1_data = np.loadtxt(
        "./FP_2d/" + saveName + ".csv",  # 読み込みたいファイルのパス
        delimiter=",",  # ファイルの区切り文字
        skiprows=1,  # 先頭の何行を無視するか（指定した行数までは読み込まない）
        usecols=(1, 2),  # 読み込みたい列番号
    )
    np.save("./FP_2d/FP_" + imgName1, file1_data)

    file2_data = np.loadtxt(
        "./FP_2d/" + saveName + ".csv",  # 読み込みたいファイルのパス
        delimiter=",",  # ファイルの区切り文字
        skiprows=1,  # 先頭の何行を無視するか（指定した行数までは読み込まない）
        usecols=(3, 4),  # 読み込みたい列番号
    )
    np.save("./FP_2d/FP_" + imgName2, file2_data)


def longerResize(img, longerSideLen):
    longer = max(img.shape[0], img.shape[1])
    fraq = float(longerSideLen) / float(longer)
    dst = cv2.resize(img, (int(img.shape[1] * fraq), int(img.shape[0] * fraq)))
    return dst


def main():
    if setFPAuto:
        hacker = cv2.imread(imgPath1, 1)
        items = cv2.imread(imgPath2, 1)
        longer = max(hacker.shape[0], hacker.shape[1])
        # if longer > lon:
        #     hacker = longerResize(hacker, 640)
        #     items = longerResize(items, 640)
        flannMatching(hacker=hacker, items=items)
        saveNpy()
    elif useManualFP:
        pass
    else:
        setFPManual()


main()
