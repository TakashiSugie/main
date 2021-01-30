import numpy as np

# 用いるのは3Dの特徴点マッチングを用いた変換行列M(3*4)
# この変換行列はカメラ１からカメラ２への変換
# この行列の中から並進T、回転R、スケールS、Z方向のスケール係数sz1,ともう一つのSz2を求める
# そしてそれらのパラメータをもとにカメラ１からカメラ２のちょうど中間を求めようとしている
# 求める際の仮定 S=1,Sz1=Sz2
# Rが直積行列に近くなるようなSz1とSz2を推定、Mからsz成分を抜いてRとTを取り出す
# 求めたパラメータを用いて中間になるような変換行列を導出
#


def calcNorm(R):
    norm0 = np.linalg.norm(R, 2, axis=0)
    norm1 = np.linalg.norm(R, 2, axis=1)
    return norm0, norm1


# Mを入力、Sz1=Sz2としてSzを求める
# M=[Sz1](R|T)[1/Sz2]の3*4行列
# 左から1/sz1を、右から1/sz2をかけることでR|Tを取り出す
def calcSz1Sz2(M):
    MAEMin = 100
    count = 0
    # SR = M[:, 0:3].T  # こっちのほうがなんかあってるぽい、多分SZ<1だから
    SR = M[:, 0:3]
    # T = M[:, 3]
    sz1 = 1
    sz2 = 1
    norm0, norm1 = calcNorm(SR)
    while True:
        sz1Matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0 / sz1]])
        sz2Matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, sz2]])
        R1 = np.dot(sz2Matrix, SR)
        R12 = np.dot(R1, sz1Matrix)
        norm0, norm1 = calcNorm(R12)
        MAE = np.array([[1, 1, 1], [1, 1, 1]]) - np.array([norm0, norm1])
        loss = np.linalg.norm(MAE, 1)
        if MAEMin > loss:
            MAEMin = loss
        elif MAEMin <= loss:
            print("this loss is minimum")
            break
        count += 1
        sz1 += 1
        sz2 += 1

    # szInv1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0 / sz1]])
    # szInv2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, sz2]])
    # print(sz1, sz2)
    # R1 = np.dot(szInv1, R12)
    # SR = np.dot(R1, szInv2)
    # print(M)
    print("R:\n", R12, "\n")

    # print(norm0)
    # print(norm1)

    return sz1, sz2, R12


def calcT12(M):
    t = M[:, 3]
    t[2] = t[2] * sz1
    return t


def calcMiddleM(sz1, sz2, R, T):
    # R = R.T  #####
    # splitRate = 1.0
    splitRate = 2.0
    MiddleR = R
    for i in range(3):
        for j in range(3):
            if i == j:
                MiddleR[i][j] = R[i][j]
            else:
                MiddleR[i][j] = R[i][j] / splitRate
    MiddleSz1, MiddleSz2 = sz1 / splitRate, sz2 / splitRate
    # MiddleSz1, MiddleSz2 = sz1, sz2
    MiddleT = T / splitRate
    MiddleT = np.reshape(MiddleT, [3, 1])
    MiddleRT = np.concatenate([MiddleR, MiddleT], axis=1)
    lower = np.reshape(np.array([0, 0, 0, 1]), [1, 4])
    MiddleRT = np.concatenate([MiddleRT, lower], axis=0)
    sz1Matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.0 / MiddleSz1, 0], [0, 0, 0, 1]]
    )
    sz2Matrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, MiddleSz2, 0], [0, 0, 0, 1]]
    )
    R1 = np.dot(sz1Matrix, MiddleRT)
    MiddleM = np.dot(R1, sz2Matrix)
    MiddleM = MiddleM[:3, :]
    # print(MiddleM)

    return MiddleM


if __name__ == "__main__":
    # M = np.load("chairDeskSave.npy")
    M = np.load("./M/antinous_GT.npy")
    print(M)
    # M = np.load("meetingRoomSave.npy")
    sz1, sz2, R12 = calcSz1Sz2(M)
    t12 = calcT12(M)
    MiddleM = calcMiddleM(sz1, sz2, R12, t12)
    # calcM(sz1, sz2, R12, t12)
    M = np.load("./M/antinous_GT.npy")

    print(MiddleM)
    # print(MiddleM - M)
    np.save("./M/middleM", MiddleM)
