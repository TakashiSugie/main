import numpy as np


def calcNorm(R):
    norm0 = np.linalg.norm(R, 2, axis=0)
    norm1 = np.linalg.norm(R, 2, axis=1)
    return norm0, norm1


if __name__ == "__main__":
    # M = np.load("chairDesk23_0_80.npy")
    M = np.load("antinous_0_80.npy")
    MAEMin = 100
    count = 0
    SR = M[:, 0:3].T
    T = M[:, 3]
    sz1 = 1
    sz2 = 1
    print(SR)
    norm0, norm1 = calcNorm(SR)
    print(norm0)
    print(norm1)
    while True:
        szInv1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0 / sz1]])
        szInv2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, sz2]])
        R1 = np.dot(szInv1, SR)
        R12 = np.dot(R1, szInv2)
        norm0, norm1 = calcNorm(R12)
        MAE = np.array([[1, 1, 1], [1, 1, 1]]) - np.array([norm0, norm1])
        # print(MAE.shape)
        loss = np.linalg.norm(MAE, 1)
        # loss = np.sum(np.abs(MAE))
        # print(np.sum(np.abs(MAE)))
        # print(count, loss)
        count += 1
        sz1 += 1
        sz2 += 1
        if MAEMin > loss:
            MAEMin = loss
        elif MAEMin <= loss:
            print("this loss is minimum")
            break

    print(R12)
    print(norm0)
    print(norm1)
