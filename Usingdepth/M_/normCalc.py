import numpy as np

if __name__ == "__main__":
    M = np.load("chairDesk23_0_80.npy")
    # M = np.load("antinous_0_80.npy")
    # M_1=M[:3][0]
    # M_2=M[:3][1]
    # M_3 = M[:3][2]
    SR = M[:, 0:3].T
    T = M[:, 3]
    print(SR, "\n")
    # szInv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 3 / 4]])
    szInv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0 / 400]])
    # szInv2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.2]])
    # szInv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0 / 2.2283]])
    # R = SR

    R = np.dot(szInv, SR)
    szInv2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 400]])

    R = np.dot(R, szInv2)
    print(R, "\n")
    # szInv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])

    # R = np.dot(SR, szInv)
    # print(R.shape)
    norm0 = np.linalg.norm(R, 2, axis=0)
    norm1 = np.linalg.norm(R, 2, axis=1)
    print(norm0)
    print(norm1)
