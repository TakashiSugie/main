# packages
import numpy as np
import copy as cp
from libs.variable import S_Z1, S_Z2

# parameters（ground truth)
S_Z1 = 1.0
S_Z2 = 1.0

R_X1 = 0.10
R_Y1 = 0.15
R_Z1 = 0.10
T_X1 = 0.15
T_Y1 = 0.10
T_Z1 = 0.15

# constants
F = 1.2
C_X = 0.5
C_Y = 0.5

S = 1000.0

# number of points
NUM_PT = 128

# step size
# STEP_SIZE = 0.001
STEP_SIZE = 0.0000000001

# world to camera
def w2c(p_w, r_x, r_y, r_z, t_x, t_y, t_z):
    p_c = np.zeros(p_w.shape)
    for i in range(NUM_PT):
        # rotation
        p_c[i, 0] = p_w[i, 0] - r_z * p_w[i, 1] + r_y * p_w[i, 2]
        p_c[i, 1] = r_z * p_w[i, 0] + p_w[i, 1] - r_x * p_w[i, 2]
        p_c[i, 2] = -r_y * p_w[i, 0] + r_x * p_w[i, 1] + p_w[i, 2]
        # translation
        p_c[i, 0] += t_x
        p_c[i, 1] += t_y
        p_c[i, 2] += t_z
    return p_c


# camera to image
def c2i(p_c, s):
    p_i = np.zeros(p_c.shape)
    for i in range(NUM_PT):
        # perspective projection with z scaling
        p_i[i, 0] = F * p_c[i, 0]
        p_i[i, 1] = F * p_c[i, 1]
        p_i[i, 0] /= p_c[i, 2]
        p_i[i, 0] += C_X
        p_i[i, 1] /= p_c[i, 2]
        p_i[i, 1] += C_Y
        p_i[i, 2] = p_c[i, 2] / s

    return p_i


def calcParameter():
    # generate depth image (u, v, \zeta)
    p_i1_org = c2i(p_c1_org, S_Z1)
    p_i2_org = c2i(p_c2_org, S_Z2)

    # initial guess
    r_x1_est = 0.0
    r_y1_est = 0.0
    r_z1_est = 0.0
    t_x1_est = 0.0
    t_y1_est = 0.0
    t_z1_est = 0.0

    r_x1_est = 0.127
    r_y1_est = -0.116
    r_z1_est = 0.045
    t_x1_est = 0.825
    t_y1_est = 0.907
    t_z1_est = 0.082

    # depth rate
    d = S_Z2 / S_Z1

    # solve non-linear least square
    for k in range(100000):
        # initialization of grad
        r_x1_grad = r_y1_grad = r_z1_grad = 0
        t_x1_grad = t_y1_grad = t_z1_grad = 0

        for i in range(NUM_PT):
            alpha = p_i1_org[i, 2] * (p_i1_org[i, 0] - C_X)
            alpha -= (
                d
                * p_i2_org[i, 2]
                * (
                    p_i2_org[i, 0]
                    - p_i2_org[i, 1] * r_z1_est
                    - C_X
                    + C_Y * r_z1_est
                    + F * r_y1_est
                )
            )
            alpha -= F * t_x1_est
            beta = p_i1_org[i, 2] * (p_i1_org[i, 1] - C_Y)
            beta -= (
                d
                * p_i2_org[i, 2]
                * (
                    p_i2_org[i, 0] * r_z1_est
                    + p_i2_org[i, 1]
                    - C_X * r_z1_est
                    - C_Y
                    - F * r_x1_est
                )
            )
            beta -= F * t_y1_est

            gamma = p_i1_org[i, 2] * F
            gamma -= (
                d
                * p_i2_org[i, 2]
                * (
                    -p_i2_org[i, 0] * r_y1_est
                    + p_i2_org[i, 1] * r_x1_est
                    + C_X * r_y1_est
                    - C_Y * r_x1_est
                    + F
                )
            )
            gamma -= F * t_z1_est

            r_x1_grad += (
                F * d * p_i2_org[i, 2] * beta
                + d * p_i2_org[i, 2] * (-p_i2_org[i, 1] + C_Y) * gamma
            )
            r_y1_grad += (
                -F * d * p_i2_org[i, 2] * alpha
                + d * p_i2_org[i, 2] * (p_i2_org[i, 0] - C_X) * gamma
            )
            r_z1_grad += (
                d * p_i2_org[i, 2] * (p_i2_org[i, 1] - C_Y) * alpha
                + d * p_i2_org[i, 2] * (-p_i2_org[i, 0] + C_X) * beta
            )
            t_x1_grad += -F * alpha
            t_y1_grad += -F * beta
            t_z1_grad += -F * gamma

        #     print(r_z1_est)
        #     break
        # break

        # gradient descend
        r_x1_est -= STEP_SIZE / F * r_x1_grad
        r_y1_est -= STEP_SIZE / F * r_y1_grad
        r_z1_est -= STEP_SIZE / F * r_z1_grad
        t_x1_est -= STEP_SIZE / F * t_x1_grad
        t_y1_est -= STEP_SIZE / F * t_y1_grad
        t_z1_est -= STEP_SIZE / F * t_z1_grad

        # Broyden–Fletcher–Goldfarb–Shanno algorithm will be implemented
        print(t_z1_est)
        # print(t_z1_est * S_Z1)
    return [r_x1_est, r_y1_est, r_z1_est, t_x1_est, t_y1_est, t_z1_est]


def makeM(paraList):
    M = np.zeros((3, 4))
    rx, ry, rz, tx, ty, tz = paraList
    M = np.array([[1, -rz, ry, tx], [rz, 1, -rx, ty], [-ry, rx, 1, tz]])
    np.save("./M/GradM_%d_%d" % (S_Z1, S_Z2), M)
    return M


def initParam():
    global F, C_X, C_Y
    F = 1462.857142857143
    C_X = 256.0
    C_Y = 256.0


if __name__ == "__main__":

    # # generate points in world coordinate system
    p_w_org = np.random.randn(NUM_PT, 3)

    # # # generate points in camera cordinate systems 1 and 2
    # p_c1_org = w2c(p_w_org, R_X1, R_Y1, R_Z1, T_X1, T_Y1, T_Z1)
    # p_c2_org = cp.deepcopy(p_w_org)

    p_c1_org = np.load("./FP_3d/input_Cam000.npy") / S
    p_c2_org = np.load("./FP_3d/input_Cam080.npy") / S
    initParam()

    # p_c1_org = np.load("2.npy") / 10000.0
    # p_c2_org = np.load("3.npy") / 10000.0
    NUM_PT = p_c1_org.shape[0]
    # print(p_c1_org.shape)
    # print(p_c2_org.shape)

    paraList = calcParameter()

    rx, ry, rz, tx, ty, tz = paraList
    # print(rx, ry, rz, tx, ty, tz)
    M = makeM(paraList)
    print(M)

