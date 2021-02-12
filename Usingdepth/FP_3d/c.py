# packages
import numpy as np
import copy as cp

# parameters（ground truth)
S_Z1 = 10.0  # カメラ1のzスケール
S_Z2 = 10.0  # カメラ2のzスケール
R_X1 = 0.1  # カメラ1のx回転
R_Y1 = 0.1  # カメラ1のy回転
R_Z1 = 0.1  # カメラ1のz回転
T_X1 = 0.1  # カメラ1のx並進
T_Y1 = 0.1  # カメラ1のy並進
T_Z1 = 0.1  # カメラ1のz並進

# constants
F = 1.2  # 焦点距離
C_X = 0.5  # 光学中心
C_Y = 0.5  # 光学中心

# number of points (点の数)
NUM_PT = 16

# step size (最急降下法のステップサイズ)
STEP_SIZE = 0.001

# ワールド　→　カメラ
def w2c(p_w, r_x, r_y, r_z, t_x, t_y, t_z):
    # p_w : ワールドのx, y, z座標
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


# カメラ　→　画像
def c2i(p_c, s):
    # p_c : カメラ
    p_i = np.zeros(p_c.shape)
    for i in range(NUM_PT):
        # perspective projection with z scaling
        p_i[i, 0] = F * p_c[i, 0]
        p_i[i, 1] = F * p_c[i, 1]
        p_i[i, 0] /= p_c[i, 2]
        p_i[i, 0] += C_X
        p_i[i, 1] /= p_c[i, 2]
        p_i[i, 1] += C_Y
        p_i[i, 2] = p_c[i, 2] / s  # p_i は \zeta p_c は Z

    return p_i


def calcParameter():
    # R と t の初期値
    r_x1_est = 0.0
    r_y1_est = 0.0
    r_z1_est = 0.0
    t_x1_est = 0.0
    t_y1_est = 0.0
    t_z1_est = 0.0

    # Gradient descend
    for k in range(1000):
        # initialization of grad
        r_x1_grad = r_y1_grad = r_z1_grad = 0
        t_x1_grad = t_y1_grad = t_z1_grad = 0

        for i in range(NUM_PT):
            alpha = p_i1_org[i, 2] * (p_i1_org[i, 0] - C_X)
            alpha -= p_i2_org[i, 2] * (
                p_i2_org[i, 0]
                - p_i2_org[i, 1] * r_z1_est
                - C_X
                + C_Y * r_z1_est
                + F * r_y1_est
            )
            alpha -= F * t_x1_est

            beta = p_i1_org[i, 2] * (p_i1_org[i, 1] - C_Y)
            beta -= p_i2_org[i, 2] * (
                p_i2_org[i, 0] * r_z1_est
                + p_i2_org[i, 1]
                - C_X * r_z1_est
                - C_Y
                - F * r_x1_est
            )
            beta -= F * t_y1_est

            gamma = p_i1_org[i, 2] * F
            gamma -= p_i2_org[i, 2] * (
                -p_i2_org[i, 0] * r_y1_est
                + p_i2_org[i, 1] * r_x1_est
                + C_X * r_y1_est
                - C_Y * r_x1_est
                + F
            )
            gamma -= F * t_z1_est

            r_x1_grad += (
                F * p_i2_org[i, 2] * beta
                + (-p_i2_org[i, 2] * p_i2_org[i, 1] + p_i2_org[i, 2] * C_Y) * gamma
            )
            r_y1_grad += (
                -F * p_i2_org[i, 2] * alpha
                + (p_i2_org[i, 2] * p_i2_org[i, 0] - p_i2_org[i, 2] * C_X) * gamma
            )
            r_z1_grad += (
                p_i2_org[i, 2] * p_i2_org[i, 1] - p_i2_org[i, 2] * C_Y
            ) * alpha + (-p_i2_org[i, 2] * p_i2_org[i, 0] + p_i2_org[i, 2] * C_X) * beta
            t_x1_grad += -F * alpha
            t_y1_grad += -F * beta
            t_z1_grad += -F * gamma

        r_x1_est -= STEP_SIZE / F * r_x1_grad
        r_y1_est -= STEP_SIZE / F * r_y1_grad
        r_z1_est -= STEP_SIZE / F * r_z1_grad
        t_x1_est -= STEP_SIZE / F * t_x1_grad
        t_y1_est -= STEP_SIZE / F * t_y1_grad
        t_z1_est -= STEP_SIZE / F * t_z1_grad

        # print(r_z1_est)
        # print(t_z1_est)
    return [r_x1_est, r_y1_est, r_z1_est, t_x1_est, t_y1_est, t_z1_est]


def makeM(paraList):
    M = np.zeros((3, 4))
    rx, ry, rz, tx, ty, tz = paraList
    M = np.array([[1, -rz, ry, tx], [rz, 1, -rx, ty], [-ry, rx, 1, tz]])
    np.save("GradM", M)
    return M


if __name__ == "__main__":

    # generate points in world coordinate system
    # p_w_org = np.random.randn(NUM_PT, 3)

    # # generate points in camera cordinate systems 1 and 2
    # p_c1_org = w2c(p_w_org, R_X1, R_Y1, R_Z1, T_X1, T_Y1, T_Z1)
    # p_c2_org = cp.deepcopy(p_w_org)

    # print(p_c1_org.shape)
    # print(p_c2_org.shape)

    p_c1_org = np.load("input_Cam000.npy")
    p_c2_org = np.load("input_Cam080.npy")
    NUM_PT = p_c1_org.shape[0]
    print(p_c1_org.shape)
    print(p_c2_org.shape)

    # generate depth image (u, v, \zeta)
    p_i1_org = c2i(p_c1_org, S_Z1)
    p_i2_org = c2i(p_c2_org, S_Z2)
    paraList = calcParameter()

    rx, ry, rz, tx, ty, tz = paraList
    print(rx, ry, rz, tx, ty, tz)
    M = makeM(paraList)
    print(M)
