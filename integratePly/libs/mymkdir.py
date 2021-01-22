import os


def my_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    dirList = [
        "M",
        "FPImg",
        "FP_2d",
        "FP_3d",
        "mesh",
        "depth",
        "FPManual/npy",
        "FPManual/img",
    ]
    for dir in dirList:
        my_mkdir(dir)
