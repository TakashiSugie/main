#!/bin/bash
#npy plyfromnpy
#matching 2d→3d LR create
#LFmesh2でやると警告が出るけど気にしない
#Matching LR createNewPly rendering
#makeNpy plyFromNpy FP3d_3d
set -e
python ./libs/mymkdir.py
# python ../WithMidas/midasOnly.py
python ../onlyAdabins/infer.py
# python alpha.py
# python makeNpy.py
# python plyFromNpy.py
python plyFromImg.py
python Matching.py
python FP2d_3d.py
python LR.py
python createNewPly.py
#python renderingGPU.py
python rendering.py
