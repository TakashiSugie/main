#!/bin/bash
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
python ./M/calcMiddleM.py
# python createNewPly.py
python plyIntegrate.py
#python renderingGPU.py
python rendering.py
