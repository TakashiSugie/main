#!/bin/bash
set -e
python calcMiddleM.py
# python createNewPly.py
python plyIntegrate.py
#python renderingGPU.py
python rendering.py
