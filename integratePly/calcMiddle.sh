#!/bin/bash
set -e
python calcMiddleM.py
python createNewPly.py
#python renderingGPU.py
python rendering.py
