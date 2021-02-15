#!/bin/bash
set -e
python ./libs/mymkdir.py
python ./M/calcMiddleM.py
python plyIntegrate.py
python rendering.py
