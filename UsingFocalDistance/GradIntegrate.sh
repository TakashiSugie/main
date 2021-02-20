#!/bin/bash
set -e
python ./libs/mymkdir.py
python gradM_ver2.py
python plyIntegrate.py
python rendering.py
