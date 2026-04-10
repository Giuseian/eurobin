#!/bin/bash

set -e  # stop se qualcosa fallisce

echo "Running screen.py"
python3 screen.py

echo "Running grasp_box_yz.py..."
python3 grasp_box_yz.py

echo "Running screen.py"
python3 screen.py

echo "Running place_box.py..."
python3 place_box.py

echo "Running place_box_2.py..."
python3 place_box_2.py

echo "Running homing.py"
python3 homing.py

echo "Running screen.py"
python3 screen.py

echo "Running grasp_box_xy.py..."
python3 grasp_box_xy.py

echo "Running screen.py"
python3 screen.py

echo "Pipeline completata!"