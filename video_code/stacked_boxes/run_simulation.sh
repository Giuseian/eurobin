#!/bin/bash

set -e  # stop se qualcosa fallisce

kill_cartesio() {
    echo "Stopping Cartesio with SIGINT..."
    pkill -INT -f "ros2 launch kyon_cartesio kyon.launch" || true

    while pgrep -f "ros2 launch kyon_cartesio kyon.launch" > /dev/null; do
        sleep 1
    done

    echo "Cartesio stopped."
}

start_cartesio() {
    echo "Starting Cartesio..."
    ros2 launch kyon_cartesio kyon.launch &

    # aspetta che sia pronto
    sleep 5
}

echo "Running screen.py"
python3 screen.py

echo "Running grasp_box_yz.py..."
python3 grasp_box_yz.py

echo "Running screen.py"
python3 screen.py

echo "Running place_box.py..."
python3 place_box.py

echo "Running screen.py"
python3 screen.py

echo "Running place_box_2.py..."
python3 place_box_2.py

echo "Running screen.py"
python3 screen.py

# STOP CARTESIO
kill_cartesio

sleep 10

echo "Running homing.py..."
python homing.py

sleep 45

echo "Running screen.py"
python3 screen.py

# RESTART CARTESIO
start_cartesio

echo "Running grasp_box_xy.py..."
python3 grasp_box_xy.py

echo "Running screen.py"
python3 screen.py

echo "Pipeline completata!"