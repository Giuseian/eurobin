# Stacked boxes

## Manipulation Pipeline

First, in `kyon_world.launch` set the `kyon_stacked.world`.

Run the simulation:

```bash
# Terminal 1
ros2 launch kyon_gazebo kyon_world.launch arms:=true dagana:=true wheels:=true camera_front_up:=true

# Terminal 2
xbot2-core  -S -C ~/xbot2_ws/src/iit-kyon-ros-pkg/kyon_config/kyon_basic.yaml

# Terminal 3
xbot2_gui   # run the homing

# Terminal 4
ros2 launch kyon_cartesio kyon.launch
```

Then, start the manipulation pipeline with:

```bash
cd ~/image/stacked_boxes

./run_simulation.sh
```

This code does the following things

```bash
# Grasp the red box (first align y and z, then the x)
python3 grasp_box_yz.py

# Place the red box
python3 place_box.py

# Lift the hands (to avoid collision with the following homing)
python3 place_box_2.py

# Run the homing
python homing.py

# Grasp the green box (first align the x and y, then the z)
python3 grasp_box_xy.py
```

It also calls the `screen.py` at the right moments.

## Scene Perception

The `scene.py` can be used this way:

```bash
python scene.py scena.json
```

where `scena.json` is something like (for the stacked boxes):

```bash
{
  "scene_description": {
    "objects": [
      {
        "name": "box_red_001",
        "category": "box",
        "color": "red"
      },
      {
        "name": "box_green_001",
        "category": "box",
        "color": "green"
      }
    ],
    "end_effectors": [
      {
        "name": "end_effector_left",
        "grasping": "nothing"
      },
      {
        "name": "end_effectector_right",
        "grasping": "nothing"
      }
    ],
    "spatial_relationships": [
      {
        "subject": "box_red_001",
        "relation": "on top of",
        "object": "box_green_001"
      }
    ]
  }
}
```

And the final output is written in another file that I call `scena_complete.json`:

```bash
{
  "scene_description": {
    "objects": [
      {
        "name": "box_red_001",
        "category": "box",
        "color": "red",
        "location": "zone B",
        "sides": {
          "left": "accessible",
          "right": "accessible",
          "front": "accessible",
          "back": "not reachable",
          "top": "accessible",
          "bottom": "blocked by box_green_001"
        }
      },
      {
        "name": "box_green_001",
        "category": "box",
        "color": "green",
        "location": "zone B",
        "sides": {
          "left": "accessible",
          "right": "accessible",
          "front": "accessible",
          "back": "not reachable",
          "top": "blocked by box_red_001",
          "bottom": "blocked by the table"
        }
      }
    ],
    "end_effectors": [
      {
        "name": "end_effector_left",
        "grasping": "nothing"
      },
      {
        "name": "end_effectector_right",
        "grasping": "nothing"
      }
    ],
    "spatial_relationships": [
      {
        "subject": "box_red_001",
        "relation": "on top of",
        "object": "box_green_001"
      }
    ]
  }
}
```

The dimensions of the objects are written directly inside `scene.py`, while the positions are read only from Gazebo. The code adds the locations of the objects together with the sides accessibility.
