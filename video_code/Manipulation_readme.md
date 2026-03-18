# 2 stacked boxes

```bash
python grasp_box_yz.py

python place_box.py

homing

python grasp_box_xy.py
```


# 2 boxes aside and 1 cup
```bash
python dagana.py set 0 1.5

python grasp_cup.py

python place_cup.py

python dagana.py set 0 0

python place_cup2.py

homing

python push_box_xyz.py

homing

python grasp_box_yz.py
```