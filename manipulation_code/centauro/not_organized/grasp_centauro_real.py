#!/usr/bin/env python3

"""
Centauro dual-arm grasp for the real robot.

Run from this folder:

    python3 grasp_centauro_real.py --ros-args \
      -p poses_file:=poses.txt \
      -p approach_mode:=yz \
      -p grasp_offset_x:=-0.05 \
      -p grasp_offset_y:=0.0 \
      -p grasp_offset_z:=0.1 \
      -p phase1_split_fraction:=0.5

What it does:
    1. Reads the box pose and Centauro base pose from a text file.
    2. Uses both position and orientation, so the robot does not need yaw=0.
    3. Converts the box pose from WORLD to the Centauro base frame.
    4. Computes the box local axes in the Centauro base frame.
    5. Moves the two Dagana end-effectors to the sides of the box.
    6. Closes symmetrically along the box local Y axis.
    7. Lifts the box after contact.

Input pose file format:
    The file is read from the same folder by default. Values are whitespace
    separated. Quaternions are ordered as qx qy qz qw.

        box.position: 0.954 0.012 1.290
        box.orientation: 0.0 0.0 0.0 1.0
        robot.position: 0.067 0.001 0.864
        robot.orientation: 0.0 0.0 0.0 1.0

Most useful ROS parameters:
    poses_file:
        Text file containing box and robot poses. Relative paths are resolved
        from this script folder.
    approach_mode:
        yz, xz, or xy. Selects which axes are solved in phase 1 vs phase 2.
    grasp_offset_x/y/z:
        Offset of the grasp center relative to the box center, expressed in
        the box local frame.
    box_width:
        Distance used to place the two grippers on opposite sides of the box.
    phase1_clearance:
        Extra distance from the box side before closing.
    phase3_extra_squeeze:
        Extra inward motion after first contact.
    lift_z_phase4:
        Vertical lift after grasping, expressed in the Centauro base frame.
    align_orientation_to_box_yaw:
        If true, rotates the commanded gripper orientation by the box yaw in
        the Centauro base frame.
    d1_start_x/y/z, d2_start_x/y/z:
        Initial safe pose for each Dagana, expressed in Centauro base frame.
    d1_qx/qy/qz/qw, d2_qx/qy/qz/qw:
        Base gripper orientations used by the Cartesian action goals.
    time_start_pose, time_phase1, time_phase2, time_phase3, time_phase4:
        Motion duration for each phase.
"""

from pathlib import Path
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class CentauroRealGrasp(Node):

    def __init__(self):
        super().__init__('centauro_real_grasp')

        self.client_1 = ActionClient(self, ReachPose, '/arm1_8/reach')
        self.client_2 = ActionClient(self, ReachPose, '/arm2_8/reach')

        self.declare_parameter('poses_file', 'poses.txt')

        self.declare_parameter('d1_start_x', 0.53)
        self.declare_parameter('d1_start_y', 0.3)
        self.declare_parameter('d1_start_z', 0.29)
        self.declare_parameter('d2_start_x', 0.53)
        self.declare_parameter('d2_start_y', -0.3)
        self.declare_parameter('d2_start_z', 0.29)

        self.declare_parameter('d1_qx', 0.5)
        self.declare_parameter('d1_qy', 0.5)
        self.declare_parameter('d1_qz', 0.5)
        self.declare_parameter('d1_qw', -0.5)
        self.declare_parameter('d2_qx', -0.5)
        self.declare_parameter('d2_qy', -0.5)
        self.declare_parameter('d2_qz', -0.5)
        self.declare_parameter('d2_qw', 0.5)

        self.declare_parameter('box_width', 0.38)
        self.declare_parameter('phase1_clearance', 0.05)
        self.declare_parameter('phase1_extra_margin_x', 0.0)
        self.declare_parameter('phase1_extra_margin_y', 0.0)
        self.declare_parameter('phase1_extra_margin_z', 0.0)
        self.declare_parameter('approach_mode', 'yz')
        self.declare_parameter('phase1_split_fraction', 0.5)

        self.declare_parameter('grasp_offset_x', 0.0)
        self.declare_parameter('grasp_offset_y', 0.0)
        self.declare_parameter('grasp_offset_z', 0.0)
        self.declare_parameter('align_orientation_to_box_yaw', True)

        self.declare_parameter('phase3_extra_squeeze', 0.0)
        self.declare_parameter('lift_z_phase4', 0.20)

        self.declare_parameter('time_start_pose', 5.0)
        self.declare_parameter('time_phase1', 10.0)
        self.declare_parameter('time_phase2', 10.0)
        self.declare_parameter('time_phase3', 10.0)
        self.declare_parameter('time_phase4', 10.0)

        self.poses_file = self.get_parameter('poses_file').value

        self.d1_start_x = self.get_parameter('d1_start_x').value
        self.d1_start_y = self.get_parameter('d1_start_y').value
        self.d1_start_z = self.get_parameter('d1_start_z').value
        self.d2_start_x = self.get_parameter('d2_start_x').value
        self.d2_start_y = self.get_parameter('d2_start_y').value
        self.d2_start_z = self.get_parameter('d2_start_z').value

        self.d1_qx = self.get_parameter('d1_qx').value
        self.d1_qy = self.get_parameter('d1_qy').value
        self.d1_qz = self.get_parameter('d1_qz').value
        self.d1_qw = self.get_parameter('d1_qw').value
        self.d2_qx = self.get_parameter('d2_qx').value
        self.d2_qy = self.get_parameter('d2_qy').value
        self.d2_qz = self.get_parameter('d2_qz').value
        self.d2_qw = self.get_parameter('d2_qw').value

        self.box_width = self.get_parameter('box_width').value
        self.phase1_clearance = self.get_parameter('phase1_clearance').value
        self.phase1_extra_margin_x = self.get_parameter('phase1_extra_margin_x').value
        self.phase1_extra_margin_y = self.get_parameter('phase1_extra_margin_y').value
        self.phase1_extra_margin_z = self.get_parameter('phase1_extra_margin_z').value
        self.approach_mode = str(self.get_parameter('approach_mode').value).strip().lower()
        self.phase1_split_fraction = self.get_parameter('phase1_split_fraction').value

        self.grasp_offset_x = self.get_parameter('grasp_offset_x').value
        self.grasp_offset_y = self.get_parameter('grasp_offset_y').value
        self.grasp_offset_z = self.get_parameter('grasp_offset_z').value
        self.align_orientation_to_box_yaw = self.get_parameter('align_orientation_to_box_yaw').value

        self.phase3_extra_squeeze = self.get_parameter('phase3_extra_squeeze').value
        self.lift_z_phase4 = self.get_parameter('lift_z_phase4').value

        self.time_start_pose = self.get_parameter('time_start_pose').value
        self.time_phase1 = self.get_parameter('time_phase1').value
        self.time_phase2 = self.get_parameter('time_phase2').value
        self.time_phase3 = self.get_parameter('time_phase3').value
        self.time_phase4 = self.get_parameter('time_phase4').value

        if self.approach_mode not in ('yz', 'xz', 'xy'):
            raise ValueError(
                f'approach_mode="{self.approach_mode}" non valido. Usa: yz, xz, xy'
            )

        self.box_position = None
        self.box_orientation = None
        self.robot_position = None
        self.robot_orientation = None

    # ------------------------------------------------------------------
    # Quaternion and vector utilities
    # ------------------------------------------------------------------
    def quat_normalize(self, q):
        x, y, z, w = q
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm, w / norm)

    def quat_conjugate(self, q):
        x, y, z, w = q
        return (-x, -y, -z, w)

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def quat_rotate_vector(self, q, v):
        q = self.quat_normalize(q)
        rotated = self.quat_multiply(
            self.quat_multiply(q, (v[0], v[1], v[2], 0.0)),
            self.quat_conjugate(q)
        )
        return (rotated[0], rotated[1], rotated[2])

    def quat_to_yaw(self, q):
        x, y, z, w = self.quat_normalize(q)
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quat(self, yaw):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def vector_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def vector_sub(self, a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def vector_scale(self, v, scale):
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    def world_position_to_robot_base(self, world_position):
        if self.robot_position is None or self.robot_orientation is None:
            self.get_logger().error('Pose del robot non disponibile.')
            return None

        delta_world = self.vector_sub(world_position, self.robot_position)
        robot_inv = self.quat_conjugate(self.robot_orientation)
        return self.quat_rotate_vector(robot_inv, delta_world)

    def box_orientation_in_robot_base(self):
        if self.box_orientation is None or self.robot_orientation is None:
            self.get_logger().error('Orientazione box/robot non disponibile.')
            return None

        robot_inv = self.quat_conjugate(self.robot_orientation)
        return self.quat_normalize(self.quat_multiply(robot_inv, self.box_orientation))

    # ------------------------------------------------------------------
    # Pose file input
    # ------------------------------------------------------------------
    def resolve_poses_path(self):
        path = Path(str(self.poses_file)).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parent / path

    def parse_pose_file(self):
        path = self.resolve_poses_path()
        if not path.is_file():
            raise FileNotFoundError(f'File pose non trovato: {path}')

        data = {}
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue

            if ':' in line:
                key, values = line.split(':', 1)
            elif '=' in line:
                key, values = line.split('=', 1)
            else:
                raise ValueError(f'{path}:{line_number}: usa "chiave: valori"')

            key = key.strip().lower().replace('.', '_')
            try:
                data[key] = tuple(float(value) for value in values.split())
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: valori numerici non validi') from exc

        aliases = {
            'box_position': ('box_position',),
            'box_orientation': ('box_orientation', 'box_quaternion'),
            'robot_position': ('robot_position', 'centauro_position'),
            'robot_orientation': ('robot_orientation', 'centauro_orientation', 'robot_quaternion'),
        }

        parsed = {}
        for canonical_key, possible_keys in aliases.items():
            for possible_key in possible_keys:
                if possible_key in data:
                    parsed[canonical_key] = data[possible_key]
                    break

        required_lengths = {
            'box_position': 3,
            'box_orientation': 4,
            'robot_position': 3,
            'robot_orientation': 4,
        }
        missing = [key for key in required_lengths if key not in parsed]
        if missing:
            raise ValueError(f'{path}: chiavi mancanti: {", ".join(missing)}')

        for key, expected_length in required_lengths.items():
            if len(parsed[key]) != expected_length:
                raise ValueError(
                    f'{path}: "{key}" deve avere {expected_length} valori, '
                    f'ne ha {len(parsed[key])}'
                )

        return parsed

    def read_input_poses(self):
        poses = self.parse_pose_file()

        self.box_position = poses['box_position']
        self.box_orientation = self.quat_normalize(poses['box_orientation'])
        self.robot_position = poses['robot_position']
        self.robot_orientation = self.quat_normalize(poses['robot_orientation'])

        self.log_pose('box', self.box_position, self.box_orientation)
        self.log_pose('robot', self.robot_position, self.robot_orientation)

    def log_pose(self, label, position, orientation):
        px, py, pz = position
        qx, qy, qz, qw = orientation
        yaw = self.quat_to_yaw(orientation)
        self.get_logger().info(
            f'{label} position: x={px:.6f}, y={py:.6f}, z={pz:.6f}'
        )
        self.get_logger().info(
            f'{label} orientation: q=({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}), '
            f'yaw={yaw:.6f} rad'
        )

    # ------------------------------------------------------------------
    # Grasp planner
    # ------------------------------------------------------------------
    def box_target_components(self):
        box_position_robot = self.world_position_to_robot_base(self.box_position)
        box_orientation_robot = self.box_orientation_in_robot_base()
        if box_position_robot is None or box_orientation_robot is None:
            return None

        box_x_axis_robot = self.quat_rotate_vector(box_orientation_robot, (1.0, 0.0, 0.0))
        box_y_axis_robot = self.quat_rotate_vector(box_orientation_robot, (0.0, 1.0, 0.0))
        box_z_axis_robot = self.quat_rotate_vector(box_orientation_robot, (0.0, 0.0, 1.0))

        grasp_offset_robot = self.vector_add(
            self.vector_add(
                self.vector_scale(box_x_axis_robot, self.grasp_offset_x),
                self.vector_scale(box_y_axis_robot, self.grasp_offset_y)
            ),
            self.vector_scale(box_z_axis_robot, self.grasp_offset_z)
        )
        grasp_position_robot = self.vector_add(box_position_robot, grasp_offset_robot)
        box_yaw_robot = self.quat_to_yaw(box_orientation_robot)

        self.get_logger().info(
            f'Box in base Centauro: x={box_position_robot[0]:.6f}, '
            f'y={box_position_robot[1]:.6f}, z={box_position_robot[2]:.6f}, '
            f'yaw={box_yaw_robot:.6f} rad'
        )
        self.get_logger().info(
            f'Grasp center in base Centauro: x={grasp_position_robot[0]:.6f}, '
            f'y={grasp_position_robot[1]:.6f}, z={grasp_position_robot[2]:.6f}'
        )

        return {
            'grasp_position': grasp_position_robot,
            'box_y_axis': box_y_axis_robot,
        }

    def split_axis_value(self, start_value, target_value):
        return start_value + self.phase1_split_fraction * (target_value - start_value)

    def compute_all_phase_targets(self):
        target = self.box_target_components()
        if target is None:
            return None

        grasp_position = target['grasp_position']
        box_y_axis = target['box_y_axis']
        half_box = self.box_width / 2.0

        start_d1_x = self.d1_start_x
        start_d1_y = self.d1_start_y
        start_d1_z = self.d1_start_z
        start_d2_x = self.d2_start_x
        start_d2_y = self.d2_start_y
        start_d2_z = self.d2_start_z

        pre_d1 = self.vector_add(
            grasp_position,
            self.vector_scale(box_y_axis, half_box + self.phase1_clearance)
        )
        pre_d2 = self.vector_add(
            grasp_position,
            self.vector_scale(box_y_axis, -half_box - self.phase1_clearance)
        )
        pre_d1_x, pre_d1_y, pre_d1_z = pre_d1
        pre_d2_x, pre_d2_y, pre_d2_z = pre_d2

        if self.approach_mode == 'yz':
            phase1_d1_x = self.split_axis_value(start_d1_x, pre_d1_x) + self.phase1_extra_margin_x
            phase1_d1_y = pre_d1_y + self.phase1_extra_margin_y
            phase1_d1_z = pre_d1_z + self.phase1_extra_margin_z
            phase1_d2_x = self.split_axis_value(start_d2_x, pre_d2_x) + self.phase1_extra_margin_x
            phase1_d2_y = pre_d2_y + self.phase1_extra_margin_y
            phase1_d2_z = pre_d2_z + self.phase1_extra_margin_z
            phase2_d1_x = pre_d1_x
            phase2_d1_y = phase1_d1_y
            phase2_d1_z = phase1_d1_z
            phase2_d2_x = pre_d2_x
            phase2_d2_y = phase1_d2_y
            phase2_d2_z = phase1_d2_z
        elif self.approach_mode == 'xz':
            phase1_d1_x = pre_d1_x + self.phase1_extra_margin_x
            phase1_d1_y = self.split_axis_value(start_d1_y, pre_d1_y) + self.phase1_extra_margin_y
            phase1_d1_z = pre_d1_z + self.phase1_extra_margin_z
            phase1_d2_x = pre_d2_x + self.phase1_extra_margin_x
            phase1_d2_y = self.split_axis_value(start_d2_y, pre_d2_y) + self.phase1_extra_margin_y
            phase1_d2_z = pre_d2_z + self.phase1_extra_margin_z
            phase2_d1_x = phase1_d1_x
            phase2_d1_y = pre_d1_y
            phase2_d1_z = phase1_d1_z
            phase2_d2_x = phase1_d2_x
            phase2_d2_y = pre_d2_y
            phase2_d2_z = phase1_d2_z
        else:
            phase1_d1_x = pre_d1_x + self.phase1_extra_margin_x
            phase1_d1_y = pre_d1_y + self.phase1_extra_margin_y
            phase1_d1_z = self.split_axis_value(start_d1_z, pre_d1_z) + self.phase1_extra_margin_z
            phase1_d2_x = pre_d2_x + self.phase1_extra_margin_x
            phase1_d2_y = pre_d2_y + self.phase1_extra_margin_y
            phase1_d2_z = self.split_axis_value(start_d2_z, pre_d2_z) + self.phase1_extra_margin_z
            phase2_d1_x = phase1_d1_x
            phase2_d1_y = phase1_d1_y
            phase2_d1_z = pre_d1_z
            phase2_d2_x = phase1_d2_x
            phase2_d2_y = phase1_d2_y
            phase2_d2_z = pre_d2_z

        contact_d1 = self.vector_add(grasp_position, self.vector_scale(box_y_axis, half_box))
        contact_d2 = self.vector_add(grasp_position, self.vector_scale(box_y_axis, -half_box))
        contact_d1_x, contact_d1_y = contact_d1[0], contact_d1[1]
        contact_d2_x, contact_d2_y = contact_d2[0], contact_d2[1]
        contact_d1_z = phase2_d1_z
        contact_d2_z = phase2_d2_z

        lift_d1 = self.vector_add(
            grasp_position,
            self.vector_scale(box_y_axis, half_box - self.phase3_extra_squeeze)
        )
        lift_d2 = self.vector_add(
            grasp_position,
            self.vector_scale(box_y_axis, -half_box + self.phase3_extra_squeeze)
        )
        lift_d1_x, lift_d1_y = lift_d1[0], lift_d1[1]
        lift_d2_x, lift_d2_y = lift_d2[0], lift_d2[1]
        lift_d1_z = contact_d1_z + self.lift_z_phase4
        lift_d2_z = contact_d2_z + self.lift_z_phase4

        phases = {
            'start_pose': (
                start_d1_x, start_d1_y, start_d1_z,
                start_d2_x, start_d2_y, start_d2_z
            ),
            'phase1': (
                phase1_d1_x, phase1_d1_y, phase1_d1_z,
                phase1_d2_x, phase1_d2_y, phase1_d2_z
            ),
            'phase2': (
                phase2_d1_x, phase2_d1_y, phase2_d1_z,
                phase2_d2_x, phase2_d2_y, phase2_d2_z
            ),
            'phase3': (
                contact_d1_x, contact_d1_y, contact_d1_z,
                contact_d2_x, contact_d2_y, contact_d2_z
            ),
            'phase4': (
                lift_d1_x, lift_d1_y, lift_d1_z,
                lift_d2_x, lift_d2_y, lift_d2_z
            ),
        }
        self.log_phase_targets(phases)
        return phases

    def log_phase_targets(self, phases):
        self.get_logger().info('=== Target grasp calcolati ===')
        self.get_logger().info(f'approach_mode = {self.approach_mode}')
        for phase_name, values in phases.items():
            x1, y1, z1, x2, y2, z2 = values
            self.get_logger().info(
                f'{phase_name}: d1=({x1:.6f}, {y1:.6f}, {z1:.6f}), '
                f'd2=({x2:.6f}, {y2:.6f}, {z2:.6f})'
            )

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def make_goal(self, x, y, z, qx, qy, qz, qw, time_s):
        goal = ReachPose.Goal()
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        goal.frames = [pose]
        goal.time = [float(time_s)]
        goal.incremental = False
        return goal

    def current_goal_orientations(self):
        d1_q = self.quat_normalize((self.d1_qx, self.d1_qy, self.d1_qz, self.d1_qw))
        d2_q = self.quat_normalize((self.d2_qx, self.d2_qy, self.d2_qz, self.d2_qw))

        if not self.align_orientation_to_box_yaw:
            return d1_q, d2_q

        box_orientation_robot = self.box_orientation_in_robot_base()
        if box_orientation_robot is None:
            return d1_q, d2_q

        box_yaw_robot = self.quat_to_yaw(box_orientation_robot)
        yaw_q = self.yaw_to_quat(box_yaw_robot)
        return (
            self.quat_normalize(self.quat_multiply(yaw_q, d1_q)),
            self.quat_normalize(self.quat_multiply(yaw_q, d2_q)),
        )

    def send_two_goals_and_wait(self, phase_name, goal1, goal2):
        self.get_logger().info(f'=== Starting {phase_name} ===')
        self.client_1.wait_for_server()
        self.client_2.wait_for_server()

        fut_send1 = self.client_1.send_goal_async(goal1)
        fut_send2 = self.client_2.send_goal_async(goal2)
        rclpy.spin_until_future_complete(self, fut_send1)
        rclpy.spin_until_future_complete(self, fut_send2)

        gh1 = fut_send1.result()
        gh2 = fut_send2.result()
        if gh1 is None or not gh1.accepted:
            raise RuntimeError(f'{phase_name}: goal arm1_8 rejected.')
        if gh2 is None or not gh2.accepted:
            raise RuntimeError(f'{phase_name}: goal arm2_8 rejected.')

        fut_res1 = gh1.get_result_async()
        fut_res2 = gh2.get_result_async()
        rclpy.spin_until_future_complete(self, fut_res1)
        rclpy.spin_until_future_complete(self, fut_res2)

        if fut_res1.result() is None:
            raise RuntimeError(f'{phase_name}: no result arm1_8.')
        if fut_res2.result() is None:
            raise RuntimeError(f'{phase_name}: no result arm2_8.')
        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase(self, phase_name, phase_targets, time_s):
        x1, y1, z1, x2, y2, z2 = phase_targets
        d1_q, d2_q = self.current_goal_orientations()
        goal1 = self.make_goal(x1, y1, z1, *d1_q, time_s=time_s)
        goal2 = self.make_goal(x2, y2, z2, *d2_q, time_s=time_s)
        self.send_two_goals_and_wait(phase_name, goal1, goal2)

    def execute(self):
        self.get_logger().info('Lettura pose da file...')
        self.read_input_poses()

        phases = self.compute_all_phase_targets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare i target assoluti delle fasi.')

        self.run_phase('START_POSE', phases['start_pose'], self.time_start_pose)
        self.run_phase(f'PHASE 1 - {self.approach_mode.upper()}', phases['phase1'], self.time_phase1)
        self.run_phase('PHASE 2 - COMPLETE_REMAINING_AXIS', phases['phase2'], self.time_phase2)
        self.run_phase('PHASE 3 - CLOSE_ON_BOX_LOCAL_Y', phases['phase3'], self.time_phase3)
        self.run_phase('PHASE 4 - SQUEEZE_AND_LIFT', phases['phase4'], self.time_phase4)
        self.get_logger().info('Grasp completed.')


def main():
    rclpy.init()
    node = CentauroRealGrasp()

    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f'Execution failed: {exc}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
