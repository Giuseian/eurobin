#!/usr/bin/env python3

"""
Clean Centauro dual-arm place script.

Run from this folder:

    python3 place_centauro_clean.py --ros-args \
      -p box_name:=box_red_001 \
      -p place_world_x:=0.95 \
      -p place_world_y:=0.0 \
      -p place_world_z:=0.9 \
      -p place_world_yaw:=0.0

What it does:
    1. Reads the current box pose, Centauro base pose, and Dagana link poses
       from Gazebo.
    2. Uses the Centauro base orientation, so the robot can be yawed relative
       to the table/world.
    3. Interprets place_world_x/y/z as the desired final box position in WORLD.
    4. Converts that world target into the Centauro base frame.
    5. Moves the grasped box in X/Y first.
    6. Moves the box to the requested Z.
    7. Opens the two Dagana end-effectors along the final place lateral axis.

Most useful ROS parameters:
    box_name:
        Gazebo entity name of the box being placed.
    place_world_x/y/z:
        Desired final box position in WORLD/table coordinates.
    place_world_yaw:
        Desired final box yaw in WORLD. Use 0.0 to align with the table/world.
    place_grasp_offset_z:
        Vertical offset between the desired box target height and the gripper
        command height.
    release_distance:
        How far the two grippers move apart during release.
    place_d1_y_bias, place_d2_y_bias:
        Small lateral biases applied before the final Z motion, expressed along
        the final place lateral axis.
    align_orientation_to_place_world_yaw:
        If true, rotates the gripper orientation so the place yaw is expressed
        correctly even when Centauro is yawed in WORLD.
    d1_qx/qy/qz/qw, d2_qx/qy/qz/qw:
        Base gripper orientations used by the Cartesian action goals.
    time_phase_xy, time_phase_z, time_phase_release:
        Motion duration for each place phase.
"""

import math
import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class CentauroCleanPlace(Node):

    def __init__(self):
        super().__init__('centauro_clean_place')

        self.client_1 = ActionClient(self, ReachPose, '/arm1_8/reach')
        self.client_2 = ActionClient(self, ReachPose, '/arm2_8/reach')

        # =====================================================
        # GAZEBO / ENTITY NAMES
        # =====================================================
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        self.declare_parameter('box_name', 'box_red_001')
        self.declare_parameter('robot_name', 'centauro')
        self.declare_parameter('dagana_1_name', 'dagana_1_bottom_link')
        self.declare_parameter('dagana_2_name', 'dagana_2_bottom_link')

        # =====================================================
        # TARGET PLACE POSITION IN WORLD
        # =====================================================
        self.declare_parameter('place_world_x', 3.72)
        self.declare_parameter('place_world_y', -0.38)
        self.declare_parameter('place_world_z', 0.67)
        self.declare_parameter('place_world_yaw', 0.0)

        # =====================================================
        # CONSTANT ABSOLUTE ORIENTATION
        # =====================================================
        self.declare_parameter('d1_qx', 0.5)
        self.declare_parameter('d1_qy', 0.5)
        self.declare_parameter('d1_qz', 0.5)
        self.declare_parameter('d1_qw', -0.5)

        self.declare_parameter('d2_qx', -0.5)
        self.declare_parameter('d2_qy', -0.5)
        self.declare_parameter('d2_qz', -0.5)
        self.declare_parameter('d2_qw', 0.5)

        # =====================================================
        # RELEASE
        # =====================================================
        self.declare_parameter('release_distance', 0.08)

        # mantengo i piccoli bias laterali che avevi nel vecchio file
        self.declare_parameter('place_d1_y_bias', -0.03)
        self.declare_parameter('place_d2_y_bias', +0.03)
        self.declare_parameter('place_grasp_offset_z', 0.0)
        self.declare_parameter('align_orientation_to_place_world_yaw', True)

        # =====================================================
        # TIMINGS
        # =====================================================
        self.declare_parameter('time_phase_xy', 15.0)
        self.declare_parameter('time_phase_z', 15.0)
        self.declare_parameter('time_phase_release', 15.0)

        # =====================================================
        # READ PARAMETERS
        # =====================================================
        self.box_name = self.get_parameter('box_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.dagana_1_name = self.get_parameter('dagana_1_name').value
        self.dagana_2_name = self.get_parameter('dagana_2_name').value

        self.place_world_x = self.get_parameter('place_world_x').value
        self.place_world_y = self.get_parameter('place_world_y').value
        self.place_world_z = self.get_parameter('place_world_z').value
        self.place_world_yaw = self.get_parameter('place_world_yaw').value

        self.d1_qx = self.get_parameter('d1_qx').value
        self.d1_qy = self.get_parameter('d1_qy').value
        self.d1_qz = self.get_parameter('d1_qz').value
        self.d1_qw = self.get_parameter('d1_qw').value

        self.d2_qx = self.get_parameter('d2_qx').value
        self.d2_qy = self.get_parameter('d2_qy').value
        self.d2_qz = self.get_parameter('d2_qz').value
        self.d2_qw = self.get_parameter('d2_qw').value

        self.release_distance = self.get_parameter('release_distance').value
        self.place_d1_y_bias = self.get_parameter('place_d1_y_bias').value
        self.place_d2_y_bias = self.get_parameter('place_d2_y_bias').value
        self.place_grasp_offset_z = self.get_parameter('place_grasp_offset_z').value
        self.align_orientation_to_place_world_yaw = self.get_parameter(
            'align_orientation_to_place_world_yaw'
        ).value

        self.time_phase_xy = self.get_parameter('time_phase_xy').value
        self.time_phase_z = self.get_parameter('time_phase_z').value
        self.time_phase_release = self.get_parameter('time_phase_release').value

        # =====================================================
        # SAVED POSITIONS
        # =====================================================
        self.box_position = None
        self.box_orientation = None
        self.robot_position = None
        self.robot_orientation = None
        self.dagana_1_position = None
        self.dagana_2_position = None

    # =========================================================
    # QUATERNION / TRANSFORM UTILS
    # =========================================================
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
        vx, vy, vz = v
        rotated = self.quat_multiply(
            self.quat_multiply(q, (vx, vy, vz, 0.0)),
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

    # =========================================================
    # READ POSITIONS FROM GAZEBO
    # =========================================================
    def get_entity_pose_from_gz(self, entity_name: str, timeout_sec: float = 3.0):
        cmd = [
            'gz', 'topic',
            '-e',
            '-n', '1',
            '-t', self.gz_pose_topic
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=True
            )
        except FileNotFoundError:
            self.get_logger().error(
                'Comando "gz" non trovato. Assicurati che Gazebo sia installato '
                'e che l\'ambiente sia caricato correttamente.'
            )
            return None
        except subprocess.TimeoutExpired:
            self.get_logger().error(
                f'Timeout mentre leggevo il topic Gazebo {self.gz_pose_topic}'
            )
            return None
        except subprocess.CalledProcessError as e:
            self.get_logger().error(
                f'Errore eseguendo "gz topic": {e.stderr.strip() if e.stderr else str(e)}'
            )
            return None

        output = result.stdout

        pattern = (
            r'name:\s*"' + re.escape(entity_name) + r'"\s*'
            r'id:\s*\d+\s*'
            r'position\s*\{\s*'
            r'x:\s*([-\d.eE+]+)\s*'
            r'y:\s*([-\d.eE+]+)\s*'
            r'z:\s*([-\d.eE+]+)\s*'
            r'\}\s*'
            r'orientation\s*\{\s*'
            r'x:\s*([-\d.eE+]+)\s*'
            r'y:\s*([-\d.eE+]+)\s*'
            r'z:\s*([-\d.eE+]+)\s*'
            r'w:\s*([-\d.eE+]+)\s*'
            r'\}'
        )

        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)

        if not match:
            self.get_logger().warn(
                f'Non ho trovato "{entity_name}" nel messaggio letto da {self.gz_pose_topic}'
            )
            return None

        try:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            qx = float(match.group(4))
            qy = float(match.group(5))
            qz = float(match.group(6))
            qw = float(match.group(7))
            return (x, y, z), self.quat_normalize((qx, qy, qz, qw))
        except ValueError as e:
            self.get_logger().error(
                f'Errore nel parsing della posa di "{entity_name}": {e}'
            )
            return None

    def get_entity_position_from_gz(self, entity_name: str, timeout_sec: float = 3.0):
        pose = self.get_entity_pose_from_gz(entity_name, timeout_sec=timeout_sec)
        if pose is None:
            return None
        position, _ = pose
        return position

    def read_all_positions(self):
        box_pose = self.get_entity_pose_from_gz(self.box_name)
        robot_pose = self.get_entity_pose_from_gz(self.robot_name)

        if box_pose is not None:
            self.box_position, self.box_orientation = box_pose
        if robot_pose is not None:
            self.robot_position, self.robot_orientation = robot_pose

        self.dagana_1_position = self.get_entity_position_from_gz_candidates(
            self.dagana_1_name,
            [
                self.dagana_1_name,
                'dagana_1_bottom_link',
                'arm1_8',
                'ball1_tip',
                'ball1',
                'dagana_1_tcp',
                'dagana_1_top_link',
            ]
        )
        self.dagana_2_position = self.get_entity_position_from_gz_candidates(
            self.dagana_2_name,
            [
                self.dagana_2_name,
                'dagana_2_bottom_link',
                'arm2_8',
                'ball2_tip',
                'ball2',
                'dagana_2_tcp',
                'dagana_2_top_link',
            ]
        )

        self._log_position(self.box_name, self.box_position)
        self._log_orientation(self.box_name, self.box_orientation)
        self._log_position(self.robot_name, self.robot_position)
        self._log_orientation(self.robot_name, self.robot_orientation)
        self._log_position(self.dagana_1_name, self.dagana_1_position)
        self._log_position(self.dagana_2_name, self.dagana_2_position)

    def get_entity_position_from_gz_candidates(self, label, entity_names):
        seen = set()

        for entity_name in entity_names:
            if entity_name in seen:
                continue
            seen.add(entity_name)

            position = self.get_entity_position_from_gz(entity_name)
            if position is not None:
                if entity_name != label:
                    self.get_logger().info(
                        f'Uso "{entity_name}" come frame corrente per "{label}".'
                    )
                return position

        self.get_logger().error(
            f'Nessun frame trovato per "{label}". Nomi provati: {", ".join(seen)}'
        )
        return None

    def _log_position(self, name, pos):
        if pos is None:
            self.get_logger().warn(f'Posizione di "{name}" non disponibile.')
            return

        x, y, z = pos
        self.get_logger().info(f'{name}: x={x:.6f}, y={y:.6f}, z={z:.6f}')

    def _log_orientation(self, name, quat):
        if quat is None:
            self.get_logger().warn(f'Orientazione di "{name}" non disponibile.')
            return
        yaw = self.quat_to_yaw(quat)
        x, y, z, w = quat
        self.get_logger().info(
            f'{name} orientation: q=({x:.6f}, {y:.6f}, {z:.6f}, {w:.6f}), '
            f'yaw={yaw:.6f} rad'
        )

    # =========================================================
    # FRAME CONVERSIONS
    # =========================================================
    def world_position_to_robot_base(self, world_position):
        """
        Convert a WORLD position to Centauro base frame.
        """
        if self.robot_position is None or self.robot_orientation is None:
            self.get_logger().error('robot pose non disponibile')
            return None

        delta_world = self.vector_sub(world_position, self.robot_position)
        robot_inv = self.quat_conjugate(self.robot_orientation)
        return self.quat_rotate_vector(robot_inv, delta_world)

    def world_vector_to_robot_base(self, world_vector):
        if self.robot_orientation is None:
            self.get_logger().error('robot_orientation è None')
            return None

        robot_inv = self.quat_conjugate(self.robot_orientation)
        return self.quat_rotate_vector(robot_inv, world_vector)

    def place_lateral_axis_base(self):
        place_world_yaw_q = self.yaw_to_quat(self.place_world_yaw)
        lateral_world = self.quat_rotate_vector(place_world_yaw_q, (0.0, 1.0, 0.0))
        lateral_base = self.world_vector_to_robot_base(lateral_world)
        if lateral_base is None:
            return None
        norm = math.sqrt(
            lateral_base[0]*lateral_base[0] +
            lateral_base[1]*lateral_base[1] +
            lateral_base[2]*lateral_base[2]
        )
        if norm <= 0.0:
            return (0.0, 1.0, 0.0)
        return (lateral_base[0] / norm, lateral_base[1] / norm, lateral_base[2] / norm)

    def current_box_base_frame(self):
        """
        Current box position expressed in Centauro base frame.
        """
        if self.box_position is None:
            self.get_logger().error('box_position è None')
            return None

        box_x_world, box_y_world, box_z_world = self.box_position
        converted = self.world_position_to_robot_base((box_x_world, box_y_world, box_z_world))
        if converted is None:
            return None

        box_x_robot, box_y_robot, box_z_robot = converted

        self.get_logger().info(
            f'Box current in Centauro base frame: '
            f'x={box_x_robot:.6f}, y={box_y_robot:.6f}, z={box_z_robot:.6f}'
        )

        return (box_x_robot, box_y_robot, box_z_robot)

    def target_box_base_frame(self):
        """
        Target place position expressed in Centauro base frame.
        """
        converted = self.world_position_to_robot_base(
            (self.place_world_x, self.place_world_y, self.place_world_z)
        )
        if converted is None:
            return None

        target_x_robot, target_y_robot, target_z_robot = converted

        self.get_logger().info(
            f'Box target in Centauro base frame: '
            f'x={target_x_robot:.6f}, y={target_y_robot:.6f}, z={target_z_robot:.6f}'
        )

        return (target_x_robot, target_y_robot, target_z_robot)

    def current_dagana_base_frame_targets(self):
        """
        Current Dagana claw positions expressed in Centauro base frame.
        """
        if self.dagana_1_position is None:
            self.get_logger().error('dagana_1_position è None')
            return None

        if self.dagana_2_position is None:
            self.get_logger().error('dagana_2_position è None')
            return None

        d1_world_x, d1_world_y, d1_world_z = self.dagana_1_position
        d2_world_x, d2_world_y, d2_world_z = self.dagana_2_position

        d1_base = self.world_position_to_robot_base((d1_world_x, d1_world_y, d1_world_z))
        d2_base = self.world_position_to_robot_base((d2_world_x, d2_world_y, d2_world_z))

        if d1_base is None or d2_base is None:
            return None

        d1x, d1y, d1z = d1_base
        d2x, d2y, d2z = d2_base

        self.get_logger().info(
            f'Dagana current in Centauro base frame: '
            f'd1=({d1x:.6f}, {d1y:.6f}, {d1z:.6f}), '
            f'd2=({d2x:.6f}, {d2y:.6f}, {d2z:.6f})'
        )

        return (
            (d1x, d1y, d1z),
            (d2x, d2y, d2z)
        )

    # =========================================================
    # PLACE TARGETS IN ABSOLUTE
    # =========================================================
    def compute_place_phase_targets(self):
        current_box = self.current_box_base_frame()
        target_box = self.target_box_base_frame()
        current_daganas = self.current_dagana_base_frame_targets()

        if current_box is None or target_box is None or current_daganas is None:
            return None

        current_bx, current_by, current_bz = current_box
        target_bx, target_by, target_bz = target_box
        lateral_axis = self.place_lateral_axis_base()
        if lateral_axis is None:
            return None

        (d1x, d1y, d1z), (d2x, d2y, d2z) = current_daganas

        delta = (
            target_bx - current_bx,
            target_by - current_by,
            target_bz - current_bz
        )
        delta_x, delta_y, delta_z = delta

        d1_bias = self.vector_scale(lateral_axis, self.place_d1_y_bias)
        d2_bias = self.vector_scale(lateral_axis, self.place_d2_y_bias)

        phase_xy_d1 = self.vector_add(
            self.vector_add((d1x, d1y, d1z), (delta_x, delta_y, 0.0)),
            d1_bias
        )
        phase_xy_d2 = self.vector_add(
            self.vector_add((d2x, d2y, d2z), (delta_x, delta_y, 0.0)),
            d2_bias
        )

        phase_xy_d1_x, phase_xy_d1_y, _ = phase_xy_d1
        phase_xy_d2_x, phase_xy_d2_y, _ = phase_xy_d2
        phase_xy_d1_z = current_bz + self.place_grasp_offset_z
        phase_xy_d2_z = current_bz + self.place_grasp_offset_z

        phase_z_d1_x = phase_xy_d1_x
        phase_z_d1_y = phase_xy_d1_y
        phase_z_d1_z = target_bz + self.place_grasp_offset_z

        phase_z_d2_x = phase_xy_d2_x
        phase_z_d2_y = phase_xy_d2_y
        phase_z_d2_z = target_bz + self.place_grasp_offset_z

        self.get_logger().info('=== TARGET ASSOLUTI PLACE CALCOLATI ===')
        self.get_logger().info(
            f'delta_box: dx={delta_x:.6f}, dy={delta_y:.6f}, dz={delta_z:.6f}'
        )
        self.get_logger().info(
            f'place_grasp_offset_z = {self.place_grasp_offset_z:.6f}'
        )
        self.get_logger().info(
            f'place_lateral_axis_base = '
            f'({lateral_axis[0]:.6f}, {lateral_axis[1]:.6f}, {lateral_axis[2]:.6f})'
        )
        self.get_logger().info(
            f'PLACE_XY dagana_1: x={phase_xy_d1_x:.6f}, y={phase_xy_d1_y:.6f}, z={phase_xy_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PLACE_XY dagana_2: x={phase_xy_d2_x:.6f}, y={phase_xy_d2_y:.6f}, z={phase_xy_d2_z:.6f}'
        )
        self.get_logger().info(
            f'PLACE_Z dagana_1: x={phase_z_d1_x:.6f}, y={phase_z_d1_y:.6f}, z={phase_z_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PLACE_Z dagana_2: x={phase_z_d2_x:.6f}, y={phase_z_d2_y:.6f}, z={phase_z_d2_z:.6f}'
        )

        return {
            'phase_xy': (
                phase_xy_d1_x, phase_xy_d1_y, phase_xy_d1_z,
                phase_xy_d2_x, phase_xy_d2_y, phase_xy_d2_z
            ),
            'phase_z': (
                phase_z_d1_x, phase_z_d1_y, phase_z_d1_z,
                phase_z_d2_x, phase_z_d2_y, phase_z_d2_z
            )
        }

    def compute_release_targets(self, current_targets):
        if current_targets is None:
            self.get_logger().error('current_targets è None')
            return None

        lateral_axis = self.place_lateral_axis_base()
        if lateral_axis is None:
            return None

        d1x, d1y, d1z, d2x, d2y, d2z = current_targets

        release_d1 = self.vector_add(
            (d1x, d1y, d1z),
            self.vector_scale(lateral_axis, self.release_distance)
        )
        release_d1_x, release_d1_y, release_d1_z = release_d1

        release_d2 = self.vector_add(
            (d2x, d2y, d2z),
            self.vector_scale(lateral_axis, -self.release_distance)
        )
        release_d2_x, release_d2_y, release_d2_z = release_d2

        self.get_logger().info('=== TARGET ASSOLUTI RELEASE CALCOLATI ===')
        self.get_logger().info(
            f'dagana_1: x={release_d1_x:.6f}, y={release_d1_y:.6f}, z={release_d1_z:.6f}'
        )
        self.get_logger().info(
            f'dagana_2: x={release_d2_x:.6f}, y={release_d2_y:.6f}, z={release_d2_z:.6f}'
        )

        return (
            release_d1_x, release_d1_y, release_d1_z,
            release_d2_x, release_d2_y, release_d2_z
        )

    # =========================================================
    # MANIPULATION
    # =========================================================
    def make_goal(self, x, y, z, qx, qy, qz, qw, time_s=15.0):
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

        if not self.align_orientation_to_place_world_yaw:
            return d1_q, d2_q

        if self.robot_orientation is None:
            self.get_logger().error('robot_orientation è None')
            return d1_q, d2_q

        robot_inv = self.quat_conjugate(self.robot_orientation)
        place_world_yaw_q = self.yaw_to_quat(self.place_world_yaw)
        place_yaw_base_q = self.quat_normalize(
            self.quat_multiply(robot_inv, place_world_yaw_q)
        )

        return (
            self.quat_normalize(self.quat_multiply(place_yaw_base_q, d1_q)),
            self.quat_normalize(self.quat_multiply(place_yaw_base_q, d2_q)),
        )

    def send_two_goals_and_wait(self, phase_name: str, goal1: ReachPose.Goal, goal2: ReachPose.Goal) -> None:
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
            raise RuntimeError(f'{phase_name}: goal dagana_1 rejected.')
        if gh2 is None or not gh2.accepted:
            raise RuntimeError(f'{phase_name}: goal dagana_2 rejected.')

        fut_res1 = gh1.get_result_async()
        fut_res2 = gh2.get_result_async()

        rclpy.spin_until_future_complete(self, fut_res1)
        rclpy.spin_until_future_complete(self, fut_res2)

        if fut_res1.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_1.')
        if fut_res2.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_2.')

        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase_absolute(self, phase_name: str, x1, y1, z1, x2, y2, z2, time_s=15.0) -> None:
        d1_q, d2_q = self.current_goal_orientations()
        goal1 = self.make_goal(
            x1, y1, z1,
            d1_q[0], d1_q[1], d1_q[2], d1_q[3],
            time_s=time_s
        )
        goal2 = self.make_goal(
            x2, y2, z2,
            d2_q[0], d2_q[1], d2_q[2], d2_q[3],
            time_s=time_s
        )
        self.send_two_goals_and_wait(phase_name, goal1, goal2)

    def execute(self):
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        place_phases = self.compute_place_phase_targets()
        if place_phases is None:
            raise RuntimeError('Impossibile calcolare i target assoluti di place.')

        self.run_phase_absolute(
            'PLACE_MOVE_XY',
            *place_phases['phase_xy'],
            time_s=self.time_phase_xy
        )

        self.run_phase_absolute(
            'PLACE_MOVE_Z',
            *place_phases['phase_z'],
            time_s=self.time_phase_z
        )

        release_targets = self.compute_release_targets(place_phases['phase_z'])
        if release_targets is None:
            raise RuntimeError('Impossibile calcolare i target assoluti di release.')

        self.run_phase_absolute(
            'PLACE_RELEASE',
            *release_targets,
            time_s=self.time_phase_release
        )

        self.get_logger().info("Place completed.")

    def print_saved_positions(self):
        self._print_one('box_position', self.box_position)
        self._print_one('robot_position', self.robot_position)
        self._print_one('dagana_1_position', self.dagana_1_position)
        self._print_one('dagana_2_position', self.dagana_2_position)

    def _print_one(self, label, pos):
        if pos is None:
            self.get_logger().info(f'{label} = None')
            return

        x, y, z = pos
        self.get_logger().info(f'{label} = ({x:.6f}, {y:.6f}, {z:.6f})')


def main():
    rclpy.init()
    node = CentauroCleanPlace()
    try:
        node.execute()
        node.print_saved_positions()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
