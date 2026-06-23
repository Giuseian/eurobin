#!/usr/bin/env python3

"""
Check-only version of grasp_centauro_test.py.

It performs the same object-pose, TF and target computations, but it never
sends ReachPose goals and never calls Cartesian task-weight services.

Run from this folder with the same relevant parameters used for the real grasp:

python3 grasp_centauro_test_world_check.py --ros-args \
    -p object_pose_file:=object_pose.txt \
    -p base_frame:=world \
    -p cartesian_world_frame:=ci/world \
    -p cartesian_robot_base_frame:=ci/pelvis \
    -p robot_base_frame:=pelvis \
    -p camera_frame:=D435_head_camera_link \
    -p approach_mode:=yz \
    -p grasp_offset_x:=0.12 \
    -p grasp_offset_y:=-0.075 \
    -p grasp_offset_z:=0.0 \
    -p phase1_split_fraction:=0.5 \
    -p constrain_orientation:=true
"""

from pathlib import Path
import math
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class CentauroCameraGraspCheck(Node):

    def __init__(self):
        super().__init__('centauro_camera_grasp_test_check')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('object_pose_file', 'object_pose.txt')
        self.declare_parameter('d1_set_weight_service', '/cartesian/dagana_1_tcp/set_weight')
        self.declare_parameter('d2_set_weight_service', '/cartesian/dagana_2_tcp/set_weight')

        self.declare_parameter('base_frame', 'world')
        self.declare_parameter('cartesian_world_frame', 'ci/world')
        self.declare_parameter('cartesian_robot_base_frame', 'ci/pelvis')
        self.declare_parameter('robot_base_frame', 'pelvis')
        self.declare_parameter('camera_frame', 'D435_head_camera_gz_optical_frame')
        self.declare_parameter('tf_lookup_timeout', 20.0)

        self.declare_parameter('d1_start_x', 0.8)
        self.declare_parameter('d1_start_y', 0.3)
        self.declare_parameter('d1_start_z', 0.25)
        self.declare_parameter('d2_start_x', 0.8)
        self.declare_parameter('d2_start_y', -0.35)
        self.declare_parameter('d2_start_z', 0.25)

        self.declare_parameter('d1_qx', 0.0)
        self.declare_parameter('d1_qy', 0.7)
        self.declare_parameter('d1_qz', 0.0)
        self.declare_parameter('d1_qw', 0.7)

        self.declare_parameter('d2_qx', 0.0)
        self.declare_parameter('d2_qy', 0.7)
        self.declare_parameter('d2_qz', 0.0)
        self.declare_parameter('d2_qw', 0.7)

        self.declare_parameter('box_width', 0.335)
        self.declare_parameter('phase1_clearance', 0.1)
        self.declare_parameter('phase1_extra_margin_x', 0.0)
        self.declare_parameter('phase1_extra_margin_y', 0.0)
        self.declare_parameter('phase1_extra_margin_z', 0.0)
        self.declare_parameter('approach_mode', 'yz')
        self.declare_parameter('phase1_split_fraction', 0.5)

        self.declare_parameter('grasp_offset_x', 0.0)
        self.declare_parameter('grasp_offset_y', 0.0)
        self.declare_parameter('grasp_offset_z', 0.0)
        self.declare_parameter('align_orientation_to_box_yaw', True)
        self.declare_parameter('constrain_orientation', True)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('restore_orientation_weight_at_end', False)
        self.declare_parameter('set_weight_timeout', 20.0)

        self.declare_parameter('phase3_extra_squeeze', 0.0)
        self.declare_parameter('lift_z_phase4', 0.15)

        self.declare_parameter('time_start_pose', 5.0)
        self.declare_parameter('time_phase1', 5.0)
        self.declare_parameter('time_phase2', 5.0)
        self.declare_parameter('time_phase3', 5.0)
        self.declare_parameter('time_phase4', 5.0)

        self.object_pose_file = self.get_parameter('object_pose_file').value
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.cartesian_world_frame = str(
            self.get_parameter('cartesian_world_frame').value
        ).strip()
        self.cartesian_robot_base_frame = str(
            self.get_parameter('cartesian_robot_base_frame').value
        ).strip()
        self.robot_base_frame = str(
            self.get_parameter('robot_base_frame').value
        ).strip()
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.tf_lookup_timeout = self.get_parameter('tf_lookup_timeout').value

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
        self.constrain_orientation = self.get_parameter('constrain_orientation').value
        self.position_weight = self.get_parameter('position_weight').value
        self.orientation_weight = self.get_parameter('orientation_weight').value

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

        self.camera_position = None
        self.camera_orientation = None
        self.box_position = None
        self.box_orientation = None
        self.object_position_camera = None

    def task_weight_for_current_orientation_mode(self):
        pos_w = float(self.position_weight)
        if self.constrain_orientation:
            ori_w = pos_w
        else:
            ori_w = float(self.orientation_weight)
        return [pos_w, pos_w, pos_w, ori_w, ori_w, ori_w]

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

    def vector_scale(self, v, scale):
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    def pose_from_transform(self, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (translation.x, translation.y, translation.z),
            self.quat_normalize((rotation.x, rotation.y, rotation.z, rotation.w)),
        )

    def compose_poses(self, first_position, first_orientation, second_position, second_orientation):
        rotated_second = self.quat_rotate_vector(first_orientation, second_position)
        position = self.vector_add(first_position, rotated_second)
        orientation = self.quat_normalize(
            self.quat_multiply(first_orientation, second_orientation)
        )
        return position, orientation

    def lookup_tf_pose(self, target_frame, source_frame, timeout_s, label):
        deadline = time.monotonic() + float(timeout_s)
        last_error = None

        self.get_logger().info(
            f'Cerco TF {target_frame} -> {source_frame} per {label}...'
        )

        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                pose = self.pose_from_transform(transform)
                self.get_logger().info(
                    f'TF trovato per {label}: {target_frame} -> {source_frame}'
                )
                return pose
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.1)

        raise RuntimeError(
            f'TF {target_frame} -> {source_frame} non disponibile dopo '
            f'{float(timeout_s):.1f} s: {last_error}'
        )

    def update_camera_pose_from_tf(self):
        timeout_s = float(self.tf_lookup_timeout)

        world_pelvis_position, world_pelvis_orientation = self.lookup_tf_pose(
            self.cartesian_world_frame,
            self.cartesian_robot_base_frame,
            timeout_s,
            'T_cartesio_world_pelvis',
        )
        pelvis_camera_position, pelvis_camera_orientation = self.lookup_tf_pose(
            self.robot_base_frame,
            self.camera_frame,
            timeout_s,
            'T_pelvis_camera',
        )

        self.camera_position, self.camera_orientation = self.compose_poses(
            world_pelvis_position,
            world_pelvis_orientation,
            pelvis_camera_position,
            pelvis_camera_orientation,
        )
        self.get_logger().info(
            f'Camera composta in world Cartesio: '
            f'{self.cartesian_world_frame} -> {self.cartesian_robot_base_frame} -> '
            f'{self.robot_base_frame} -> {self.camera_frame}'
        )

    def camera_position_to_cartesian_world(self, camera_position):
        if self.camera_position is None or self.camera_orientation is None:
            raise RuntimeError('Posa camera non disponibile: lookup TF non eseguito.')
        rotated = self.quat_rotate_vector(self.camera_orientation, camera_position)
        return self.vector_add(self.camera_position, rotated)

    def resolve_object_pose_path(self):
        path = Path(str(self.object_pose_file)).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parent / path

    def parse_object_pose_file(self):
        path = self.resolve_object_pose_path()
        if not path.is_file():
            raise FileNotFoundError(f'File pose oggetto non trovato: {path}')

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
            'object_position': (
                'object_position',
                'object_pose_position',
                'box_position',
                'position',
            ),
        }

        parsed = {}
        for canonical_key, possible_keys in aliases.items():
            for possible_key in possible_keys:
                if possible_key in data:
                    parsed[canonical_key] = data[possible_key]
                    break

        required_lengths = {
            'object_position': 3,
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
        poses = self.parse_object_pose_file()

        self.object_position_camera = poses['object_position']
        self.box_position = self.camera_position_to_cartesian_world(self.object_position_camera)
        self.box_orientation = (0.0, 0.0, 0.0, 1.0)

        self.log_pose('camera_in_cartesian_world', self.camera_position, self.camera_orientation)
        self.get_logger().info(
            f'object_in_camera position: x={self.object_position_camera[0]:.6f}, '
            f'y={self.object_position_camera[1]:.6f}, '
            f'z={self.object_position_camera[2]:.6f}'
        )
        self.log_pose('object_in_cartesian_world', self.box_position, self.box_orientation)

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

    def box_target_components(self):
        box_position_base = self.box_position
        box_orientation_base = self.box_orientation
        if box_position_base is None or box_orientation_base is None:
            return None

        box_x_axis_base = self.quat_rotate_vector(box_orientation_base, (1.0, 0.0, 0.0))
        box_y_axis_base = self.quat_rotate_vector(box_orientation_base, (0.0, 1.0, 0.0))
        box_z_axis_base = self.quat_rotate_vector(box_orientation_base, (0.0, 0.0, 1.0))

        grasp_offset_base = self.vector_add(
            self.vector_add(
                self.vector_scale(box_x_axis_base, self.grasp_offset_x),
                self.vector_scale(box_y_axis_base, self.grasp_offset_y)
            ),
            self.vector_scale(box_z_axis_base, self.grasp_offset_z)
        )
        grasp_position_base = self.vector_add(box_position_base, grasp_offset_base)
        box_yaw_base = self.quat_to_yaw(box_orientation_base)

        self.get_logger().info(
            f'Oggetto in world Cartesio: x={box_position_base[0]:.6f}, '
            f'y={box_position_base[1]:.6f}, z={box_position_base[2]:.6f}, '
            f'yaw={box_yaw_base:.6f} rad'
        )
        self.get_logger().info(
            f'Grasp center in world Cartesio: x={grasp_position_base[0]:.6f}, '
            f'y={grasp_position_base[1]:.6f}, z={grasp_position_base[2]:.6f}'
        )

        return {
            'grasp_position': grasp_position_base,
            'box_y_axis': box_y_axis_base,
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
        return phases

    def current_goal_orientations(self):
        d1_q = self.quat_normalize((self.d1_qx, self.d1_qy, self.d1_qz, self.d1_qw))
        d2_q = self.quat_normalize((self.d2_qx, self.d2_qy, self.d2_qz, self.d2_qw))

        if not self.align_orientation_to_box_yaw:
            return d1_q, d2_q

        box_orientation_base = self.box_orientation
        if box_orientation_base is None:
            return d1_q, d2_q

        box_yaw_base = self.quat_to_yaw(box_orientation_base)
        yaw_q = self.yaw_to_quat(box_yaw_base)
        return (
            self.quat_normalize(self.quat_multiply(yaw_q, d1_q)),
            self.quat_normalize(self.quat_multiply(yaw_q, d2_q)),
        )

    def print_check_report(self, phases):
        d1_q, d2_q = self.current_goal_orientations()
        weights = self.task_weight_for_current_orientation_mode()
        pose_path = self.resolve_object_pose_path()

        print()
        print('=' * 78)
        print('CHECK GRASP CENTAURO - NESSUN COMANDO INVIATO AL ROBOT')
        print('=' * 78)
        print(f'Object pose file : {pose_path}')
        print(f'Frame camera     : {self.camera_frame}')
        print(f'Frame target     : {self.base_frame}')
        print(f'Cartesio world TF: {self.cartesian_world_frame}')
        print(f'Cartesio base TF : {self.cartesian_robot_base_frame}')
        print(f'Robot base TF    : {self.robot_base_frame}')
        print(f'Approach mode    : {self.approach_mode}')
        print(f'Box width        : {self.box_width:.6f} m')
        print(f'Phase1 clearance : {self.phase1_clearance:.6f} m')
        print(
            'Grasp offsets    : '
            f'x={self.grasp_offset_x:.6f}, '
            f'y={self.grasp_offset_y:.6f}, '
            f'z={self.grasp_offset_z:.6f} m'
        )
        print(
            'Task weights     : '
            f'tx={weights[0]:.3f}, ty={weights[1]:.3f}, tz={weights[2]:.3f}, '
            f'rx={weights[3]:.3f}, ry={weights[4]:.3f}, rz={weights[5]:.3f} '
            '(solo informativo, servizio non chiamato)'
        )
        print('-' * 78)
        print(
            'object_in_camera : '
            f'x={self.object_position_camera[0]: .6f}, '
            f'y={self.object_position_camera[1]: .6f}, '
            f'z={self.object_position_camera[2]: .6f} m'
        )
        print(
            'camera_in_world  : '
            f'x={self.camera_position[0]: .6f}, '
            f'y={self.camera_position[1]: .6f}, '
            f'z={self.camera_position[2]: .6f} m'
        )
        print(
            'object_in_world  : '
            f'x={self.box_position[0]: .6f}, '
            f'y={self.box_position[1]: .6f}, '
            f'z={self.box_position[2]: .6f} m'
        )
        print('-' * 78)
        print('POSIZIONI DAGANA RISPETTO AL WORLD CARTESIO')
        print('Valori in metri. d1 = /dagana_1_tcp, d2 = /dagana_2_tcp')
        print('-' * 78)

        phase_labels = {
            'start_pose': 'START_POSE',
            'phase1': f'PHASE 1 - {self.approach_mode.upper()}',
            'phase2': 'PHASE 2 - COMPLETE_REMAINING_AXIS',
            'phase3': 'PHASE 3 - CLOSE_ON_OBJECT_LOCAL_Y',
            'phase4': 'PHASE 4 - SQUEEZE_AND_LIFT',
        }
        phase_times = {
            'start_pose': self.time_start_pose,
            'phase1': self.time_phase1,
            'phase2': self.time_phase2,
            'phase3': self.time_phase3,
            'phase4': self.time_phase4,
        }

        for phase_name, values in phases.items():
            x1, y1, z1, x2, y2, z2 = values
            print(f'{phase_labels[phase_name]}  (time={phase_times[phase_name]:.3f} s)')
            print(f'  dagana_1: x={x1: .6f}  y={y1: .6f}  z={z1: .6f}')
            print(f'  dagana_2: x={x2: .6f}  y={y2: .6f}  z={z2: .6f}')

        print('-' * 78)
        print('ORIENTAZIONI CHE SAREBBERO USATE NEI GOAL')
        print(
            f'  dagana_1 q: x={d1_q[0]: .6f}  y={d1_q[1]: .6f}  '
            f'z={d1_q[2]: .6f}  w={d1_q[3]: .6f}'
        )
        print(
            f'  dagana_2 q: x={d2_q[0]: .6f}  y={d2_q[1]: .6f}  '
            f'z={d2_q[2]: .6f}  w={d2_q[3]: .6f}'
        )
        print('=' * 78)
        print('FINE CHECK: nessun action goal e nessun service call eseguiti.')
        print('=' * 78)
        print()

    def execute(self):
        self.update_camera_pose_from_tf()

        self.get_logger().info('Lettura pose da file...')
        self.read_input_poses()

        phases = self.compute_all_phase_targets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare i target assoluti delle fasi.')

        self.print_check_report(phases)


def main():
    rclpy.init()
    node = CentauroCameraGraspCheck()

    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f'Check failed: {exc}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
