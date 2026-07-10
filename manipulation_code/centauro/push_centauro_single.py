#!/usr/bin/env python3

"""
Centauro single-arm push test from an object pose already expressed in world.

Run from this folder:

python3 push_centauro_single.py --ros-args \
    -p object_pose_file:=object_pose.txt \
    -p base_frame:=world \
    -p push_offset_x:=0.0 \
    -p push_offset_y:=-0.0 \
    -p push_offset_z:=0.02 \
    -p push_direction_x:=1.0 \
    -p push_direction_y:=0.0 \
    -p push_direction_z:=0.0 \
    -p pre_push_distance:=0.08 \
    -p push_distance:=0.15 \
    -p retreat_distance:=0.0 \
    -p constrain_orientation:=true \
    -p push_dagana:=2

What it does:
    1. Reads the object pose from a text file already expressed in base_frame.
    2. Uses a base-frame-aligned object frame, ignoring object orientation.
    3. Deactivates the non-pushing Dagana Cartesian task.
    4. Moves only the pushing Dagana to its configurable safe start pose.
    5. Moves to a pre-push pose, then to contact.
    6. Pushes along a configurable direction for a configurable distance.
    7. Optionally retreats after the push.
    8. Reactivates the non-pushing Dagana Cartesian task.

Input pose file format:
    The file is read from the same folder by default. Values are whitespace
    separated.

        object.position: 0.50 0.00 0.20

    The position above is expressed in base_frame. Object orientation is
    intentionally ignored; the push frame is aligned with base_frame.

Most useful ROS parameters:
    object_pose_file:
        Text file containing the object pose in the world/base frame. Relative
        paths are resolved from this script folder.
    base_frame:
        Robot base frame used as target frame for the push. Default: world.
    push_offset_x/y/z:
        Offset of the push contact point relative to the box center, expressed
        in the task-base-aligned object frame.
    push_direction_x/y/z:
        Push direction expressed in the task-base-aligned object frame. It is
        normalized internally.
    push_dagana:
        Which Dagana executes homing and push after the other one is disabled.
        Use 1 or 2.
    pre_push_distance:
        Distance before the contact point, opposite to the push direction.
    push_distance:
        Distance covered during the push.
    retreat_distance:
        Distance moved back after the push. Set to 0.0 to skip retreat.
    align_orientation_to_box_yaw:
        If true, rotates the commanded gripper orientation by the box yaw in
        the Cartesio task base frame.
    constrain_orientation:
        If false, sets the Dagana 2 Cartesian task weights to position-only
        [tx, ty, tz, rx, ry, rz] = [1, 1, 1, 0, 0, 0] before moving.
        At the end the script restores full 6D weights only if
        restore_orientation_weight_at_end is true.
    d1_start_x/y/z, d2_start_x/y/z:
        Initial safe pose for each Dagana, expressed in Cartesio task base frame.
    d1_qx/qy/qz/qw, d2_qx/qy/qz/qw:
        Base gripper orientations used by the Cartesian action goals.
    time_start_pose, time_pre_push, time_contact, time_push, time_retreat:
        Motion duration for each phase.
"""

from pathlib import Path
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cartesian_interface_ros.action import ReachPose
from cartesian_interface_ros.srv import SetTaskActive, SetWeight
from geometry_msgs.msg import Pose


class CentauroDirectPush(Node):

    def __init__(self):
        super().__init__('centauro_direct_push_test')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_tcp/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_tcp/reach')
        self.set_active_client_1 = self.create_client(
            SetTaskActive, '/cartesian/dagana_1_tcp/set_active'
        )
        self.set_active_client_2 = self.create_client(
            SetTaskActive, '/cartesian/dagana_2_tcp/set_active'
        )
        self.set_weight_client_1 = self.create_client(
            SetWeight, '/cartesian/dagana_1_tcp/set_weight'
        )
        self.set_weight_client_2 = self.create_client(
            SetWeight, '/cartesian/dagana_2_tcp/set_weight'
        )
        self.declare_parameter('object_pose_file', 'object_pose.txt')
        self.declare_parameter('d1_set_active_service', '/cartesian/dagana_1_tcp/set_active')
        self.declare_parameter('d2_set_active_service', '/cartesian/dagana_2_tcp/set_active')
        self.declare_parameter('d1_set_weight_service', '/cartesian/dagana_1_tcp/set_weight')
        self.declare_parameter('d2_set_weight_service', '/cartesian/dagana_2_tcp/set_weight')

        self.declare_parameter('base_frame', 'world')
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

        self.declare_parameter('push_dagana', 2)
        self.declare_parameter('push_offset_x', 0.0)
        self.declare_parameter('push_offset_y', 0.0)
        self.declare_parameter('push_offset_z', 0.0)
        self.declare_parameter('push_direction_x', 1.0)
        self.declare_parameter('push_direction_y', 0.0)
        self.declare_parameter('push_direction_z', 0.0)
        self.declare_parameter('pre_push_distance', 0.08)
        self.declare_parameter('push_distance', 0.10)
        self.declare_parameter('retreat_distance', 0.05)

        self.declare_parameter('align_orientation_to_box_yaw', True)
        self.declare_parameter('constrain_orientation', True)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('restore_orientation_weight_at_end', False)
        self.declare_parameter('set_active_timeout', 20.0)
        self.declare_parameter('set_weight_timeout', 20.0)

        self.declare_parameter('time_start_pose', 3.0)
        self.declare_parameter('time_pre_push', 3.0)
        self.declare_parameter('time_contact', 3.0)
        self.declare_parameter('time_push', 3.0)
        self.declare_parameter('time_retreat', 3.0)

        self.object_pose_file = self.get_parameter('object_pose_file').value
        self.d1_set_active_service = self.get_parameter('d1_set_active_service').value
        self.d2_set_active_service = self.get_parameter('d2_set_active_service').value
        self.d1_set_weight_service = self.get_parameter('d1_set_weight_service').value
        self.d2_set_weight_service = self.get_parameter('d2_set_weight_service').value
        self.base_frame = str(self.get_parameter('base_frame').value)
        if self.d1_set_active_service != '/cartesian/dagana_1_tcp/set_active':
            self.set_active_client_1 = self.create_client(
                SetTaskActive, self.d1_set_active_service
            )
        if self.d2_set_active_service != '/cartesian/dagana_2_tcp/set_active':
            self.set_active_client_2 = self.create_client(
                SetTaskActive, self.d2_set_active_service
            )

        if self.d1_set_weight_service != '/cartesian/dagana_1_tcp/set_weight':
            self.set_weight_client_1 = self.create_client(
                SetWeight, self.d1_set_weight_service
            )
        if self.d2_set_weight_service != '/cartesian/dagana_2_tcp/set_weight':
            self.set_weight_client_2 = self.create_client(
                SetWeight, self.d2_set_weight_service
            )

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

        self.push_dagana = int(self.get_parameter('push_dagana').value)
        self.push_offset_x = self.get_parameter('push_offset_x').value
        self.push_offset_y = self.get_parameter('push_offset_y').value
        self.push_offset_z = self.get_parameter('push_offset_z').value
        self.push_direction_x = self.get_parameter('push_direction_x').value
        self.push_direction_y = self.get_parameter('push_direction_y').value
        self.push_direction_z = self.get_parameter('push_direction_z').value
        self.pre_push_distance = self.get_parameter('pre_push_distance').value
        self.push_distance = self.get_parameter('push_distance').value
        self.retreat_distance = self.get_parameter('retreat_distance').value

        self.align_orientation_to_box_yaw = self.get_parameter('align_orientation_to_box_yaw').value
        self.constrain_orientation = self.get_parameter('constrain_orientation').value
        self.position_weight = self.get_parameter('position_weight').value
        self.orientation_weight = self.get_parameter('orientation_weight').value
        self.restore_orientation_weight_at_end = self.get_parameter(
            'restore_orientation_weight_at_end'
        ).value
        self.set_active_timeout = self.get_parameter('set_active_timeout').value
        self.set_weight_timeout = self.get_parameter('set_weight_timeout').value

        self.time_start_pose = self.get_parameter('time_start_pose').value
        self.time_pre_push = self.get_parameter('time_pre_push').value
        self.time_contact = self.get_parameter('time_contact').value
        self.time_push = self.get_parameter('time_push').value
        self.time_retreat = self.get_parameter('time_retreat').value

        if self.push_dagana not in (1, 2):
            raise ValueError(f'push_dagana={self.push_dagana} non valido. Usa 1 oppure 2.')

        if self.vector_norm((
            self.push_direction_x,
            self.push_direction_y,
            self.push_direction_z,
        )) <= 0.0:
            raise ValueError('La direzione di push non puo essere nulla.')

        self.box_position = None
        self.box_orientation = None

    # ------------------------------------------------------------------
    # Arm activation
    # ------------------------------------------------------------------
    def push_arm(self):
        if self.push_dagana == 1:
            return 'left'
        return 'right'

    def inactive_arm(self):
        if self.push_dagana == 1:
            return 'right'
        return 'left'

    def set_active_client_for_arm(self, arm):
        if arm == 'left':
            return self.set_active_client_1
        return self.set_active_client_2

    def active_service_name_for_arm(self, arm):
        if arm == 'left':
            return self.d1_set_active_service
        return self.d2_set_active_service

    def set_arm_active(self, arm, active):
        client = self.set_active_client_for_arm(arm)
        service_name = self.active_service_name_for_arm(arm)
        self.get_logger().info(
            f'{"Attivo" if active else "Disattivo"} task {arm}: {service_name}'
        )

        if not client.wait_for_service(timeout_sec=float(self.set_active_timeout)):
            raise RuntimeError(
                f'Service {service_name} non disponibile dopo '
                f'{self.set_active_timeout:.1f} s.'
            )

        request = SetTaskActive.Request()
        request.activation_state = bool(active)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=float(self.set_active_timeout)
        )

        response = future.result()
        if response is None:
            raise RuntimeError(
                f'Chiamata a {service_name} scaduta dopo '
                f'{self.set_active_timeout:.1f} s.'
            )
        if not response.success:
            raise RuntimeError(
                f'{service_name} ha risposto success=false: {response.message}'
            )
        self.get_logger().info(f'{service_name}: {response.message}')

    # ------------------------------------------------------------------
    # Cartesian task weights
    # ------------------------------------------------------------------
    def task_weight_for_current_orientation_mode(self):
        pos_w = float(self.position_weight)
        if self.constrain_orientation:
            ori_w = pos_w
        else:
            ori_w = float(self.orientation_weight)
        return [pos_w, pos_w, pos_w, ori_w, ori_w, ori_w]

    def full_pose_task_weight(self):
        pos_w = float(self.position_weight)
        return [pos_w, pos_w, pos_w, pos_w, pos_w, pos_w]

    def set_weight_client_for_arm(self, arm):
        if arm == 'left':
            return self.set_weight_client_1
        return self.set_weight_client_2

    def weight_service_name_for_arm(self, arm):
        if arm == 'left':
            return self.d1_set_weight_service
        return self.d2_set_weight_service

    def set_arm_task_weight(self, arm, weight=None):
        client = self.set_weight_client_for_arm(arm)
        service_name = self.weight_service_name_for_arm(arm)
        if weight is None:
            weight = self.task_weight_for_current_orientation_mode()

        self.get_logger().info(
            f'Imposto peso task {arm}: '
            f'tx={weight[0]:.3f}, ty={weight[1]:.3f}, tz={weight[2]:.3f}, '
            f'rx={weight[3]:.3f}, ry={weight[4]:.3f}, rz={weight[5]:.3f}'
        )

        if not client.wait_for_service(timeout_sec=float(self.set_weight_timeout)):
            raise RuntimeError(
                f'Service {service_name} non disponibile dopo '
                f'{self.set_weight_timeout:.1f} s.'
            )

        request = SetWeight.Request()
        request.weight = weight
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=float(self.set_weight_timeout)
        )

        response = future.result()
        if response is None:
            raise RuntimeError(
                f'Chiamata a {service_name} scaduta dopo '
                f'{self.set_weight_timeout:.1f} s.'
            )
        if not response.success:
            raise RuntimeError(
                f'{service_name} ha risposto success=false: {response.message}'
            )
        self.get_logger().info(f'{service_name}: {response.message}')

    def configure_task_weights(self):
        self.set_arm_task_weight(self.push_arm())

    def restore_full_pose_task_weights(self):
        errors = []
        weight = self.full_pose_task_weight()
        for arm in ('left', 'right'):
            try:
                self.set_arm_task_weight(arm, weight)
            except Exception as exc:
                errors.append(f'{arm}: {exc}')

        if errors:
            raise RuntimeError(
                'Ripristino incompleto dei pesi 6D arm: ' + '; '.join(errors)
            )

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

    def vector_scale(self, v, scale):
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    def vector_norm(self, v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    def vector_normalize(self, v):
        norm = self.vector_norm(v)
        if norm <= 0.0:
            raise ValueError('Impossibile normalizzare un vettore nullo.')
        return (v[0] / norm, v[1] / norm, v[2] / norm)

    # ------------------------------------------------------------------
    # Pose file input
    # ------------------------------------------------------------------
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
            'object_yaw': (
                'object_yaw',
                'object_pose_yaw',
                'box_yaw',
                'yaw',
            ),
            'object_yaw_world': (
                'object_yaw_world',
                'object_yaw_task',
                'object_yaw_cartesian_world',
                'box_yaw_world',
                'yaw_world',
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

        optional_lengths = {
            'object_yaw': 1,
            'object_yaw_world': 1,
        }
        for key, expected_length in required_lengths.items():
            if len(parsed[key]) != expected_length:
                raise ValueError(
                    f'{path}: "{key}" deve avere {expected_length} valori, '
                    f'ne ha {len(parsed[key])}'
                )
        for key, expected_length in optional_lengths.items():
            if key in parsed and len(parsed[key]) != expected_length:
                raise ValueError(
                    f'{path}: "{key}" deve avere {expected_length} valore, '
                    f'ne ha {len(parsed[key])}'
                )

        return parsed

    def read_input_poses(self):
        poses = self.parse_object_pose_file()

        self.box_position = poses['object_position']
        object_yaw = poses.get('object_yaw')
        object_yaw_world = poses.get('object_yaw_world')
        if object_yaw_world is not None:
            box_yaw = object_yaw_world[0]
            yaw_source = 'object.yaw_world'
        elif object_yaw is not None:
            box_yaw = object_yaw[0]
            yaw_source = 'object.yaw'
        else:
            box_yaw = 0.0
            yaw_source = 'default yaw=0'
        self.box_orientation = self.yaw_to_quat(box_yaw)

        self.get_logger().info(
            f'object_in_{self.base_frame} position: x={self.box_position[0]:.6f}, '
            f'y={self.box_position[1]:.6f}, z={self.box_position[2]:.6f}'
        )
        self.get_logger().info(
            f'object yaw source: {yaw_source}, {self.base_frame} yaw={box_yaw:.6f} rad'
        )
        self.log_pose(f'object_in_{self.base_frame}', self.box_position, self.box_orientation)

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
    # Push planner
    # ------------------------------------------------------------------
    def push_target_components(self):
        box_position_robot = self.box_position
        box_orientation_robot = self.box_orientation
        if box_position_robot is None or box_orientation_robot is None:
            return None

        box_x_axis_robot = self.quat_rotate_vector(box_orientation_robot, (1.0, 0.0, 0.0))
        box_y_axis_robot = self.quat_rotate_vector(box_orientation_robot, (0.0, 1.0, 0.0))
        box_z_axis_robot = self.quat_rotate_vector(box_orientation_robot, (0.0, 0.0, 1.0))

        push_offset_robot = self.vector_add(
            self.vector_add(
                self.vector_scale(box_x_axis_robot, self.push_offset_x),
                self.vector_scale(box_y_axis_robot, self.push_offset_y)
            ),
            self.vector_scale(box_z_axis_robot, self.push_offset_z)
        )
        contact_position_robot = self.vector_add(box_position_robot, push_offset_robot)

        local_push_direction = self.vector_normalize((
            self.push_direction_x,
            self.push_direction_y,
            self.push_direction_z,
        ))
        push_direction_robot = self.vector_normalize(self.vector_add(
            self.vector_add(
                self.vector_scale(box_x_axis_robot, local_push_direction[0]),
                self.vector_scale(box_y_axis_robot, local_push_direction[1])
            ),
            self.vector_scale(box_z_axis_robot, local_push_direction[2])
        ))
        box_yaw_robot = self.quat_to_yaw(box_orientation_robot)

        self.get_logger().info(
            f'Oggetto in frame di calcolo: x={box_position_robot[0]:.6f}, '
            f'y={box_position_robot[1]:.6f}, z={box_position_robot[2]:.6f}, '
            f'yaw={box_yaw_robot:.6f} rad'
        )
        self.get_logger().info(
            f'Push contact in frame di calcolo: x={contact_position_robot[0]:.6f}, '
            f'y={contact_position_robot[1]:.6f}, z={contact_position_robot[2]:.6f}'
        )
        self.get_logger().info(
            f'Push direction in frame di calcolo: x={push_direction_robot[0]:.6f}, '
            f'y={push_direction_robot[1]:.6f}, z={push_direction_robot[2]:.6f}'
        )

        return {
            'contact_position': contact_position_robot,
            'push_direction': push_direction_robot,
        }

    def compute_all_phase_targets(self):
        target = self.push_target_components()
        if target is None:
            return None

        contact_position = target['contact_position']
        push_direction = target['push_direction']

        start_pose = (
            self.d1_start_x, self.d1_start_y, self.d1_start_z,
            self.d2_start_x, self.d2_start_y, self.d2_start_z,
        )
        pre_push = self.vector_add(
            contact_position,
            self.vector_scale(push_direction, -float(self.pre_push_distance))
        )
        push_end = self.vector_add(
            contact_position,
            self.vector_scale(push_direction, float(self.push_distance))
        )
        retreat = self.vector_add(
            push_end,
            self.vector_scale(push_direction, -float(self.retreat_distance))
        )

        phases = {
            'start_pose': start_pose,
            'pre_push': pre_push,
            'contact': contact_position,
            'push': push_end,
        }
        if abs(float(self.retreat_distance)) > 0.0:
            phases['retreat'] = retreat

        self.log_phase_targets(phases)
        return phases

    def log_phase_targets(self, phases):
        self.get_logger().info('=== Target push calcolati ===')
        self.get_logger().info(
            f'braccio push attivo = {self.push_arm()}, braccio disattivato = {self.inactive_arm()}'
        )
        self.get_logger().info(f'target frame = {self.base_frame}')
        self.get_logger().info(f'push dagana = {self.push_dagana}')
        for phase_name, values in phases.items():
            if phase_name == 'start_pose':
                x1, y1, z1, x2, y2, z2 = values
                self.get_logger().info(
                    f'{phase_name}: d1=({x1:.6f}, {y1:.6f}, {z1:.6f}), '
                    f'd2=({x2:.6f}, {y2:.6f}, {z2:.6f})'
                )
            else:
                x, y, z = values
                self.get_logger().info(
                    f'{phase_name}: d{self.push_dagana}=({x:.6f}, {y:.6f}, {z:.6f})'
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

        box_orientation_robot = self.box_orientation
        if box_orientation_robot is None:
            return d1_q, d2_q

        box_yaw_robot = self.quat_to_yaw(box_orientation_robot)
        yaw_q = self.yaw_to_quat(box_yaw_robot)
        return (
            self.quat_normalize(self.quat_multiply(yaw_q, d1_q)),
            self.quat_normalize(self.quat_multiply(yaw_q, d2_q)),
        )

    def push_client(self):
        if self.push_dagana == 1:
            return self.client_1
        return self.client_2

    def push_goal_orientation(self):
        d1_q, d2_q = self.current_goal_orientations()
        if self.push_dagana == 1:
            return d1_q
        return d2_q

    def push_dagana_label(self):
        return f'dagana_{self.push_dagana}_tcp'

    def send_goal_and_wait(self, phase_name, goal):
        self.get_logger().info(f'=== Starting {phase_name} ===')
        client = self.push_client()
        dagana_label = self.push_dagana_label()
        client.wait_for_server()

        fut_send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut_send)

        gh = fut_send.result()
        if gh is None or not gh.accepted:
            raise RuntimeError(f'{phase_name}: goal {dagana_label} rejected.')

        fut_res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, fut_res)

        if fut_res.result() is None:
            raise RuntimeError(f'{phase_name}: no result {dagana_label}.')
        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_start_pose(self, phase_targets, time_s):
        x1, y1, z1, x2, y2, z2 = phase_targets
        d1_q, d2_q = self.current_goal_orientations()
        if self.push_dagana == 1:
            goal = self.make_goal(x1, y1, z1, *d1_q, time_s=time_s)
        else:
            goal = self.make_goal(x2, y2, z2, *d2_q, time_s=time_s)
        self.send_goal_and_wait('START_POSE', goal)

    def run_phase(self, phase_name, phase_target, time_s):
        x, y, z = phase_target
        q = self.push_goal_orientation()
        goal = self.make_goal(x, y, z, *q, time_s=time_s)
        self.send_goal_and_wait(phase_name, goal)

    def execute(self):
        task_weights_configured = False
        inactive_arm_disabled = False
        try:
            self.get_logger().info(
                f'Disattivo {self.inactive_arm()} per push con {self.push_arm()}...'
            )
            self.set_arm_active(self.inactive_arm(), False)
            inactive_arm_disabled = True
            self.set_arm_active(self.push_arm(), True)

            self.get_logger().info('Lettura pose da file...')
            self.read_input_poses()

            phases = self.compute_all_phase_targets()
            if phases is None:
                raise RuntimeError('Impossibile calcolare i target assoluti delle fasi.')

            self.get_logger().info('Configuro pesi task arm...')
            self.configure_task_weights()
            task_weights_configured = True

            self.run_start_pose(phases['start_pose'], self.time_start_pose)
            self.run_phase('PHASE 1 - PRE_PUSH', phases['pre_push'], self.time_pre_push)
            self.run_phase('PHASE 2 - CONTACT', phases['contact'], self.time_contact)
            self.run_phase('PHASE 3 - PUSH', phases['push'], self.time_push)
            if 'retreat' in phases:
                self.run_phase('PHASE 4 - RETREAT', phases['retreat'], self.time_retreat)
            self.get_logger().info('Push completed.')
        finally:
            if task_weights_configured and self.restore_orientation_weight_at_end:
                try:
                    self.get_logger().info('Ripristino pesi 6D completi dei task arm...')
                    self.restore_full_pose_task_weights()
                    self.get_logger().info('Pesi 6D arm ripristinati.')
                except Exception as exc:
                    self.get_logger().warn(f'Ripristino pesi 6D arm fallito: {exc}')
            if inactive_arm_disabled:
                try:
                    self.get_logger().info(
                        f'Riattivo {self.inactive_arm()} dopo il push...'
                    )
                    self.set_arm_active(self.inactive_arm(), True)
                    self.get_logger().info(f'{self.inactive_arm()} riattivato.')
                except Exception as exc:
                    self.get_logger().warn(
                        f'Riattivazione {self.inactive_arm()} fallita: {exc}'
                    )


def main():
    rclpy.init()
    node = CentauroDirectPush()

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
