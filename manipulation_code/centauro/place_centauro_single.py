#!/usr/bin/env python3

"""
Centauro single-arm Dagana place from a target pose already expressed in target frame.

Run from this folder:

python3 place_centauro_single.py --ros-args \
    -p object_pose_file:=object_pose.txt \
    -p base_frame:=world \
    -p place_dagana:=2 \
    -p place_offset_x:=0.0 \
    -p place_offset_y:=0.0 \
    -p place_offset_z:=0.02 \
    -p place_clearance_z:=0.15 \
    -p release_retreat_z:=0.15 \
    -p dagana_open_pos:=1.25 \
    -p dagana_close_pos:=0.0

What it does:
    1. Reads the desired place pose from a text file already expressed in base_frame.
    2. Deactivates the non-placing Dagana Cartesian task.
    3. Moves only the selected Dagana to a pre-place pose above the target.
    4. Moves down to the place pose.
    5. Opens the selected Dagana clamp to release the object.
    6. Retreats upward and reactivates the non-placing Dagana Cartesian task.

This file is standalone: it does not import the other helper scripts.
"""

from pathlib import Path
import math
from typing import List

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cartesian_interface_ros.action import ReachPose
from cartesian_interface_ros.srv import SetTaskActive, SetWeight
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from xbot_msgs.msg import JointCommand
try:
    from xbot_msgs.msg import JointState as XbotJointState
except ImportError:
    XbotJointState = None


class CentauroSingleDaganaPlace(Node):

    def __init__(self):
        super().__init__('centauro_single_dagana_place')

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

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.clamp_pub = self.create_publisher(JointCommand, '/xbotcore/command', qos)
        self.clamp_joint_names = ['dagana_1_claw_joint', 'dagana_2_claw_joint']
        self.current_joint_positions = {}
        self.logged_joint_state_format = False

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

        self.declare_parameter('place_dagana', 2)
        self.declare_parameter('place_offset_x', 0.0)
        self.declare_parameter('place_offset_y', 0.0)
        self.declare_parameter('place_offset_z', 0.0)
        self.declare_parameter('place_clearance_z', 0.15)
        self.declare_parameter('release_retreat_z', 0.12)
        self.declare_parameter('release_retreat_x', 0.0)
        self.declare_parameter('release_retreat_y', 0.0)

        self.declare_parameter('dagana_open_pos', 0.0)
        self.declare_parameter('dagana_close_pos', 0.6)
        self.declare_parameter('joint_state_topic', '/xbotcore/joint_states')
        self.declare_parameter('joint_state_timeout', 2.0)
        self.declare_parameter('fallback_to_initial_clamp_pos', False)
        self.declare_parameter('dagana_initial_pos', 0.6)
        self.declare_parameter('inactive_dagana_hold_pos', 0.0)
        self.declare_parameter('clamp_command_duration', 2.0)
        self.declare_parameter('clamp_command_rate', 100.0)
        self.declare_parameter('clamp_max_speed', 2.0)
        self.declare_parameter('clamp_max_step', 0.05)
        self.declare_parameter('clamp_ctrl_mode', 1)

        self.declare_parameter('align_orientation_to_box_yaw', True)
        self.declare_parameter('constrain_orientation', True)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('restore_orientation_weight_at_end', False)
        self.declare_parameter('set_active_timeout', 20.0)
        self.declare_parameter('set_weight_timeout', 20.0)

        self.declare_parameter('time_pre_place', 3.0)
        self.declare_parameter('time_place', 3.0)
        self.declare_parameter('time_retreat', 3.0)

        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.joint_state_timeout = self.get_parameter('joint_state_timeout').value
        self.fallback_to_initial_clamp_pos = self.get_parameter('fallback_to_initial_clamp_pos').value
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_callback,
            joint_state_qos,
        )
        self.xbot_joint_state_sub = None
        if XbotJointState is not None:
            self.xbot_joint_state_sub = self.create_subscription(
                XbotJointState,
                self.joint_state_topic,
                self.joint_state_callback,
                joint_state_qos,
            )

        self.object_pose_file = self.get_parameter('object_pose_file').value
        self.d1_set_active_service = self.get_parameter('d1_set_active_service').value
        self.d2_set_active_service = self.get_parameter('d2_set_active_service').value
        self.d1_set_weight_service = self.get_parameter('d1_set_weight_service').value
        self.d2_set_weight_service = self.get_parameter('d2_set_weight_service').value
        self.base_frame = str(self.get_parameter('base_frame').value)

        if self.d1_set_active_service != '/cartesian/dagana_1_tcp/set_active':
            self.set_active_client_1 = self.create_client(SetTaskActive, self.d1_set_active_service)
        if self.d2_set_active_service != '/cartesian/dagana_2_tcp/set_active':
            self.set_active_client_2 = self.create_client(SetTaskActive, self.d2_set_active_service)
        if self.d1_set_weight_service != '/cartesian/dagana_1_tcp/set_weight':
            self.set_weight_client_1 = self.create_client(SetWeight, self.d1_set_weight_service)
        if self.d2_set_weight_service != '/cartesian/dagana_2_tcp/set_weight':
            self.set_weight_client_2 = self.create_client(SetWeight, self.d2_set_weight_service)

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

        self.place_dagana = int(self.get_parameter('place_dagana').value)
        self.place_offset_x = self.get_parameter('place_offset_x').value
        self.place_offset_y = self.get_parameter('place_offset_y').value
        self.place_offset_z = self.get_parameter('place_offset_z').value
        self.place_clearance_z = self.get_parameter('place_clearance_z').value
        self.release_retreat_z = self.get_parameter('release_retreat_z').value
        self.release_retreat_x = self.get_parameter('release_retreat_x').value
        self.release_retreat_y = self.get_parameter('release_retreat_y').value

        self.dagana_open_pos = self.get_parameter('dagana_open_pos').value
        self.dagana_close_pos = self.get_parameter('dagana_close_pos').value
        self.dagana_initial_pos = self.get_parameter('dagana_initial_pos').value
        self.inactive_dagana_hold_pos = self.get_parameter('inactive_dagana_hold_pos').value
        self.clamp_command_duration = self.get_parameter('clamp_command_duration').value
        self.clamp_command_rate = self.get_parameter('clamp_command_rate').value
        self.clamp_max_speed = self.get_parameter('clamp_max_speed').value
        self.clamp_max_step = self.get_parameter('clamp_max_step').value
        self.clamp_ctrl_mode = int(self.get_parameter('clamp_ctrl_mode').value)

        self.align_orientation_to_box_yaw = self.get_parameter('align_orientation_to_box_yaw').value
        self.constrain_orientation = self.get_parameter('constrain_orientation').value
        self.position_weight = self.get_parameter('position_weight').value
        self.orientation_weight = self.get_parameter('orientation_weight').value
        self.restore_orientation_weight_at_end = self.get_parameter(
            'restore_orientation_weight_at_end'
        ).value
        self.set_active_timeout = self.get_parameter('set_active_timeout').value
        self.set_weight_timeout = self.get_parameter('set_weight_timeout').value

        self.time_pre_place = self.get_parameter('time_pre_place').value
        self.time_place = self.get_parameter('time_place').value
        self.time_retreat = self.get_parameter('time_retreat').value

        if self.place_dagana not in (1, 2):
            raise ValueError(f'place_dagana={self.place_dagana} non valido. Usa 1 oppure 2.')
        if float(self.clamp_command_rate) <= 0.0:
            raise ValueError('clamp_command_rate deve essere positivo.')
        if float(self.clamp_max_speed) <= 0.0:
            raise ValueError('clamp_max_speed deve essere positivo.')
        if float(self.clamp_max_step) <= 0.0:
            raise ValueError('clamp_max_step deve essere positivo.')

        self.box_position = None
        self.box_orientation = None
        self.last_clamp_positions = [
            float(self.inactive_dagana_hold_pos),
            float(self.inactive_dagana_hold_pos),
        ]
        self.last_clamp_positions[self.place_dagana - 1] = float(self.dagana_initial_pos)

    def place_arm(self):
        return 'left' if self.place_dagana == 1 else 'right'

    def inactive_arm(self):
        return 'right' if self.place_dagana == 1 else 'left'

    def set_active_client_for_arm(self, arm):
        return self.set_active_client_1 if arm == 'left' else self.set_active_client_2

    def active_service_name_for_arm(self, arm):
        return self.d1_set_active_service if arm == 'left' else self.d2_set_active_service

    def set_arm_active(self, arm, active):
        client = self.set_active_client_for_arm(arm)
        service_name = self.active_service_name_for_arm(arm)
        self.get_logger().info(f'{"Attivo" if active else "Disattivo"} task {arm}: {service_name}')

        if not client.wait_for_service(timeout_sec=float(self.set_active_timeout)):
            raise RuntimeError(
                f'Service {service_name} non disponibile dopo {self.set_active_timeout:.1f} s.'
            )

        request = SetTaskActive.Request()
        request.activation_state = bool(active)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.set_active_timeout))

        response = future.result()
        if response is None:
            raise RuntimeError(
                f'Chiamata a {service_name} scaduta dopo {self.set_active_timeout:.1f} s.'
            )
        if not response.success:
            raise RuntimeError(f'{service_name} ha risposto success=false: {response.message}')
        self.get_logger().info(f'{service_name}: {response.message}')

    def set_weight_client_for_arm(self, arm):
        return self.set_weight_client_1 if arm == 'left' else self.set_weight_client_2

    def weight_service_name_for_arm(self, arm):
        return self.d1_set_weight_service if arm == 'left' else self.d2_set_weight_service

    def task_weight_for_current_orientation_mode(self):
        pos_w = float(self.position_weight)
        ori_w = pos_w if self.constrain_orientation else float(self.orientation_weight)
        return [pos_w, pos_w, pos_w, ori_w, ori_w, ori_w]

    def full_pose_task_weight(self):
        pos_w = float(self.position_weight)
        return [pos_w, pos_w, pos_w, pos_w, pos_w, pos_w]

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
                f'Service {service_name} non disponibile dopo {self.set_weight_timeout:.1f} s.'
            )

        request = SetWeight.Request()
        request.weight = weight
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.set_weight_timeout))

        response = future.result()
        if response is None:
            raise RuntimeError(
                f'Chiamata a {service_name} scaduta dopo {self.set_weight_timeout:.1f} s.'
            )
        if not response.success:
            raise RuntimeError(f'{service_name} ha risposto success=false: {response.message}')
        self.get_logger().info(f'{service_name}: {response.message}')

    def configure_task_weights(self):
        self.set_arm_task_weight(self.place_arm())

    def restore_full_pose_task_weights(self):
        errors = []
        weight = self.full_pose_task_weight()
        for arm in ('left', 'right'):
            try:
                self.set_arm_task_weight(arm, weight)
            except Exception as exc:
                errors.append(f'{arm}: {exc}')
        if errors:
            raise RuntimeError('Ripristino incompleto dei pesi 6D arm: ' + '; '.join(errors))

    def joint_state_callback(self, msg):
        names = list(getattr(msg, 'name', []))
        position_field = None
        positions = []
        for candidate in ('position', 'link_position', 'motor_position', 'q', 'pos'):
            if hasattr(msg, candidate):
                values = list(getattr(msg, candidate))
                if len(values) == len(names):
                    position_field = candidate
                    positions = values
                    break

        if not self.logged_joint_state_format:
            self.logged_joint_state_format = True
            field_names = list(msg.get_fields_and_field_types().keys())
            preview_names = ', '.join(names[:8]) if names else '<nessun nome>'
            self.get_logger().info(
                f'Formato joint state ricevuto: type={type(msg).__name__}, '
                f'fields={field_names}, position_field={position_field}, '
                f'first_names=[{preview_names}]'
            )

        if not names or not positions:
            return

        for name, position in zip(names, positions):
            self.current_joint_positions[name] = float(position)

    def read_current_clamp_positions(self):
        missing = set(self.clamp_joint_names)
        deadline = self.get_clock().now() + Duration(seconds=float(self.joint_state_timeout))
        self.get_logger().info(
            f'Leggo posizione corrente clamp da {self.joint_state_topic}...'
        )

        while rclpy.ok() and missing and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            missing = {
                name for name in self.clamp_joint_names
                if name not in self.current_joint_positions
            }

        if missing:
            if not self.fallback_to_initial_clamp_pos:
                raise RuntimeError(
                    f'Non ho ricevuto posizione corrente per: {", ".join(sorted(missing))}. '
                    f'Controlla joint_state_topic={self.joint_state_topic} oppure imposta '
                    f'fallback_to_initial_clamp_pos:=true.'
                )
            self.get_logger().warn(
                'Uso dagana_initial_pos/inactive_dagana_hold_pos perche non ho ricevuto tutti i joint state.'
            )
            positions = [
                float(self.inactive_dagana_hold_pos),
                float(self.inactive_dagana_hold_pos),
            ]
            positions[self.place_dagana - 1] = float(self.dagana_initial_pos)
            return positions

        positions = [
            self.current_joint_positions[self.clamp_joint_names[0]],
            self.current_joint_positions[self.clamp_joint_names[1]],
        ]
        self.get_logger().info(
            f'Posizione corrente clamp: {self.clamp_joint_names[0]}={positions[0]:.4f}, '
            f'{self.clamp_joint_names[1]}={positions[1]:.4f}'
        )
        return positions

    def make_clamp_msg(self, positions: List[float]) -> JointCommand:
        if len(positions) != 2:
            raise ValueError('positions deve contenere esattamente 2 valori')

        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = list(self.clamp_joint_names)
        msg.position = [float(positions[0]), float(positions[1])]
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        msg.stiffness = [0.0, 0.0]
        msg.damping = [0.0, 0.0]
        msg.ctrl_mode = [self.clamp_ctrl_mode, self.clamp_ctrl_mode]
        msg.aux_name = ''
        msg.aux = []
        return msg

    def publish_clamp_for_duration(self, positions: List[float], label: str):
        duration_sec = float(self.clamp_command_duration)
        rate_hz = float(self.clamp_command_rate)
        max_speed = float(self.clamp_max_speed)
        period = 1.0 / rate_hz
        n_steps = max(1, int(duration_sec * rate_hz))
        target_positions = [float(positions[0]), float(positions[1])]
        current_positions = list(self.last_clamp_positions)
        max_step = min(max_speed * period, float(self.clamp_max_step))

        self.get_logger().info(
            f'{label}: target {self.clamp_joint_names[0]}={target_positions[0]:.4f}, '
            f'{self.clamp_joint_names[1]}={target_positions[1]:.4f}, '
            f'duration={duration_sec:.2f}s, rate={rate_hz:.1f}Hz, '
            f'max_speed={max_speed:.3f}, max_step={max_step:.4f}'
        )
        for _ in range(n_steps):
            for index, target in enumerate(target_positions):
                delta = target - current_positions[index]
                if abs(delta) <= max_step:
                    current_positions[index] = target
                else:
                    current_positions[index] += math.copysign(max_step, delta)

            msg = self.make_clamp_msg(current_positions)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.clamp_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.get_clock().sleep_for(Duration(seconds=period))
        self.last_clamp_positions = current_positions

        remaining = [
            abs(target_positions[index] - current_positions[index])
            for index in range(len(target_positions))
        ]
        if max(remaining) > 1e-6:
            self.get_logger().warn(
                f'{label}: target clamp non raggiunto entro clamp_command_duration; '
                f'aumenta clamp_command_duration o clamp_max_speed. '
                f'residuo massimo={max(remaining):.6f}'
            )

    def selected_clamp_positions(self, selected_position):
        current_positions = self.read_current_clamp_positions()
        self.last_clamp_positions = list(current_positions)
        target_positions = list(current_positions)
        target_positions[self.place_dagana - 1] = float(selected_position)
        return target_positions

    def open_selected_dagana(self):
        self.publish_clamp_for_duration(
            self.selected_clamp_positions(self.dagana_open_pos),
            f'Apro dagana_{self.place_dagana}',
        )

    def close_selected_dagana(self):
        self.publish_clamp_for_duration(
            self.selected_clamp_positions(self.dagana_close_pos),
            f'Chiudo dagana_{self.place_dagana}',
        )

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
            self.quat_conjugate(q),
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
                'place_position',
                'place_pose_position',
                'target_position',
                'target_pose_position',
                'position',
            ),
            'object_yaw': (
                'object_yaw',
                'object_pose_yaw',
                'box_yaw',
                'yaw',
                'place_yaw',
                'place_pose_yaw',
                'target_yaw',
                'target_pose_yaw',
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

        if 'object_position' not in parsed:
            raise ValueError(f'{path}: chiavi mancanti: object_position')
        if len(parsed['object_position']) != 3:
            raise ValueError(
                f'{path}: "object_position" deve avere 3 valori, '
                f'ne ha {len(parsed["object_position"])}'
            )
        for key in ('object_yaw', 'object_yaw_world'):
            if key in parsed and len(parsed[key]) != 1:
                raise ValueError(f'{path}: "{key}" deve avere 1 valore, ne ha {len(parsed[key])}')
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
        self.get_logger().info(f'{label} position: x={px:.6f}, y={py:.6f}, z={pz:.6f}')
        self.get_logger().info(
            f'{label} orientation: q=({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}), '
            f'yaw={yaw:.6f} rad'
        )

    def place_target_components(self):
        if self.box_position is None or self.box_orientation is None:
            return None

        box_x_axis = self.quat_rotate_vector(self.box_orientation, (1.0, 0.0, 0.0))
        box_y_axis = self.quat_rotate_vector(self.box_orientation, (0.0, 1.0, 0.0))
        box_z_axis = self.quat_rotate_vector(self.box_orientation, (0.0, 0.0, 1.0))

        place_offset = self.vector_add(
            self.vector_add(
                self.vector_scale(box_x_axis, self.place_offset_x),
                self.vector_scale(box_y_axis, self.place_offset_y),
            ),
            self.vector_scale(box_z_axis, self.place_offset_z),
        )
        place_position = self.vector_add(self.box_position, place_offset)

        self.get_logger().info(
            f'Place point in frame di calcolo: x={place_position[0]:.6f}, '
            f'y={place_position[1]:.6f}, z={place_position[2]:.6f}'
        )
        return {
            'place_position': place_position,
        }

    def compute_all_phase_targets(self):
        target = self.place_target_components()
        if target is None:
            return None

        place_position = target['place_position']
        pre_place = self.vector_add(place_position, (0.0, 0.0, float(self.place_clearance_z)))
        retreat = self.vector_add(
            place_position,
            (
                float(self.release_retreat_x),
                float(self.release_retreat_y),
                float(self.release_retreat_z),
            ),
        )

        phases = {
            'pre_place': pre_place,
            'place': place_position,
            'retreat': retreat,
        }

        self.log_phase_targets(phases)
        return phases

    def log_phase_targets(self, phases):
        self.get_logger().info('=== Target single-arm place calcolati ===')
        self.get_logger().info(
            f'braccio place attivo = {self.place_arm()}, braccio disattivato = {self.inactive_arm()}'
        )
        self.get_logger().info(f'target frame = {self.base_frame}')
        self.get_logger().info(f'place dagana = {self.place_dagana}')
        for phase_name, values in phases.items():
            x, y, z = values
            self.get_logger().info(
                f'{phase_name}: d{self.place_dagana}=({x:.6f}, {y:.6f}, {z:.6f})'
            )

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
        if not self.align_orientation_to_box_yaw or self.box_orientation is None:
            return d1_q, d2_q

        box_yaw = self.quat_to_yaw(self.box_orientation)
        yaw_q = self.yaw_to_quat(box_yaw)
        return (
            self.quat_normalize(self.quat_multiply(yaw_q, d1_q)),
            self.quat_normalize(self.quat_multiply(yaw_q, d2_q)),
        )

    def place_client(self):
        return self.client_1 if self.place_dagana == 1 else self.client_2

    def place_goal_orientation(self):
        d1_q, d2_q = self.current_goal_orientations()
        return d1_q if self.place_dagana == 1 else d2_q

    def place_dagana_label(self):
        return f'dagana_{self.place_dagana}_tcp'

    def send_goal_and_wait(self, phase_name, goal):
        self.get_logger().info(f'=== Starting {phase_name} ===')
        client = self.place_client()
        dagana_label = self.place_dagana_label()
        client.wait_for_server()

        fut_send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut_send)

        goal_handle = fut_send.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f'{phase_name}: goal {dagana_label} rejected.')

        fut_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, fut_result)

        if fut_result.result() is None:
            raise RuntimeError(f'{phase_name}: no result {dagana_label}.')
        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase(self, phase_name, phase_target, time_s):
        x, y, z = phase_target
        q = self.place_goal_orientation()
        goal = self.make_goal(x, y, z, *q, time_s=time_s)
        self.send_goal_and_wait(phase_name, goal)

    def execute(self):
        task_weights_configured = False
        inactive_arm_disabled = False
        try:
            self.get_logger().info(
                f'Disattivo {self.inactive_arm()} per place con {self.place_arm()}...'
            )
            self.set_arm_active(self.inactive_arm(), False)
            inactive_arm_disabled = True
            self.set_arm_active(self.place_arm(), True)

            self.get_logger().info('Lettura pose da file...')
            self.read_input_poses()

            phases = self.compute_all_phase_targets()
            if phases is None:
                raise RuntimeError('Impossibile calcolare i target assoluti delle fasi di place.')

            self.get_logger().info('Configuro pesi task arm...')
            self.configure_task_weights()
            task_weights_configured = True

            self.run_phase('PHASE 1 - PRE_PLACE', phases['pre_place'], self.time_pre_place)
            self.run_phase('PHASE 2 - PLACE', phases['place'], self.time_place)
            self.open_selected_dagana()
            self.run_phase('PHASE 3 - RETREAT_AFTER_RELEASE', phases['retreat'], self.time_retreat)
            self.get_logger().info('Single-arm Dagana place completed.')
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
                        f'Riattivo {self.inactive_arm()} dopo il place...'
                    )
                    self.set_arm_active(self.inactive_arm(), True)
                    self.get_logger().info(f'{self.inactive_arm()} riattivato.')
                except Exception as exc:
                    self.get_logger().warn(f'Riattivazione {self.inactive_arm()} fallita: {exc}')


def main():
    rclpy.init()
    node = CentauroSingleDaganaPlace()

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
