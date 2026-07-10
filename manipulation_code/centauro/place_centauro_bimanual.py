#!/usr/bin/env python3

"""
Centauro dual-arm place test from an object pose already expressed in target frame.

Run from this folder:

python3 place_centauro_bimanual.py --ros-args \
    -p object_pose_file:=object_pose.txt \
    -p base_frame:=world \
    -p place_grasp_offset_x:=0.12 \
    -p place_grasp_offset_y:=-0.03 \
    -p place_grasp_offset_z:=0.02 \
    -p place_clearance_z:=0.15 \
    -p release_distance:=0.1 \
    -p retreat_z_after_release:=0.15 \
    -p constrain_orientation:=true \
    -p box_width:=0.514

Input pose file format:
    Values are whitespace separated. The position is expressed directly in the
    Cartesio task target frame used by the ReachPose action goals.

        object.position: 0.95 0.00 0.70
        object.yaw: 0.0

    Accepted aliases:
        object.position, object_pose.position, box.position, position
        object.yaw, object_pose.yaw, box.yaw, yaw
        object.yaw_world, object_yaw_task, object_yaw_cartesian_world,
        box.yaw_world, yaw_world

What it does:
    1. Reads the desired object/place pose from object_pose.txt.
    2. Builds a task-base-aligned object frame using object.yaw in base_frame.
    3. Moves both Dagana TCPs to pre-place positions above the object/place pose.
    4. Moves both TCPs down to place height.
    5. Opens symmetrically along the object local Y axis.

This script has no Gazebo dependency. It keeps the same direct object pose
convention, Centauro action servers, and task weight handling used by
grasp_centauro_test_direct.py and push_box_xyz_direct.py.
"""

from pathlib import Path
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cartesian_interface_ros.action import ReachPose
from cartesian_interface_ros.srv import SetWeight
from geometry_msgs.msg import Pose


class CentauroDirectPlace(Node):

    def __init__(self):
        super().__init__('centauro_direct_place_test')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_tcp/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_tcp/reach')
        self.set_weight_client_1 = self.create_client(
            SetWeight, '/cartesian/dagana_1_tcp/set_weight'
        )
        self.set_weight_client_2 = self.create_client(
            SetWeight, '/cartesian/dagana_2_tcp/set_weight'
        )

        self.declare_parameter('object_pose_file', 'object_pose.txt')
        self.declare_parameter('d1_set_weight_service', '/cartesian/dagana_1_tcp/set_weight')
        self.declare_parameter('d2_set_weight_service', '/cartesian/dagana_2_tcp/set_weight')
        self.declare_parameter('base_frame', 'world')

        self.declare_parameter('d1_qx', 0.0)
        self.declare_parameter('d1_qy', 0.7)
        self.declare_parameter('d1_qz', 0.0)
        self.declare_parameter('d1_qw', 0.7)
        self.declare_parameter('d2_qx', 0.0)
        self.declare_parameter('d2_qy', 0.7)
        self.declare_parameter('d2_qz', 0.0)
        self.declare_parameter('d2_qw', 0.7)

        self.declare_parameter('box_width', 0.35)
        self.declare_parameter('place_clearance_z', 0.15)
        self.declare_parameter('place_grasp_offset_x', 0.0)
        self.declare_parameter('place_grasp_offset_y', 0.0)
        self.declare_parameter('place_grasp_offset_z', 0.0)
        self.declare_parameter('place_d1_y_bias', 0.0)
        self.declare_parameter('place_d2_y_bias', 0.0)
        self.declare_parameter('release_distance', 0.08)
        self.declare_parameter('release_extra_z', 0.0)
        self.declare_parameter('retreat_z_after_release', 0.0)
        self.declare_parameter('align_orientation_to_box_yaw', True)

        self.declare_parameter('constrain_orientation', True)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('restore_orientation_weight_at_end', False)
        self.declare_parameter('set_weight_timeout', 20.0)

        self.declare_parameter('time_phase_pre_place', 3.0)
        self.declare_parameter('time_phase_place', 3.0)
        self.declare_parameter('time_phase_release', 3.0)
        self.declare_parameter('time_phase_retreat', 3.0)

        self.object_pose_file = self.get_parameter('object_pose_file').value
        self.d1_set_weight_service = self.get_parameter('d1_set_weight_service').value
        self.d2_set_weight_service = self.get_parameter('d2_set_weight_service').value
        self.base_frame = str(self.get_parameter('base_frame').value)

        if self.d1_set_weight_service != '/cartesian/dagana_1_tcp/set_weight':
            self.set_weight_client_1 = self.create_client(SetWeight, self.d1_set_weight_service)
        if self.d2_set_weight_service != '/cartesian/dagana_2_tcp/set_weight':
            self.set_weight_client_2 = self.create_client(SetWeight, self.d2_set_weight_service)

        self.d1_qx = self.get_parameter('d1_qx').value
        self.d1_qy = self.get_parameter('d1_qy').value
        self.d1_qz = self.get_parameter('d1_qz').value
        self.d1_qw = self.get_parameter('d1_qw').value
        self.d2_qx = self.get_parameter('d2_qx').value
        self.d2_qy = self.get_parameter('d2_qy').value
        self.d2_qz = self.get_parameter('d2_qz').value
        self.d2_qw = self.get_parameter('d2_qw').value

        self.box_width = self.get_parameter('box_width').value
        self.place_clearance_z = self.get_parameter('place_clearance_z').value
        self.place_grasp_offset_x = self.get_parameter('place_grasp_offset_x').value
        self.place_grasp_offset_y = self.get_parameter('place_grasp_offset_y').value
        self.place_grasp_offset_z = self.get_parameter('place_grasp_offset_z').value
        self.place_d1_y_bias = self.get_parameter('place_d1_y_bias').value
        self.place_d2_y_bias = self.get_parameter('place_d2_y_bias').value
        self.release_distance = self.get_parameter('release_distance').value
        self.release_extra_z = self.get_parameter('release_extra_z').value
        self.retreat_z_after_release = self.get_parameter('retreat_z_after_release').value
        self.align_orientation_to_box_yaw = self.get_parameter(
            'align_orientation_to_box_yaw'
        ).value

        self.constrain_orientation = self.get_parameter('constrain_orientation').value
        self.position_weight = self.get_parameter('position_weight').value
        self.orientation_weight = self.get_parameter('orientation_weight').value
        self.restore_orientation_weight_at_end = self.get_parameter(
            'restore_orientation_weight_at_end'
        ).value
        self.set_weight_timeout = self.get_parameter('set_weight_timeout').value

        self.time_phase_pre_place = self.get_parameter('time_phase_pre_place').value
        self.time_phase_place = self.get_parameter('time_phase_place').value
        self.time_phase_release = self.get_parameter('time_phase_release').value
        self.time_phase_retreat = self.get_parameter('time_phase_retreat').value

        self.box_position = None
        self.box_orientation = None

    def set_weight_client_for_arm(self, arm):
        if arm == 'left':
            return self.set_weight_client_1
        return self.set_weight_client_2

    def weight_service_name_for_arm(self, arm):
        if arm == 'left':
            return self.d1_set_weight_service
        return self.d2_set_weight_service

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
        self.set_arm_task_weight('left')
        self.set_arm_task_weight('right')

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

    def quat_normalize(self, q):
        x, y, z, w = q
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm, w / norm)

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def quat_conjugate(self, q):
        x, y, z, w = q
        return (-x, -y, -z, w)

    def quat_rotate_vector(self, q, v):
        q = self.quat_normalize(q)
        rotated = self.quat_multiply(
            self.quat_multiply(q, (v[0], v[1], v[2], 0.0)),
            self.quat_conjugate(q)
        )
        return (rotated[0], rotated[1], rotated[2])

    def yaw_to_quat(self, yaw):
        half = 0.5 * yaw
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def vector_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def vector_scale(self, v, scale):
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    def quat_to_yaw(self, q):
        x, y, z, w = self.quat_normalize(q)
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)

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

        required_lengths = {
            'object_position': 3,
        }
        optional_lengths = {
            'object_yaw': 1,
            'object_yaw_world': 1,
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
        for key, expected_length in optional_lengths.items():
            if key in parsed and len(parsed[key]) != expected_length:
                raise ValueError(
                    f'{path}: "{key}" deve avere {expected_length} valore, '
                    f'ne ha {len(parsed[key])}'
                )
        return parsed

    def read_input_poses(self):
        poses = self.parse_object_pose_file()

        object_position = poses['object_position']
        object_yaw = poses.get('object_yaw')
        object_yaw_world = poses.get('object_yaw_world')

        self.box_position = object_position
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
            f'object_in_{self.base_frame} position: x={object_position[0]:.6f}, '
            f'y={object_position[1]:.6f}, z={object_position[2]:.6f}'
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

    def box_axes(self):
        box_q = self.box_orientation
        return {
            'x': self.quat_rotate_vector(box_q, (1.0, 0.0, 0.0)),
            'y': self.quat_rotate_vector(box_q, (0.0, 1.0, 0.0)),
            'z': self.quat_rotate_vector(box_q, (0.0, 0.0, 1.0)),
        }

    def compute_all_phase_targets(self):
        if self.box_position is None or self.box_orientation is None:
            return None

        axes = self.box_axes()
        half_box = 0.5 * self.box_width
        place_center = self.vector_add(
            self.vector_add(
                self.vector_add(
                    self.box_position,
                    self.vector_scale(axes['x'], self.place_grasp_offset_x),
                ),
                self.vector_scale(axes['y'], self.place_grasp_offset_y),
            ),
            self.vector_scale(axes['z'], self.place_grasp_offset_z),
        )
        pre_center = self.vector_add(place_center, (0.0, 0.0, self.place_clearance_z))

        pre_d1 = self.vector_add(
            pre_center,
            self.vector_scale(axes['y'], half_box + self.place_d1_y_bias),
        )
        pre_d2 = self.vector_add(
            pre_center,
            self.vector_scale(axes['y'], -half_box + self.place_d2_y_bias),
        )
        place_d1 = self.vector_add(
            place_center,
            self.vector_scale(axes['y'], half_box + self.place_d1_y_bias),
        )
        place_d2 = self.vector_add(
            place_center,
            self.vector_scale(axes['y'], -half_box + self.place_d2_y_bias),
        )
        release_d1 = self.vector_add(
            place_d1,
            self.vector_add(
                self.vector_scale(axes['y'], self.release_distance),
                (0.0, 0.0, self.release_extra_z),
            ),
        )
        release_d2 = self.vector_add(
            place_d2,
            self.vector_add(
                self.vector_scale(axes['y'], -self.release_distance),
                (0.0, 0.0, self.release_extra_z),
            ),
        )
        retreat_d1 = self.vector_add(release_d1, (0.0, 0.0, self.retreat_z_after_release))
        retreat_d2 = self.vector_add(release_d2, (0.0, 0.0, self.retreat_z_after_release))

        phases = {
            'phase_pre_place': (*pre_d1, *pre_d2),
            'phase_place': (*place_d1, *place_d2),
            'phase_release': (*release_d1, *release_d2),
        }
        if abs(float(self.retreat_z_after_release)) > 0.0:
            phases['phase_retreat'] = (*retreat_d1, *retreat_d2)
        self.log_phase_targets(phases, axes['y'], place_center)
        return phases

    def log_phase_targets(self, phases, lateral_axis, place_center):
        self.get_logger().info('=== Target place calcolati ===')
        self.get_logger().info(f'target frame = {self.base_frame}')
        self.get_logger().info(
            f'place center TCP: x={place_center[0]:.6f}, '
            f'y={place_center[1]:.6f}, z={place_center[2]:.6f}'
        )
        self.get_logger().info(
            f'target local Y axis = ({lateral_axis[0]:.6f}, {lateral_axis[1]:.6f}, {lateral_axis[2]:.6f})'
        )
        for phase_name, values in phases.items():
            x1, y1, z1, x2, y2, z2 = values
            self.get_logger().info(
                f'{phase_name}: d1=({x1:.6f}, {y1:.6f}, {z1:.6f}), '
                f'd2=({x2:.6f}, {y2:.6f}, {z2:.6f})'
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
            raise RuntimeError(f'{phase_name}: goal dagana_1_tcp rejected.')
        if gh2 is None or not gh2.accepted:
            raise RuntimeError(f'{phase_name}: goal dagana_2_tcp rejected.')

        fut_res1 = gh1.get_result_async()
        fut_res2 = gh2.get_result_async()
        rclpy.spin_until_future_complete(self, fut_res1)
        rclpy.spin_until_future_complete(self, fut_res2)

        if fut_res1.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_1_tcp.')
        if fut_res2.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_2_tcp.')
        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase(self, phase_name, phase_targets, time_s):
        x1, y1, z1, x2, y2, z2 = phase_targets
        d1_q, d2_q = self.current_goal_orientations()
        goal1 = self.make_goal(x1, y1, z1, *d1_q, time_s=time_s)
        goal2 = self.make_goal(x2, y2, z2, *d2_q, time_s=time_s)
        self.send_two_goals_and_wait(phase_name, goal1, goal2)

    def execute(self):
        task_weights_configured = False
        try:
            self.get_logger().info('Lettura pose da file...')
            self.read_input_poses()

            phases = self.compute_all_phase_targets()
            if phases is None:
                raise RuntimeError('Impossibile calcolare i target assoluti di place.')

            self.get_logger().info('Configuro pesi task arm...')
            self.configure_task_weights()
            task_weights_configured = True

            self.run_phase(
                'PLACE_PRE_PLACE',
                phases['phase_pre_place'],
                self.time_phase_pre_place,
            )
            self.run_phase('PLACE_DOWN', phases['phase_place'], self.time_phase_place)
            self.run_phase(
                'PLACE_RELEASE_LOCAL_Y',
                phases['phase_release'],
                self.time_phase_release,
            )
            if 'phase_retreat' in phases:
                self.run_phase(
                    'PLACE_RETREAT_AFTER_RELEASE',
                    phases['phase_retreat'],
                    self.time_phase_retreat,
                )
            self.get_logger().info('Place completed.')
        finally:
            if task_weights_configured and self.restore_orientation_weight_at_end:
                try:
                    self.get_logger().info('Ripristino pesi 6D completi dei task arm...')
                    self.restore_full_pose_task_weights()
                    self.get_logger().info('Pesi 6D arm ripristinati.')
                except Exception as exc:
                    self.get_logger().warn(f'Ripristino pesi 6D arm fallito: {exc}')


def main():
    rclpy.init()
    node = CentauroDirectPlace()
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
