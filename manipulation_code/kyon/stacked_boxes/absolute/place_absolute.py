#!/usr/bin/env python3

'''
python3 place_absolute.py --ros-args \
  -p box_name:=box_red_001 \
  -p place_world_x:=3.72 \
  -p place_world_y:=-0.38 \
  -p place_world_z:=0.67
'''

# Absolute dual-arm place of a box.
#
# Conventions used here:
# - box pose is read from Gazebo in WORLD frame
# - robot base (kyon) pose is read from Gazebo in WORLD frame
# - Dagana absolute commands must be sent as:
#     x, y -> relative to Kyon base frame
#     z    -> absolute in WORLD frame
#
# Practical assumption used in this file:
# - box pose read from Gazebo must be converted from WORLD to robot frame on x,y
# - dagana_1_claw and dagana_2_claw positions read from Gazebo are already
#   compatible with the Dagana command convention on x,y, so they are used directly
# - z is always treated as WORLD
#
# Parameters configurable from terminal:
# - box_name
# - place_world_x
# - place_world_y
# - place_world_z

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualDaganaPlaceAbsolute(Node):

    def __init__(self):
        super().__init__('dual_dagana_place_absolute')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_base/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # =====================================================
        # GAZEBO / ENTITY NAMES
        # =====================================================
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        self.declare_parameter('box_name', 'box_red_001')
        self.declare_parameter('robot_name', 'kyon')
        self.declare_parameter('dagana_1_name', 'dagana_1_claw')
        self.declare_parameter('dagana_2_name', 'dagana_2_claw')

        # =====================================================
        # TARGET PLACE POSITION IN WORLD
        # =====================================================
        self.declare_parameter('place_world_x', 3.72)
        self.declare_parameter('place_world_y', -0.38)
        self.declare_parameter('place_world_z', 0.67)

        # =====================================================
        # CONSTANT ABSOLUTE ORIENTATION
        # =====================================================
        self.declare_parameter('d1_qx', 0.5)
        self.declare_parameter('d1_qy', 0.5)
        self.declare_parameter('d1_qz', 0.5)
        self.declare_parameter('d1_qw', 0.5)

        self.declare_parameter('d2_qx', 0.5)
        self.declare_parameter('d2_qy', 0.5)
        self.declare_parameter('d2_qz', 0.5)
        self.declare_parameter('d2_qw', 0.5)

        # =====================================================
        # RELEASE
        # =====================================================
        self.declare_parameter('release_distance', 0.08)

        # mantengo i piccoli bias laterali che avevi nel vecchio file
        self.declare_parameter('place_d1_y_bias', -0.03)
        self.declare_parameter('place_d2_y_bias', +0.03)

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

        self.time_phase_xy = self.get_parameter('time_phase_xy').value
        self.time_phase_z = self.get_parameter('time_phase_z').value
        self.time_phase_release = self.get_parameter('time_phase_release').value

        # =====================================================
        # SAVED POSITIONS
        # =====================================================
        self.box_position = None
        self.robot_position = None
        self.dagana_1_position = None
        self.dagana_2_position = None

    # =========================================================
    # READ POSITIONS FROM GAZEBO
    # =========================================================
    def get_entity_position_from_gz(self, entity_name: str, timeout_sec: float = 3.0):
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
            return (x, y, z)
        except ValueError as e:
            self.get_logger().error(
                f'Errore nel parsing dei numeri di "{entity_name}": {e}'
            )
            return None

    def read_all_positions(self):
        self.box_position = self.get_entity_position_from_gz(self.box_name)
        self.robot_position = self.get_entity_position_from_gz(self.robot_name)
        self.dagana_1_position = self.get_entity_position_from_gz(self.dagana_1_name)
        self.dagana_2_position = self.get_entity_position_from_gz(self.dagana_2_name)

        self._log_position(self.box_name, self.box_position)
        self._log_position(self.robot_name, self.robot_position)
        self._log_position(self.dagana_1_name, self.dagana_1_position)
        self._log_position(self.dagana_2_name, self.dagana_2_position)

    def _log_position(self, name, pos):
        if pos is None:
            self.get_logger().warn(f'Posizione di "{name}" non disponibile.')
            return

        x, y, z = pos
        self.get_logger().info(f'{name}: x={x:.6f}, y={y:.6f}, z={z:.6f}')

    # =========================================================
    # FRAME CONVERSIONS
    # =========================================================
    def world_xy_to_robot_xy(self, world_x, world_y):
        """
        Convert x,y from WORLD to Kyon base frame.
        Assumption: Kyon base orientation aligned with WORLD.
        """
        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        robot_x_world, robot_y_world, _ = self.robot_position

        robot_x = world_x - robot_x_world
        robot_y = world_y - robot_y_world

        return (robot_x, robot_y)

    def current_box_mixed_frame(self):
        """
        Box current position:
        - x,y in robot frame
        - z in world
        """
        if self.box_position is None:
            self.get_logger().error('box_position è None')
            return None

        box_x_world, box_y_world, box_z_world = self.box_position
        converted = self.world_xy_to_robot_xy(box_x_world, box_y_world)
        if converted is None:
            return None

        box_x_robot, box_y_robot = converted

        self.get_logger().info(
            f'Box current mixed-frame: x_robot={box_x_robot:.6f}, '
            f'y_robot={box_y_robot:.6f}, z_world={box_z_world:.6f}'
        )

        return (box_x_robot, box_y_robot, box_z_world)

    def target_box_mixed_frame(self):
        """
        Target place position:
        - x,y in robot frame
        - z in world
        """
        converted = self.world_xy_to_robot_xy(
            self.place_world_x,
            self.place_world_y
        )
        if converted is None:
            return None

        target_x_robot, target_y_robot = converted
        target_z_world = self.place_world_z

        self.get_logger().info(
            f'Box target mixed-frame: x_robot={target_x_robot:.6f}, '
            f'y_robot={target_y_robot:.6f}, z_world={target_z_world:.6f}'
        )

        return (target_x_robot, target_y_robot, target_z_world)

    def current_dagana_mixed_frame_targets(self):
        """
        IMPORTANT:
        Dagana positions read from Gazebo are used directly as command-compatible values:
        - x,y already treated as robot/base frame values
        - z treated as world value

        This avoids subtracting the robot pose twice.
        """
        if self.dagana_1_position is None:
            self.get_logger().error('dagana_1_position è None')
            return None

        if self.dagana_2_position is None:
            self.get_logger().error('dagana_2_position è None')
            return None

        d1x, d1y, d1z = self.dagana_1_position
        d2x, d2y, d2z = self.dagana_2_position

        self.get_logger().info(
            f'Dagana current mixed-frame (used directly): '
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
        current_box = self.current_box_mixed_frame()
        target_box = self.target_box_mixed_frame()
        current_daganas = self.current_dagana_mixed_frame_targets()

        if current_box is None or target_box is None or current_daganas is None:
            return None

        current_bx, current_by, current_bz = current_box
        target_bx, target_by, target_bz = target_box

        (d1x, d1y, d1z), (d2x, d2y, d2z) = current_daganas

        delta_x = target_bx - current_bx
        delta_y = target_by - current_by
        delta_z = target_bz - current_bz

        phase_xy_d1_x = d1x + delta_x
        phase_xy_d1_y = d1y + delta_y + self.place_d1_y_bias
        phase_xy_d1_z = d1z

        phase_xy_d2_x = d2x + delta_x
        phase_xy_d2_y = d2y + delta_y + self.place_d2_y_bias
        phase_xy_d2_z = d2z

        phase_z_d1_x = phase_xy_d1_x
        phase_z_d1_y = phase_xy_d1_y
        phase_z_d1_z = d1z + delta_z

        phase_z_d2_x = phase_xy_d2_x
        phase_z_d2_y = phase_xy_d2_y
        phase_z_d2_z = d2z + delta_z

        self.get_logger().info('=== TARGET ASSOLUTI PLACE CALCOLATI ===')
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

    def compute_release_targets(self):
        current_daganas = self.current_dagana_mixed_frame_targets()
        if current_daganas is None:
            return None

        (d1x, d1y, d1z), (d2x, d2y, d2z) = current_daganas

        release_d1_x = d1x
        release_d1_y = d1y + self.release_distance
        release_d1_z = d1z

        release_d2_x = d2x
        release_d2_y = d2y - self.release_distance
        release_d2_z = d2z

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
        goal1 = self.make_goal(
            x1, y1, z1,
            self.d1_qx, self.d1_qy, self.d1_qz, self.d1_qw,
            time_s=time_s
        )
        goal2 = self.make_goal(
            x2, y2, z2,
            self.d2_qx, self.d2_qy, self.d2_qz, self.d2_qw,
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

        self.get_logger().info('Rilettura posizioni prima del release...')
        self.read_all_positions()

        release_targets = self.compute_release_targets()
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
    node = DualDaganaPlaceAbsolute()
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