#!/usr/bin/env python3

# Code for placing a box with both hands.
# Phases:
# - PHASE 1: move the box center on x,y in WORLD
# - PHASE 2: move the box center on z in WORLD
# - PHASE 3: move the two end effectors away from the box

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualDaganaPlace(Node):

    def __init__(self):
        super().__init__('dual_dagana_place')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_base/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # Topic Gazebo
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        # Entities
        self.box_name = 'box3_001'
        self.robot_name = 'kyon'
        self.dagana_1_name = 'dagana_1_claw'
        self.dagana_2_name = 'dagana_2_claw'

        # Target finale della box nel world
        self.target_box_world = (3.72, -0.38, 0.67)

        # Apertura finale dopo il place
        self.release_distance = 0.08

        # Saved positions
        self.box_position = None          # WORLD
        self.robot_position = None        # WORLD
        self.dagana_1_position = None     # BODY
        self.dagana_2_position = None     # BODY

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
    # WORLD -> ROBOT/BODY
    # =========================================================
    def box_position_in_robot_frame(self):
        if self.box_position is None:
            self.get_logger().error('box_position è None')
            return None

        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        box_x_world, box_y_world, box_z_world = self.box_position
        robot_x_world, robot_y_world, robot_z_world = self.robot_position

        box_x_robot = box_x_world - robot_x_world
        box_y_robot = box_y_world - robot_y_world
        box_z_robot = box_z_world - robot_z_world

        self.get_logger().info(
            f'Box nel frame robot: x={box_x_robot:.6f}, y={box_y_robot:.6f}, z={box_z_robot:.6f}'
        )

        return (box_x_robot, box_y_robot, box_z_robot)

    def target_box_in_robot_frame(self):
        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        tx_w, ty_w, tz_w = self.target_box_world
        rx_w, ry_w, rz_w = self.robot_position

        tx_r = tx_w - rx_w
        ty_r = ty_w - ry_w
        tz_r = tz_w - rz_w

        self.get_logger().info(
            f'Target box nel frame robot: x={tx_r:.6f}, y={ty_r:.6f}, z={tz_r:.6f}'
        )

        return (tx_r, ty_r, tz_r)

    # =========================================================
    # PLACE
    # =========================================================
    def compute_place_phase_offsets(self):
        """
        Calcola due fasi di place:
        - phase_xy: spostamento solo lungo x e y
        - phase_z:  spostamento solo lungo z
        """
        if self.box_position is None:
            self.get_logger().error('box_position è None')
            return None

        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        if self.dagana_1_position is None:
            self.get_logger().error('dagana_1_position è None')
            return None

        if self.dagana_2_position is None:
            self.get_logger().error('dagana_2_position è None')
            return None

        current_box_robot = self.box_position_in_robot_frame()
        target_box_robot = self.target_box_in_robot_frame()

        if current_box_robot is None or target_box_robot is None:
            return None

        current_bx, current_by, current_bz = current_box_robot
        target_bx, target_by, target_bz = target_box_robot

        delta_x = target_bx - current_bx
        delta_y = target_by - current_by
        delta_z = target_bz - current_bz

        phase_xy = (
            delta_x, delta_y-0.03, 0.0,
            delta_x, delta_y+0.03, 0.0
        )

        phase_z = (
            0.0, 0.0, delta_z,
            0.0, 0.0, delta_z
        )

        self.get_logger().info('=== OFFSETS PLACE CALCOLATI ===')
        self.get_logger().info(
            f'PHASE_XY dagana_1: dx={phase_xy[0]:.6f}, dy={phase_xy[1]:.6f}, dz={phase_xy[2]:.6f}'
        )
        self.get_logger().info(
            f'PHASE_XY dagana_2: dx={phase_xy[3]:.6f}, dy={phase_xy[4]:.6f}, dz={phase_xy[5]:.6f}'
        )
        self.get_logger().info(
            f'PHASE_Z dagana_1: dx={phase_z[0]:.6f}, dy={phase_z[1]:.6f}, dz={phase_z[2]:.6f}'
        )
        self.get_logger().info(
            f'PHASE_Z dagana_2: dx={phase_z[3]:.6f}, dy={phase_z[4]:.6f}, dz={phase_z[5]:.6f}'
        )

        return {
            'phase_xy': phase_xy,
            'phase_z': phase_z
        }

    def compute_release_offsets(self):
        if self.dagana_1_position is None:
            self.get_logger().error('dagana_1_position è None')
            return None

        if self.dagana_2_position is None:
            self.get_logger().error('dagana_2_position è None')
            return None

        release_dx1 = 0.0
        release_dy1 = +self.release_distance
        release_dz1 = 0.0

        release_dx2 = 0.0
        release_dy2 = -self.release_distance
        release_dz2 = 0.0

        self.get_logger().info('=== OFFSETS RELEASE CALCOLATI ===')
        self.get_logger().info(
            f'dagana_1: dx={release_dx1:.6f}, dy={release_dy1:.6f}, dz={release_dz1:.6f}'
        )
        self.get_logger().info(
            f'dagana_2: dx={release_dx2:.6f}, dy={release_dy2:.6f}, dz={release_dz2:.6f}'
        )

        return (
            release_dx1, release_dy1, release_dz1,
            release_dx2, release_dy2, release_dz2
        )

    # =========================================================
    # MANIPULATION
    # =========================================================
    def _make_goal(self, dx, dy, dz,
                   qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                   time_s=15.0,
                   incremental=True) -> ReachPose.Goal:
        goal = ReachPose.Goal()

        pose = Pose()
        pose.position.x = float(dx)
        pose.position.y = float(dy)
        pose.position.z = float(dz)

        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)

        goal.frames = [pose]
        goal.time = [float(time_s)]
        goal.incremental = bool(incremental)

        return goal

    def _send_two_goals_and_wait(self, phase_name: str, goal1: ReachPose.Goal, goal2: ReachPose.Goal) -> None:
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

    def run_phase(self, phase_name: str, dx1, dy1, dz1, dx2, dy2, dz2, time_s=15.0) -> None:
        goal1 = self._make_goal(dx1, dy1, dz1, time_s=time_s)
        goal2 = self._make_goal(dx2, dy2, dz2, time_s=time_s)
        self._send_two_goals_and_wait(phase_name, goal1, goal2)

    def execute(self):
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        place_phases = self.compute_place_phase_offsets()
        if place_phases is None:
            raise RuntimeError('Impossibile calcolare gli offset di place.')

        self.run_phase("PLACE_MOVE_XY", *place_phases['phase_xy'], time_s=15.0)
        self.run_phase("PLACE_MOVE_Z", *place_phases['phase_z'], time_s=15.0)

        release_offsets = self.compute_release_offsets()
        if release_offsets is None:
            raise RuntimeError('Impossibile calcolare gli offset di release.')

        self.run_phase("PLACE_RELEASE", *release_offsets, time_s=15.0)

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
    node = DualDaganaPlace()
    try:
        node.execute()
        node.print_saved_positions()
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()