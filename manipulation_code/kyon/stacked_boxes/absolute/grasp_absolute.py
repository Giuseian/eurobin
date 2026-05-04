#!/usr/bin/env python3

'''
python3 grasp_absolute.py --ros-args \
  -p box_name:=box_red_001 \
  -p approach_mode:=yz \
  -p grasp_offset_x:=-0.05 \
  -p grasp_offset_y:=0.0 \
  -p grasp_offset_z:=0.0 \
  -p phase1_split_fraction:=0.5
'''

# Modular absolute dual-arm grasp of a box.
#
# Conventions:
# - box pose is read from Gazebo in WORLD frame
# - robot base (kyon) pose is read from Gazebo in WORLD frame
# - Dagana absolute commands:
#     x, y -> relative to Kyon base frame
#     z    -> absolute in WORLD frame
#
# Parameters configurable from terminal:
# - box_name
# - approach_mode: yz | xz | xy
# - grasp_offset_x, grasp_offset_y, grasp_offset_z
#
# Behavior of approach_mode:
# - yz: phase1 moves Y,Z + half X ; phase2 completes X
# - xz: phase1 moves X,Z + half Y ; phase2 completes Y
# - xy: phase1 moves X,Y + half Z ; phase2 completes Z
#
# Then:
# - PHASE 3: close on Y until box contact
# - PHASE 4: extra squeeze on Y + lift on Z
#
# Orientation is kept constant for the whole task.

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualDaganaReachAbsoluteModular(Node):

    def __init__(self):
        super().__init__('dual_dagana_reach_absolute_modular')

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
        # START ABSOLUTE REFERENCE POSE
        # x,y relative to Kyon base
        # z absolute in world
        # =====================================================
        self.declare_parameter('d1_start_x', 0.35936)
        self.declare_parameter('d1_start_y', 0.23365)
        self.declare_parameter('d1_start_z', 1.1709)

        self.declare_parameter('d2_start_x', 0.35936)
        self.declare_parameter('d2_start_y', -0.23365)
        self.declare_parameter('d2_start_z', 1.1709)

        # =====================================================
        # CONSTANT ABSOLUTE ORIENTATION FOR WHOLE TASK
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
        # GRASP GEOMETRY / TUNING
        # =====================================================
        self.declare_parameter('box_width', 0.38)
        self.declare_parameter('phase1_clearance', 0.05)
        self.declare_parameter('phase1_extra_margin_x', 0.0)
        self.declare_parameter('phase1_extra_margin_y', 0.0)
        self.declare_parameter('phase1_extra_margin_z', 0.0)

        # approach_mode:
        # - yz -> phase1: yz + half x ; phase2: rest x
        # - xz -> phase1: xz + half y ; phase2: rest y
        # - xy -> phase1: xy + half z ; phase2: rest z
        self.declare_parameter('approach_mode', 'yz')
        self.declare_parameter('phase1_split_fraction', 0.5)

        # Final target offsets of the grasp center relative to box center
        self.declare_parameter('grasp_offset_x', 0.0)
        self.declare_parameter('grasp_offset_y', 0.0)
        self.declare_parameter('grasp_offset_z', 0.0)

        # Contact / squeeze / lift
        self.declare_parameter('phase3_extra_squeeze', 0.1)
        self.declare_parameter('lift_z_phase4', 0.20)

        # =====================================================
        # TIMINGS
        # =====================================================
        self.declare_parameter('time_start_pose', 5.0)
        self.declare_parameter('time_phase1', 10.0)
        self.declare_parameter('time_phase2', 10.0)
        self.declare_parameter('time_phase3', 10.0)
        self.declare_parameter('time_phase4', 10.0)

        # =====================================================
        # READ PARAMETERS
        # =====================================================
        self.box_name = self.get_parameter('box_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.dagana_1_name = self.get_parameter('dagana_1_name').value
        self.dagana_2_name = self.get_parameter('dagana_2_name').value

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

        # =====================================================
        # SAVED POSITIONS FROM GAZEBO
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
                'Comando "gz" non trovato. Controlla che Gazebo sia installato '
                'e che l\'ambiente sia caricato.'
            )
            return None
        except subprocess.TimeoutExpired:
            self.get_logger().error(
                f'Timeout durante la lettura del topic {self.gz_pose_topic}'
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
                f'Errore parsing posizione di "{entity_name}": {e}'
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
    # WORLD -> KYON BASE FRAME FOR X,Y ONLY
    # Z REMAINS WORLD
    # =========================================================
    def box_target_components(self):
        if self.box_position is None:
            self.get_logger().error('box_position è None')
            return None

        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        box_x_world, box_y_world, box_z_world = self.box_position
        robot_x_world, robot_y_world, _ = self.robot_position

        box_x_robot = box_x_world - robot_x_world
        box_y_robot = box_y_world - robot_y_world

        # apply user-defined grasp center offsets
        grasp_x_robot = box_x_robot + self.grasp_offset_x
        grasp_y_robot = box_y_robot + self.grasp_offset_y
        grasp_z_world = box_z_world + self.grasp_offset_z

        self.get_logger().info(
            f'Box target mixed-frame -> '
            f'x_robot={box_x_robot:.6f}, y_robot={box_y_robot:.6f}, z_world={box_z_world:.6f}'
        )
        self.get_logger().info(
            f'Grasp target mixed-frame with offsets -> '
            f'x_robot={grasp_x_robot:.6f}, y_robot={grasp_y_robot:.6f}, z_world={grasp_z_world:.6f}'
        )

        return (grasp_x_robot, grasp_y_robot, grasp_z_world)

    # =========================================================
    # PHASE PLANNER
    # =========================================================
    def split_axis_value(self, start_value, target_value):
        return start_value + self.phase1_split_fraction * (target_value - start_value)

    def compute_all_phase_targets(self):
        target = self.box_target_components()
        if target is None:
            return None

        grasp_x_robot, grasp_y_robot, grasp_z_world = target
        half_box = self.box_width / 2.0

        # -----------------------------------------------------
        # START POSE
        # -----------------------------------------------------
        start_d1_x = self.d1_start_x
        start_d1_y = self.d1_start_y
        start_d1_z = self.d1_start_z

        start_d2_x = self.d2_start_x
        start_d2_y = self.d2_start_y
        start_d2_z = self.d2_start_z

        # -----------------------------------------------------
        # PRE-GRASP TARGET BEFORE CONTACT
        # contact still happens symmetrically along Y
        # -----------------------------------------------------
        pre_d1_x = grasp_x_robot
        pre_d2_x = grasp_x_robot

        pre_d1_y = grasp_y_robot + half_box + self.phase1_clearance
        pre_d2_y = grasp_y_robot - half_box - self.phase1_clearance

        pre_d1_z = grasp_z_world
        pre_d2_z = grasp_z_world

        # -----------------------------------------------------
        # PHASE 1 / PHASE 2 ACCORDING TO approach_mode
        # -----------------------------------------------------
        if self.approach_mode == 'yz':
            # phase1: Y,Z + half X ; phase2: rest X
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
            # phase1: X,Z + half Y ; phase2: rest Y
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

        else:  # xy
            # phase1: X,Y + half Z ; phase2: rest Z
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

        # -----------------------------------------------------
        # PHASE 3: close on Y until contact
        # -----------------------------------------------------
        contact_d1_x = phase2_d1_x
        contact_d1_y = grasp_y_robot + half_box
        contact_d1_z = phase2_d1_z

        contact_d2_x = phase2_d2_x
        contact_d2_y = grasp_y_robot - half_box
        contact_d2_z = phase2_d2_z

        # -----------------------------------------------------
        # PHASE 4: extra squeeze + lift
        # -----------------------------------------------------
        lift_d1_x = contact_d1_x
        lift_d1_y = grasp_y_robot + half_box - self.phase3_extra_squeeze
        lift_d1_z = contact_d1_z + self.lift_z_phase4

        lift_d2_x = contact_d2_x
        lift_d2_y = grasp_y_robot - half_box + self.phase3_extra_squeeze
        lift_d2_z = contact_d2_z + self.lift_z_phase4

        self.get_logger().info('=== TARGET ASSOLUTI CALCOLATI ===')
        self.get_logger().info(f'approach_mode = {self.approach_mode}')
        self.get_logger().info(
            f'START dag1:  x={start_d1_x:.6f}, y={start_d1_y:.6f}, z={start_d1_z:.6f}'
        )
        self.get_logger().info(
            f'START dag2:  x={start_d2_x:.6f}, y={start_d2_y:.6f}, z={start_d2_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 1 dag1: x={phase1_d1_x:.6f}, y={phase1_d1_y:.6f}, z={phase1_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 1 dag2: x={phase1_d2_x:.6f}, y={phase1_d2_y:.6f}, z={phase1_d2_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 dag1: x={phase2_d1_x:.6f}, y={phase2_d1_y:.6f}, z={phase2_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 dag2: x={phase2_d2_x:.6f}, y={phase2_d2_y:.6f}, z={phase2_d2_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 3 dag1: x={contact_d1_x:.6f}, y={contact_d1_y:.6f}, z={contact_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 3 dag2: x={contact_d2_x:.6f}, y={contact_d2_y:.6f}, z={contact_d2_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 4 dag1: x={lift_d1_x:.6f}, y={lift_d1_y:.6f}, z={lift_d1_z:.6f}'
        )
        self.get_logger().info(
            f'PHASE 4 dag2: x={lift_d2_x:.6f}, y={lift_d2_y:.6f}, z={lift_d2_z:.6f}'
        )

        return {
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
            )
        }

    # =========================================================
    # ACTION UTILS
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

    def send_two_goals_and_wait(self, phase_name: str, goal1: ReachPose.Goal, goal2: ReachPose.Goal):
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

    def run_phase_absolute(self, phase_name: str, x1, y1, z1, x2, y2, z2, time_s: float):
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

    # =========================================================
    # MAIN EXECUTION
    # =========================================================
    def execute(self):
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        phases = self.compute_all_phase_targets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare i target assoluti delle fasi.')

        self.run_phase_absolute(
            'START_POSE',
            *phases['start_pose'],
            time_s=self.time_start_pose
        )

        self.run_phase_absolute(
            f'PHASE 1 - {self.approach_mode.upper()}',
            *phases['phase1'],
            time_s=self.time_phase1
        )

        self.run_phase_absolute(
            'PHASE 2 - COMPLETE_REMAINING_AXIS',
            *phases['phase2'],
            time_s=self.time_phase2
        )

        self.run_phase_absolute(
            'PHASE 3 - CLOSE_ON_Y',
            *phases['phase3'],
            time_s=self.time_phase3
        )

        self.run_phase_absolute(
            'PHASE 4 - SQUEEZE_AND_LIFT',
            *phases['phase4'],
            time_s=self.time_phase4
        )

        self.get_logger().info('All phases completed.')

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
    node = DualDaganaReachAbsoluteModular()

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