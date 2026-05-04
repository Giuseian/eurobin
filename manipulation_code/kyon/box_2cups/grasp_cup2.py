#!/usr/bin/env python3

# Single-arm grasp pre-alignment for a small cup (glass_001)
# Uses only dagana_1
#
# Sequence:
# - read positions from Gazebo
# - pre-rotation of dagana_1
# - PHASE 1: align x and y with the glass + rotate +90 deg around x
# - PHASE 2: align z with the glass

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class SingleDaganaGlassReach(Node):

    def __init__(self):
        super().__init__('single_dagana_glass_reach')

        self.client = ActionClient(self, ReachPose, '/dagana_1_base/reach')

        # Topic Gazebo
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        # Entità
        self.glass_name = 'glass_002'
        self.robot_name = 'kyon'
        self.dagana_name = 'dagana_1_claw'

        # Timing
        self.time_init_rot = 3.0
        self.time_phase1_xy_rot = 20.0
        self.time_phase2_z = 20.0

        # Posizioni salvate
        self.glass_position = None   # WORLD
        self.robot_position = None   # WORLD
        self.dagana_position = None  # BODY

    # =========================================================
    # LETTURA POSIZIONI DA GAZEBO
    # =========================================================
    def get_entity_position_from_gz(self, entity_name: str, timeout_sec: float = 3.0):
        """
        Legge un messaggio da:
            gz topic -e -n 1 -t /world/default/dynamic_pose/info
        e cerca il blocco relativo a entity_name.

        Restituisce una tupla (x, y, z) oppure None.
        """
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
        self.glass_position = self.get_entity_position_from_gz(self.glass_name)
        self.robot_position = self.get_entity_position_from_gz(self.robot_name)
        self.dagana_position = self.get_entity_position_from_gz(self.dagana_name)

        self._log_position(self.glass_name, self.glass_position)
        self._log_position(self.robot_name, self.robot_position)
        self._log_position(self.dagana_name, self.dagana_position)

    def _log_position(self, name, pos):
        if pos is None:
            self.get_logger().warn(f'Posizione di "{name}" non disponibile.')
            return

        x, y, z = pos
        self.get_logger().info(f'{name}: x={x:.6f}, y={y:.6f}, z={z:.6f}')

    # =========================================================
    # TRASFORMAZIONE WORLD -> ROBOT/BODY
    # =========================================================
    def glass_position_in_robot_frame(self):
        """
        Ipotesi:
        - glass_position è nel frame WORLD
        - robot_position è nel frame WORLD
        - orientamento robot = 0

        Allora:
            p_glass_robot = p_glass_world - p_robot_world
        """
        if self.glass_position is None:
            self.get_logger().error('glass_position è None')
            return None

        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        glass_x_world, glass_y_world, glass_z_world = self.glass_position
        robot_x_world, robot_y_world, robot_z_world = self.robot_position

        glass_x_robot = glass_x_world - robot_x_world
        glass_y_robot = glass_y_world - robot_y_world
        glass_z_robot = glass_z_world - robot_z_world

        self.get_logger().info(
            f'Glass nel frame robot: x={glass_x_robot:.6f}, '
            f'y={glass_y_robot:.6f}, z={glass_z_robot:.6f}'
        )

        return (glass_x_robot, glass_y_robot, glass_z_robot)

    # =========================================================
    # CALCOLO FASI
    # =========================================================
    def compute_phase_offsets(self):
        """
        Calcola:
        - PHASE 1: allineamento x e y del dagana_1 rispetto al glass
                   + rotazione di +90 gradi rispetto a x
        - PHASE 2: allineamento z del dagana_1 rispetto al glass

        Restituisce:
            {
                'phase1_xy_rot': (dx, dy, dz),
                'phase2_z':      (dx, dy, dz)
            }
        """
        if self.dagana_position is None:
            self.get_logger().error('dagana_position è None')
            return None

        glass_robot = self.glass_position_in_robot_frame()
        if glass_robot is None:
            return None

        glass_x_robot, glass_y_robot, glass_z_robot = glass_robot
        dag_x, dag_y, dag_z = self.dagana_position

        target_x = glass_x_robot
        target_y = glass_y_robot
        target_z = glass_z_robot

        # PHASE 1: align Y and Z
        phase1_dx = (target_x - dag_x)/2
        phase1_dy = target_y - dag_y + 0.1
        phase1_dz = target_z - dag_z + 0.025

        # PHASE 2: align X
        phase2_dx = (target_x - dag_x)/2 - 0.08
        phase2_dy = 0.0
        phase2_dz = 0.0

        self.get_logger().info('=== OFFSETS CALCOLATI ===')
        self.get_logger().info(
            f'PHASE 1 XY+ROT: dx={phase1_dx:.6f}, dy={phase1_dy:.6f}, dz={phase1_dz:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 Z:      dx={phase2_dx:.6f}, dy={phase2_dy:.6f}, dz={phase2_dz:.6f}'
        )

        return {
            'phase1_xy_rot': (phase1_dx, phase1_dy, phase1_dz),
            'phase2_z': (phase2_dx, phase2_dy, phase2_dz)
        }

    # =========================================================
    # MANIPOLAZIONE
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

    def _send_goal_and_wait(self, phase_name: str, goal: ReachPose.Goal) -> None:
        self.get_logger().info(f'=== Starting {phase_name} ===')

        self.client.wait_for_server()

        fut_send = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut_send)

        gh = fut_send.result()

        if gh is None or not gh.accepted:
            raise RuntimeError(f'{phase_name}: goal dagana_1 rejected.')

        fut_res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, fut_res)

        if fut_res.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_1.')

        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase(self, phase_name: str, dx, dy, dz,
                  qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                  time_s=8.0) -> None:
        goal = self._make_goal(dx, dy, dz, qx=qx, qy=qy, qz=qz, qw=qw, time_s=time_s)
        self._send_goal_and_wait(phase_name, goal)

    def execute(self):
        # =====================================================
        # LETTURA INIZIALE
        # =====================================================
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        # =====================================================
        # PRE-ROTATION
        # Solita prerotazione del grasp box per dagana_1
        # =====================================================
        init_goal = self._make_goal(
            0.0, 0.0, 0.0,
            qx=0.0, qy=0.0, qz=0.173, qw=0.984,
            time_s=self.time_init_rot,
            incremental=True
        )
        self._send_goal_and_wait("INIT_ROT", init_goal)

        # =====================================================
        # RILETTURA DOPO INIT_ROT
        # =====================================================
        self.get_logger().info(
            'Rilettura posizioni dopo INIT_ROT per calcolare le fasi...'
        )
        self.read_all_positions()

        phases = self.compute_phase_offsets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare gli offset delle fasi.')

        # =====================================================
        # PHASE 1: XY + ROT_X_90
        # =====================================================
        self.run_phase(
            "PHASE 1 - ALIGN XY + ROT_X_90",
            *phases['phase1_xy_rot'],
            #qx=0.70710678, qy=0.0, qz=0.0, qw=0.70710678,      # 90 gradi x
            #qx=0.76604444, qy=0.0, qz=0.0, qw=0.64278761,       # 100 gradi x
            qx=0.0, qy=0.0, qz=0.76604444, qw=0.64278761,       # 90 gradi z 
            time_s=self.time_phase1_xy_rot
        )

        # =====================================================
        # PHASE 2: Z
        # =====================================================
        self.run_phase(
            "PHASE 2 - ALIGN Z",
            *phases['phase2_z'],
            time_s=self.time_phase2_z
        )

        self.get_logger().info("Pre-grasp alignment completed.")

    def print_saved_positions(self):
        self._print_one('glass_position', self.glass_position)
        self._print_one('robot_position', self.robot_position)
        self._print_one('dagana_position', self.dagana_position)

    def _print_one(self, label, pos):
        if pos is None:
            self.get_logger().info(f'{label} = None')
            return

        x, y, z = pos
        self.get_logger().info(f'{label} = ({x:.6f}, {y:.6f}, {z:.6f})')


def main():
    rclpy.init()
    node = SingleDaganaGlassReach()
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