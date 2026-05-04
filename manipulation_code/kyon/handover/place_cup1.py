#!/usr/bin/env python3

# Single-arm place for a small cup / glass
# Uses only dagana_2
#
# Sequence:
# - read positions from Gazebo
# - PHASE 1: lift along z by 0.2
# - PHASE 2: move dagana_2 to a target point (x, y, z) in WORLD

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class SingleDaganaGlassPlace(Node):

    def __init__(self):
        super().__init__('single_dagana_glass_place')

        self.client = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # Topic Gazebo
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        # Entità
        self.robot_name = 'kyon'
        self.dagana_name = 'dagana_2_claw'

        # Target finale nel WORLD
        #self.target_world = (3.7, -0.04, 1.1)
        self.target_world = (3.7, 0.1, 0.62)

        # Parametri place
        self.lift_z = 0.2

        # Timing
        self.time_phase1_lift = 15.0
        self.time_phase2_place = 15.0

        # Posizioni salvate
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
        self.robot_position = self.get_entity_position_from_gz(self.robot_name)
        self.dagana_position = self.get_entity_position_from_gz(self.dagana_name)

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
    def target_in_robot_frame(self):
        """
        Ipotesi:
        - target_world è nel frame WORLD
        - robot_position è nel frame WORLD
        - orientamento robot = 0

        Allora:
            p_target_robot = p_target_world - p_robot_world
        """
        if self.robot_position is None:
            self.get_logger().error('robot_position è None')
            return None

        tx_w, ty_w, tz_w = self.target_world
        rx_w, ry_w, rz_w = self.robot_position

        tx_r = tx_w - rx_w
        ty_r = ty_w - ry_w
        tz_r = tz_w - rz_w

        self.get_logger().info(
            f'Target nel frame robot: x={tx_r:.6f}, y={ty_r:.6f}, z={tz_r:.6f}'
        )

        return (tx_r, ty_r, tz_r)

    # =========================================================
    # CALCOLO FASI
    # =========================================================
    def compute_phase_offsets(self):
        """
        Calcola:
        - PHASE 1: lift su z di +0.2
        - PHASE 2: raggiungimento target xyz nel frame robot,
                   partendo dalla nuova posa dopo il lift

        Restituisce:
            {
                'phase1_lift': (dx, dy, dz),
                'phase2_place': (dx, dy, dz)
            }
        """
        if self.dagana_position is None:
            self.get_logger().error('dagana_position è None')
            return None

        target_robot = self.target_in_robot_frame()
        if target_robot is None:
            return None

        dag_x, dag_y, dag_z = self.dagana_position
        target_x, target_y, target_z = target_robot

        # -----------------------------------------------------
        # PHASE 1: sollevamento verticale
        # -----------------------------------------------------
        phase1_dx = 0.0
        phase1_dy = 0.0
        phase1_dz = self.lift_z

        # -----------------------------------------------------
        # PHASE 2: raggiungimento del target finale
        # calcolato dalla posa dopo il lift
        # -----------------------------------------------------
        dag_z_after_lift = dag_z + self.lift_z

        phase2_dx = target_x - dag_x
        phase2_dy = target_y - dag_y
        phase2_dz = target_z - dag_z_after_lift

        self.get_logger().info('=== OFFSETS PLACE CALCOLATI ===')
        self.get_logger().info(
            f'PHASE 1 LIFT:  dx={phase1_dx:.6f}, dy={phase1_dy:.6f}, dz={phase1_dz:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 PLACE: dx={phase2_dx:.6f}, dy={phase2_dy:.6f}, dz={phase2_dz:.6f}'
        )

        return {
            'phase1_lift': (phase1_dx, phase1_dy, phase1_dz),
            'phase2_place': (phase2_dx, phase2_dy, phase2_dz)
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
            raise RuntimeError(f'{phase_name}: goal dagana_2 rejected.')

        fut_res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, fut_res)

        if fut_res.result() is None:
            raise RuntimeError(f'{phase_name}: no result dagana_2.')

        self.get_logger().info(f'=== Finished {phase_name} ===')

    def run_phase(self, phase_name: str, dx, dy, dz, time_s=8.0) -> None:
        goal = self._make_goal(dx, dy, dz, time_s=time_s)
        self._send_goal_and_wait(phase_name, goal)

    def execute(self):
        # =====================================================
        # LETTURA INIZIALE
        # =====================================================
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        phases = self.compute_phase_offsets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare gli offset del place.')

        # =====================================================
        # PHASE 1: LIFT Z +0.2
        # =====================================================
        self.run_phase(
            "PHASE 1 - LIFT Z",
            *phases['phase1_lift'],
            time_s=self.time_phase1_lift
        )

        # =====================================================
        # PHASE 2: GO TO TARGET XYZ
        # =====================================================
        self.run_phase(
            "PHASE 2 - PLACE TO TARGET",
            *phases['phase2_place'],
            time_s=self.time_phase2_place
        )

        self.get_logger().info("Place completed.")

    def print_saved_positions(self):
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
    node = SingleDaganaGlassPlace()
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