#!/usr/bin/env python3

# Code for grasping a box with both hands without the claw and laterally.
# Parameters:
# - grasp
# - dual-arm symmetric
# - without claw
# - laterally
#
# Modified behavior:
# - PHASE 1A: approach on x and y
# - PHASE 1B: approach on z
# - PHASE 2: close on y until box contact
# - PHASE 3: extra squeeze on y + lift on z

import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualDaganaReach(Node):

    def __init__(self):
        super().__init__('dual_dagana_reach')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_base/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # Topic Gazebo
        self.gz_pose_topic = '/world/default/dynamic_pose/info'

        # Entità da leggere
        self.box_name = 'box_green_001'
        self.robot_name = 'kyon'
        self.dagana_1_name = 'dagana_1_claw'
        self.dagana_2_name = 'dagana_2_claw'

        # Geometria / tuning presa
        self.box_width = 0.38
        self.phase1_clearance = 0.05
        self.phase3_extra_squeeze = 0.05
        self.lift_z_phase3 = 0.2

        # Durate
        self.time_init_rot = 3.0
        self.time_phase1_xy = 15.0
        self.time_phase1_z = 15.0
        self.time_phase2 = 15.0
        self.time_phase3 = 15.0

        # Posizioni salvate
        self.box_position = None
        self.robot_position = None
        self.dagana_1_position = None
        self.dagana_2_position = None

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
        """
        Legge e salva tutte le posizioni che ci interessano.
        """
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
    # TRASFORMAZIONE WORLD -> ROBOT
    # =========================================================
    def box_position_in_robot_frame(self):
        """
        Ipotesi:
        - box_position è nel frame WORLD
        - robot_position è nel frame WORLD
        - orientamento robot = 0

        Allora:
            p_box_robot = p_box_world - p_robot_world
        """
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

    # =========================================================
    # CALCOLO OFFSETS AUTOMATICI DELLE FASI
    # =========================================================
    def compute_all_phase_offsets(self):
        """
        Calcola:
        - PHASE 1A: approccio su x e y, z fermo
        - PHASE 1B: approccio su z, x e y fermi
        - PHASE 2: chiusura su y fino ai lati della scatola
        - PHASE 3: ulteriore chiusura su y + lift su z

        Restituisce un dict con tutti gli offset incrementali.
        """
        if self.dagana_1_position is None:
            self.get_logger().error('dagana_1_position è None')
            return None

        if self.dagana_2_position is None:
            self.get_logger().error('dagana_2_position è None')
            return None

        box_robot = self.box_position_in_robot_frame()
        if box_robot is None:
            return None

        box_x_robot, box_y_robot, box_z_robot = box_robot
        dag1_x, dag1_y, dag1_z = self.dagana_1_position
        dag2_x, dag2_y, dag2_z = self.dagana_2_position

        half_box = self.box_width / 2.0

        # -----------------------------------------------------
        # Target finale dell'approccio (vecchia PHASE 1)
        # -----------------------------------------------------
        target1_x_p1 = box_x_robot
        target2_x_p1 = box_x_robot

        target1_z_p1 = box_z_robot
        target2_z_p1 = box_z_robot

        target1_y_p1 = box_y_robot + half_box + self.phase1_clearance
        target2_y_p1 = box_y_robot - half_box - self.phase1_clearance

        # -----------------------------------------------------
        # PHASE 1A: prima muovo solo su X e Y
        # Z resta fermo
        # -----------------------------------------------------
        phase1a_dx1 = target1_x_p1 - dag1_x
        phase1a_dy1 = target1_y_p1 - dag1_y + 0.025
        phase1a_dz1 = 0.0

        phase1a_dx2 = target2_x_p1 - dag2_x
        phase1a_dy2 = target2_y_p1 - dag2_y + 0.025
        phase1a_dz2 = 0.0

        # -----------------------------------------------------
        # PHASE 1B: poi muovo solo su Z
        # X e Y restano fermi
        # -----------------------------------------------------
        phase1b_dx1 = 0.0
        phase1b_dy1 = 0.0
        phase1b_dz1 = target1_z_p1 - dag1_z

        phase1b_dx2 = 0.0
        phase1b_dy2 = 0.0
        phase1b_dz2 = target2_z_p1 - dag2_z

        # -----------------------------------------------------
        # Target PHASE 2
        # Andiamo a contatto con la scatola solo lungo y
        # -----------------------------------------------------
        target1_y_p2 = box_y_robot + half_box
        target2_y_p2 = box_y_robot - half_box

        phase2_dx1 = 0.0
        phase2_dy1 = target1_y_p2 - target1_y_p1
        phase2_dz1 = 0.0

        phase2_dx2 = 0.0
        phase2_dy2 = target2_y_p2 - target2_y_p1
        phase2_dz2 = 0.0

        # -----------------------------------------------------
        # Target PHASE 3
        # Stringiamo ancora un po' e alziamo di 0.2 in z
        # -----------------------------------------------------
        target1_y_p3 = box_y_robot + half_box - self.phase3_extra_squeeze
        target2_y_p3 = box_y_robot - half_box + self.phase3_extra_squeeze

        phase3_dx1 = 0.0
        phase3_dy1 = target1_y_p3 - target1_y_p2
        phase3_dz1 = self.lift_z_phase3

        phase3_dx2 = 0.0
        phase3_dy2 = target2_y_p3 - target2_y_p2
        phase3_dz2 = self.lift_z_phase3

        self.get_logger().info('=== OFFSETS CALCOLATI AUTOMATICAMENTE ===')
        self.get_logger().info(
            f'PHASE 1A dag1: dx={phase1a_dx1:.6f}, dy={phase1a_dy1:.6f}, dz={phase1a_dz1:.6f}'
        )
        self.get_logger().info(
            f'PHASE 1A dag2: dx={phase1a_dx2:.6f}, dy={phase1a_dy2:.6f}, dz={phase1a_dz2:.6f}'
        )
        self.get_logger().info(
            f'PHASE 1B dag1: dx={phase1b_dx1:.6f}, dy={phase1b_dy1:.6f}, dz={phase1b_dz1:.6f}'
        )
        self.get_logger().info(
            f'PHASE 1B dag2: dx={phase1b_dx2:.6f}, dy={phase1b_dy2:.6f}, dz={phase1b_dz2:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 dag1: dx={phase2_dx1:.6f}, dy={phase2_dy1:.6f}, dz={phase2_dz1:.6f}'
        )
        self.get_logger().info(
            f'PHASE 2 dag2: dx={phase2_dx2:.6f}, dy={phase2_dy2:.6f}, dz={phase2_dz2:.6f}'
        )
        self.get_logger().info(
            f'PHASE 3 dag1: dx={phase3_dx1:.6f}, dy={phase3_dy1:.6f}, dz={phase3_dz1:.6f}'
        )
        self.get_logger().info(
            f'PHASE 3 dag2: dx={phase3_dx2:.6f}, dy={phase3_dy2:.6f}, dz={phase3_dz2:.6f}'
        )

        return {
            'phase1a_xy': (
                phase1a_dx1, phase1a_dy1, phase1a_dz1,
                phase1a_dx2, phase1a_dy2, phase1a_dz2
            ),
            'phase1b_z': (
                phase1b_dx1, phase1b_dy1, phase1b_dz1,
                phase1b_dx2, phase1b_dy2, phase1b_dz2
            ),
            'phase2': (
                phase2_dx1, phase2_dy1, phase2_dz1,
                phase2_dx2, phase2_dy2, phase2_dz2
            ),
            'phase3': (
                phase3_dx1, phase3_dy1, phase3_dz1,
                phase3_dx2, phase3_dy2, phase3_dz2
            )
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
        # =====================================================
        # LETTURA INIZIALE
        # =====================================================
        self.get_logger().info('Lettura iniziale posizioni da Gazebo...')
        self.read_all_positions()

        # =====================================================
        # INIT ROT
        # =====================================================
        init_goal_1 = self._make_goal(
            0.0, 0.0, 0.0,
            qx=0.0, qy=0.0, qz=0.173, qw=0.984,
            time_s=self.time_init_rot,
            incremental=True
        )
        init_goal_2 = self._make_goal(
            0.0, 0.0, 0.0,
            qx=0.0, qy=0.0, qz=-0.173, qw=0.984,
            time_s=self.time_init_rot,
            incremental=True
        )
        self._send_two_goals_and_wait("INIT_ROT", init_goal_1, init_goal_2)

        # =====================================================
        # DOPO INIT_ROT
        # =====================================================
        self.get_logger().info(
            'Rilettura posizioni dopo INIT_ROT per calcolare le fasi...'
        )
        self.read_all_positions()

        phases = self.compute_all_phase_offsets()
        if phases is None:
            raise RuntimeError('Impossibile calcolare gli offset automatici delle fasi.')

        # =====================================================
        # PHASE 1A: prima X e Y
        # =====================================================
        self.run_phase("PHASE 1A - XY", *phases['phase1a_xy'], time_s=self.time_phase1_xy)

        # =====================================================
        # PHASE 1B: poi Z
        # =====================================================
        self.run_phase("PHASE 1B - Z", *phases['phase1b_z'], time_s=self.time_phase1_z)

        # =====================================================
        # PHASE 2
        # =====================================================
        self.run_phase("PHASE 2", *phases['phase2'], time_s=self.time_phase2)

        # =====================================================
        # PHASE 3
        # =====================================================
        self.run_phase("PHASE 3", *phases['phase3'], time_s=self.time_phase3)

        self.get_logger().info("All phases completed.")

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
    node = DualDaganaReach()
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