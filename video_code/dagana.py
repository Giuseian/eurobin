#!/usr/bin/env python3

# python3 dagana.py set 0.3 0.3

import sys
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from xbot_msgs.msg import JointCommand


class DaganaClampCommander(Node):
    def __init__(self):
        super().__init__('dagana_clamp_commander')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub = self.create_publisher(JointCommand, '/xbotcore/command', qos)

        self.joint_names = [
            'dagana_1_clamp_joint',
            'dagana_2_clamp_joint'
        ]

    def make_msg(self, positions: List[float], ctrl_mode: int = 1) -> JointCommand:
        """
        Crea un JointCommand minimale per i due clamp joint.
        """
        if len(positions) != 2:
            raise ValueError('positions deve contenere esattamente 2 valori')

        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''

        msg.name = list(self.joint_names)
        msg.position = [float(positions[0]), float(positions[1])]

        # Campi opzionali: li mettiamo coerenti con la stessa lunghezza di `name`
        msg.velocity = [0.0, 0.0]
        msg.effort = [0.0, 0.0]
        msg.stiffness = [0.0, 0.0]
        msg.damping = [0.0, 0.0]
        msg.ctrl_mode = [int(ctrl_mode), int(ctrl_mode)]

        msg.aux_name = ''
        msg.aux = []

        return msg

    def publish_for_duration(self, positions: List[float], duration_sec: float = 2.0, rate_hz: float = 20.0):
        """
        Pubblica il comando per un certo tempo, così il controller lo riceve con continuità.
        """
        msg = self.make_msg(positions)
        period = 1.0 / rate_hz
        n_steps = max(1, int(duration_sec * rate_hz))

        self.get_logger().info(
            f'Invio comando clamp: {self.joint_names[0]}={positions[0]:.4f}, '
            f'{self.joint_names[1]}={positions[1]:.4f}, '
            f'duration={duration_sec:.2f}s, rate={rate_hz:.1f}Hz'
        )

        for _ in range(n_steps):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=period))

        self.get_logger().info('Comando completato.')


def print_usage():
    print(
        '\nUso:\n'
        '  python3 dagana_clamp_command.py open\n'
        '  python3 dagana_clamp_command.py close\n'
        '  python3 dagana_clamp_command.py set <pos_joint1> <pos_joint2>\n'
        '\nEsempi:\n'
        '  python3 dagana_clamp_command.py open\n'
        '  python3 dagana_clamp_command.py close\n'
        '  python3 dagana_clamp_command.py set 0.2 0.2\n'
        '  python3 dagana_clamp_command.py set 0.5 0.4\n'
    )


def main():
    # -----------------------------------------------------------------
    # Valori di default da tarare sul tuo robot
    # -----------------------------------------------------------------
    OPEN_POS = 0.0
    CLOSE_POS = 0.6

    rclpy.init()
    node = DaganaClampCommander()

    try:
        if len(sys.argv) < 2:
            print_usage()
            raise RuntimeError('Argomento mancante')

        cmd = sys.argv[1].strip().lower()

        if cmd == 'open':
            positions = [OPEN_POS, OPEN_POS]

        elif cmd == 'close':
            positions = [CLOSE_POS, CLOSE_POS]

        elif cmd == 'set':
            if len(sys.argv) != 4:
                print_usage()
                raise RuntimeError('Per "set" devi specificare 2 valori')
            positions = [float(sys.argv[2]), float(sys.argv[3])]

        else:
            print_usage()
            raise RuntimeError(f'Comando non riconosciuto: {cmd}')

        node.publish_for_duration(positions, duration_sec=2.0, rate_hz=20.0)

    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()