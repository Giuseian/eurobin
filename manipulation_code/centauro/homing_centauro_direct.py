#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from xbot_msgs.msg import JointCommand
try:
    from xbot_msgs.msg import JointState as XbotJointState
except ImportError:
    XbotJointState = None


class DualArmHoming(Node):

    def __init__(self):
        super().__init__('dual_arm_homing')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_tcp/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_tcp/reach')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.clamp_pub = self.create_publisher(JointCommand, '/xbotcore/command', qos)
        self.clamp_joint_names = ['dagana_1_claw_joint', 'dagana_2_claw_joint']
        self.current_joint_positions = {}
        self.logged_joint_state_format = False

        # # ===== POSIZIONI ASSOLUTE HOMING =====
        # self.declare_parameter('d1_x', 0.50693)
        # self.declare_parameter('d1_y', 0.1753)
        # self.declare_parameter('d1_z', 0.25352)

        # self.declare_parameter('d2_x', 0.52485)
        # self.declare_parameter('d2_y', -0.22406)
        # self.declare_parameter('d2_z', 0.27441)

        # # ===== ORIENTAZIONI HOMING =====
        # self.declare_parameter('d1_qx', 0.24991)
        # self.declare_parameter('d1_qy', 0.50097)
        # self.declare_parameter('d1_qz', 0.82353)
        # self.declare_parameter('d1_qw', -0.091496)

        # self.declare_parameter('d2_qx', 0.091491)
        # self.declare_parameter('d2_qy', 0.82353)
        # self.declare_parameter('d2_qz', 0.50097)
        # self.declare_parameter('d2_qw', -0.24991)

        # # ===== POSIZIONI ASSOLUTE HOMING =====
        # self.declare_parameter('d1_x', 0.5)
        # self.declare_parameter('d1_y', 0.3)
        # self.declare_parameter('d1_z', 0.2)

        # self.declare_parameter('d2_x', 0.5)
        # self.declare_parameter('d2_y', -0.3)
        # self.declare_parameter('d2_z', 0.2)

        # # ===== ORIENTAZIONI HOMING =====
        # self.declare_parameter('d1_qx', 0.0)
        # self.declare_parameter('d1_qy', 0.0)
        # self.declare_parameter('d1_qz', -0.7)
        # self.declare_parameter('d1_qw', 0.7)

        # self.declare_parameter('d2_qx', 0.7)
        # self.declare_parameter('d2_qy', 0.7)
        # self.declare_parameter('d2_qz', 0.0)
        # self.declare_parameter('d2_qw', 0.0)

        # # ===== POSIZIONI ASSOLUTE HOMING =====
        # self.declare_parameter('d1_x', 0.53)
        # self.declare_parameter('d1_y', 0.3)
        # self.declare_parameter('d1_z', 0.29)

        # self.declare_parameter('d2_x', 0.53)
        # self.declare_parameter('d2_y', -0.3)
        # self.declare_parameter('d2_z', 0.29)

        # # ===== ORIENTAZIONI HOMING =====
        # self.declare_parameter('d1_qx', 0.5)
        # self.declare_parameter('d1_qy', 0.5)
        # self.declare_parameter('d1_qz', 0.5)
        # self.declare_parameter('d1_qw', -0.5)

        # self.declare_parameter('d2_qx', -0.5)
        # self.declare_parameter('d2_qy', -0.5)
        # self.declare_parameter('d2_qz', -0.5)
        # self.declare_parameter('d2_qw', 0.5)

        # # ===== POSIZIONI ASSOLUTE HOMING TCP =====
        # self.declare_parameter('d1_x', 0.8)
        # self.declare_parameter('d1_y', 0.3)
        # self.declare_parameter('d1_z', 0.25)

        # self.declare_parameter('d2_x', 0.8)
        # self.declare_parameter('d2_y', -0.35)
        # self.declare_parameter('d2_z', 0.25)

        # # ===== ORIENTAZIONI HOMING TCP =====
        # self.declare_parameter('d1_qx', 0.0)
        # self.declare_parameter('d1_qy', 0.7)
        # self.declare_parameter('d1_qz', 0.0)
        # self.declare_parameter('d1_qw', 0.7)

        # self.declare_parameter('d2_qx', 0.0)
        # self.declare_parameter('d2_qy', 0.7)
        # self.declare_parameter('d2_qz', 0.0)
        # self.declare_parameter('d2_qw', 0.7)

        # ===== POSIZIONI ASSOLUTE HOMING TCP DEFAULT =====
        self.declare_parameter('d1_x', 0.54711)
        self.declare_parameter('d1_y', 0.11202)
        self.declare_parameter('d1_z', 0.33458)

        self.declare_parameter('d2_x', 0.57271)
        self.declare_parameter('d2_y', -0.18169)
        self.declare_parameter('d2_z', 0.36443)

        # ===== ORIENTAZIONI HOMING TCP DEFAULT =====
        self.declare_parameter('d1_qx', 0.22808)
        self.declare_parameter('d1_qy', 0.24141)
        self.declare_parameter('d1_qz', -0.11202)
        self.declare_parameter('d1_qw', 0.93656)

        self.declare_parameter('d2_qx', -0.22808)
        self.declare_parameter('d2_qy', 0.24141)
        self.declare_parameter('d2_qz', 0.11202)
        self.declare_parameter('d2_qw', 0.93656)

        self.declare_parameter('motion_time', 5.0)
        self.declare_parameter('close_clamps_after_homing', True)
        self.declare_parameter('clamp_close_pos', 0.0)
        self.declare_parameter('joint_state_topic', '/xbotcore/joint_states')
        self.declare_parameter('joint_state_timeout', 2.0)
        self.declare_parameter('fallback_to_initial_clamp_pos', False)
        self.declare_parameter('clamp_initial_pos_d1', 0.0)
        self.declare_parameter('clamp_initial_pos_d2', 0.0)
        self.declare_parameter('clamp_command_duration', 2.0)
        self.declare_parameter('clamp_command_rate', 100.0)
        self.declare_parameter('clamp_max_speed', 2.0)
        self.declare_parameter('clamp_max_step', 0.05)
        self.declare_parameter('clamp_ctrl_mode', 1)
        self.declare_parameter('joint_state_type', 'auto')

        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value)
        self.joint_state_type = str(self.get_parameter('joint_state_type').value).lower()
        self.joint_state_timeout = self.get_parameter('joint_state_timeout').value
        self.fallback_to_initial_clamp_pos = self.get_parameter('fallback_to_initial_clamp_pos').value
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        joint_state_msg_type = self.resolve_joint_state_msg_type()
        self.joint_state_sub = self.create_subscription(
            joint_state_msg_type,
            self.joint_state_topic,
            self.joint_state_callback,
            joint_state_qos,
        )
        self.get_logger().info(
            f'Sottoscritto {self.joint_state_topic} come '
            f'{joint_state_msg_type.__module__}.{joint_state_msg_type.__name__}'
        )

        self.d1_x = self.get_parameter('d1_x').value
        self.d1_y = self.get_parameter('d1_y').value
        self.d1_z = self.get_parameter('d1_z').value

        self.d2_x = self.get_parameter('d2_x').value
        self.d2_y = self.get_parameter('d2_y').value
        self.d2_z = self.get_parameter('d2_z').value

        self.d1_qx = self.get_parameter('d1_qx').value
        self.d1_qy = self.get_parameter('d1_qy').value
        self.d1_qz = self.get_parameter('d1_qz').value
        self.d1_qw = self.get_parameter('d1_qw').value

        self.d2_qx = self.get_parameter('d2_qx').value
        self.d2_qy = self.get_parameter('d2_qy').value
        self.d2_qz = self.get_parameter('d2_qz').value
        self.d2_qw = self.get_parameter('d2_qw').value

        self.motion_time = self.get_parameter('motion_time').value
        self.close_clamps_after_homing = self.get_parameter('close_clamps_after_homing').value
        self.clamp_close_pos = self.get_parameter('clamp_close_pos').value
        self.clamp_initial_pos_d1 = self.get_parameter('clamp_initial_pos_d1').value
        self.clamp_initial_pos_d2 = self.get_parameter('clamp_initial_pos_d2').value
        self.clamp_command_duration = self.get_parameter('clamp_command_duration').value
        self.clamp_command_rate = self.get_parameter('clamp_command_rate').value
        self.clamp_max_speed = self.get_parameter('clamp_max_speed').value
        self.clamp_max_step = self.get_parameter('clamp_max_step').value
        self.clamp_ctrl_mode = int(self.get_parameter('clamp_ctrl_mode').value)

        if float(self.clamp_command_rate) <= 0.0:
            raise ValueError('clamp_command_rate deve essere positivo.')
        if float(self.clamp_max_speed) <= 0.0:
            raise ValueError('clamp_max_speed deve essere positivo.')
        if float(self.clamp_max_step) <= 0.0:
            raise ValueError('clamp_max_step deve essere positivo.')

        self.last_clamp_positions = [
            float(self.clamp_initial_pos_d1),
            float(self.clamp_initial_pos_d2),
        ]

    def resolve_joint_state_msg_type(self):
        if self.joint_state_type in ('sensor_msgs', 'sensor', 'sensor_msgs/msg/jointstate'):
            return JointState

        if self.joint_state_type in ('xbot_msgs', 'xbot', 'xbot_msgs/msg/jointstate'):
            if XbotJointState is None:
                raise RuntimeError(
                    'joint_state_type=xbot richiesto, ma xbot_msgs.msg.JointState '
                    'non e disponibile.'
                )
            return XbotJointState

        if self.joint_state_type != 'auto':
            raise ValueError(
                'joint_state_type deve essere auto, xbot oppure sensor_msgs.'
            )

        if self.joint_state_topic == '/xbotcore/joint_states' and XbotJointState is not None:
            return XbotJointState

        return JointState

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
                'Uso clamp_initial_pos_d1/d2 perche non ho ricevuto tutti i joint state.'
            )
            return [
                float(self.clamp_initial_pos_d1),
                float(self.clamp_initial_pos_d2),
            ]

        positions = [
            self.current_joint_positions[self.clamp_joint_names[0]],
            self.current_joint_positions[self.clamp_joint_names[1]],
        ]
        self.get_logger().info(
            f'Posizione corrente clamp: {self.clamp_joint_names[0]}={positions[0]:.4f}, '
            f'{self.clamp_joint_names[1]}={positions[1]:.4f}'
        )
        return positions

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

    def send_two_goals_and_wait(self, goal1, goal2):
        self.get_logger().info('Waiting for action servers...')
        self.client_1.wait_for_server()
        self.client_2.wait_for_server()

        self.get_logger().info('Sending absolute homing goals...')

        future_1 = self.client_1.send_goal_async(goal1)
        future_2 = self.client_2.send_goal_async(goal2)

        rclpy.spin_until_future_complete(self, future_1)
        rclpy.spin_until_future_complete(self, future_2)

        gh1 = future_1.result()
        gh2 = future_2.result()

        if gh1 is None or not gh1.accepted:
            raise RuntimeError('Goal dagana_1_tcp rejected')
        if gh2 is None or not gh2.accepted:
            raise RuntimeError('Goal dagana_2_tcp rejected')

        r1 = gh1.get_result_async()
        r2 = gh2.get_result_async()

        rclpy.spin_until_future_complete(self, r1)
        rclpy.spin_until_future_complete(self, r2)

        if r1.result() is None:
            raise RuntimeError('No result from dagana_1_tcp')
        if r2.result() is None:
            raise RuntimeError('No result from dagana_2_tcp')

        self.get_logger().info('Homing motion completed successfully.')


    def make_clamp_msg(self, positions):
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

    def publish_clamp_for_duration(self, target_positions, label):
        duration_sec = float(self.clamp_command_duration)
        rate_hz = float(self.clamp_command_rate)
        max_speed = float(self.clamp_max_speed)
        period = 1.0 / rate_hz
        n_steps = max(1, int(duration_sec * rate_hz))
        target_positions = [float(target_positions[0]), float(target_positions[1])]
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
                    current_positions[index] += max_step if delta > 0.0 else -max_step

            msg = self.make_clamp_msg(current_positions)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.clamp_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.get_clock().sleep_for(Duration(seconds=period))
        self.last_clamp_positions = current_positions

    def close_both_clamps(self):
        self.last_clamp_positions = self.read_current_clamp_positions()
        close_pos = float(self.clamp_close_pos)
        self.publish_clamp_for_duration(
            [close_pos, close_pos],
            'Chiudo entrambi gli end effector',
        )

    def execute(self):
        self.get_logger().info(
            f"dagana_1_tcp pos: ({self.d1_x}, {self.d1_y}, {self.d1_z}) | "
            f"quat: ({self.d1_qx}, {self.d1_qy}, {self.d1_qz}, {self.d1_qw})"
        )
        self.get_logger().info(
            f"dagana_2_tcp pos: ({self.d2_x}, {self.d2_y}, {self.d2_z}) | "
            f"quat: ({self.d2_qx}, {self.d2_qy}, {self.d2_qz}, {self.d2_qw})"
        )

        goal1 = self.make_goal(
            self.d1_x, self.d1_y, self.d1_z,
            self.d1_qx, self.d1_qy, self.d1_qz, self.d1_qw,
            self.motion_time
        )

        goal2 = self.make_goal(
            self.d2_x, self.d2_y, self.d2_z,
            self.d2_qx, self.d2_qy, self.d2_qz, self.d2_qw,
            self.motion_time
        )

        self.send_two_goals_and_wait(goal1, goal2)
        if self.close_clamps_after_homing:
            self.close_both_clamps()


def main():
    rclpy.init()
    node = DualArmHoming()

    try:
        node.execute()
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()