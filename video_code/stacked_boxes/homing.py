#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualDaganaLift(Node):

    def __init__(self):
        super().__init__('dual_dagana_lift')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_base/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # ===== POSIZIONI ASSOLUTE =====
        # Dagana 1
        self.declare_parameter('d1_x', 0.35936)
        self.declare_parameter('d1_y', 0.23365)
        self.declare_parameter('d1_z', 1.1709)

        # Dagana 2
        self.declare_parameter('d2_x', 0.35936)
        self.declare_parameter('d2_y', -0.23365)
        self.declare_parameter('d2_z', 1.1709)

        # ===== ORIENTAZIONE DAGANA 1 =====
        self.declare_parameter('d1_qx', 0.38483)
        self.declare_parameter('d1_qy', 0.56451)
        self.declare_parameter('d1_qz', 0.42583)
        self.declare_parameter('d1_qw', 0.59322)

        # ===== ORIENTAZIONE DAGANA 2 =====
        self.declare_parameter('d2_qx', 0.56451)
        self.declare_parameter('d2_qy', 0.38483)
        self.declare_parameter('d2_qz', 0.59322)
        self.declare_parameter('d2_qw', 0.42583)

        self.declare_parameter('motion_time', 10.0)

        # ===== LETTURA PARAMETRI POSIZIONE =====
        self.d1_x = self.get_parameter('d1_x').value
        self.d1_y = self.get_parameter('d1_y').value
        self.d1_z = self.get_parameter('d1_z').value

        self.d2_x = self.get_parameter('d2_x').value
        self.d2_y = self.get_parameter('d2_y').value
        self.d2_z = self.get_parameter('d2_z').value

        # ===== LETTURA PARAMETRI ORIENTAZIONE DAGANA 1 =====
        self.d1_qx = self.get_parameter('d1_qx').value
        self.d1_qy = self.get_parameter('d1_qy').value
        self.d1_qz = self.get_parameter('d1_qz').value
        self.d1_qw = self.get_parameter('d1_qw').value

        # ===== LETTURA PARAMETRI ORIENTAZIONE DAGANA 2 =====
        self.d2_qx = self.get_parameter('d2_qx').value
        self.d2_qy = self.get_parameter('d2_qy').value
        self.d2_qz = self.get_parameter('d2_qz').value
        self.d2_qw = self.get_parameter('d2_qw').value

        self.motion_time = self.get_parameter('motion_time').value

    def make_goal(self, x, y, z, qx, qy, qz, qw, time_s=5.0):
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
        goal.incremental = False  # assoluto

        return goal

    def send_two_goals_and_wait(self, goal1, goal2):
        self.get_logger().info('Waiting for action servers...')
        self.client_1.wait_for_server()
        self.client_2.wait_for_server()

        self.get_logger().info('Sending absolute goals (dual arms)...')

        future_1 = self.client_1.send_goal_async(goal1)
        future_2 = self.client_2.send_goal_async(goal2)

        rclpy.spin_until_future_complete(self, future_1)
        rclpy.spin_until_future_complete(self, future_2)

        gh1 = future_1.result()
        gh2 = future_2.result()

        if gh1 is None or not gh1.accepted:
            raise RuntimeError('Goal dagana_1 rejected')
        if gh2 is None or not gh2.accepted:
            raise RuntimeError('Goal dagana_2 rejected')

        r1 = gh1.get_result_async()
        r2 = gh2.get_result_async()

        rclpy.spin_until_future_complete(self, r1)
        rclpy.spin_until_future_complete(self, r2)

        if r1.result() is None:
            raise RuntimeError('No result from dagana_1')
        if r2.result() is None:
            raise RuntimeError('No result from dagana_2')

        self.get_logger().info('Motion completed successfully.')

    def execute(self):
        self.get_logger().info(
            f"D1 pos: ({self.d1_x}, {self.d1_y}, {self.d1_z}) | "
            f"quat: ({self.d1_qx}, {self.d1_qy}, {self.d1_qz}, {self.d1_qw})"
        )
        self.get_logger().info(
            f"D2 pos: ({self.d2_x}, {self.d2_y}, {self.d2_z}) | "
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


def main():
    rclpy.init()
    node = DualDaganaLift()

    try:
        node.execute()
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()