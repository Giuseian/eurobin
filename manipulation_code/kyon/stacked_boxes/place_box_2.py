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

        self.lift_z = 0.3
        self.motion_time = 15.0

    def make_goal(self, dx, dy, dz,
                  qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                  time_s=5.0,
                  incremental=True):
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

    def send_two_goals_and_wait(self, goal1, goal2):
        self.get_logger().info('Waiting for action servers...')
        self.client_1.wait_for_server()
        self.client_2.wait_for_server()

        self.get_logger().info('Sending incremental upward motion...')
        future_1 = self.client_1.send_goal_async(goal1)
        future_2 = self.client_2.send_goal_async(goal2)

        rclpy.spin_until_future_complete(self, future_1)
        rclpy.spin_until_future_complete(self, future_2)

        goal_handle_1 = future_1.result()
        goal_handle_2 = future_2.result()

        if goal_handle_1 is None or not goal_handle_1.accepted:
            raise RuntimeError('Goal dagana_1 rejected')
        if goal_handle_2 is None or not goal_handle_2.accepted:
            raise RuntimeError('Goal dagana_2 rejected')

        result_future_1 = goal_handle_1.get_result_async()
        result_future_2 = goal_handle_2.get_result_async()

        rclpy.spin_until_future_complete(self, result_future_1)
        rclpy.spin_until_future_complete(self, result_future_2)

        if result_future_1.result() is None:
            raise RuntimeError('No result from dagana_1')
        if result_future_2.result() is None:
            raise RuntimeError('No result from dagana_2')

        self.get_logger().info('Motion completed successfully.')

    def execute(self):
        goal1 = self.make_goal(
            dx=0.0,
            dy=0.0,
            dz=self.lift_z,
            time_s=self.motion_time,
            incremental=True
        )

        goal2 = self.make_goal(
            dx=0.0,
            dy=0.0,
            dz=self.lift_z,
            time_s=self.motion_time,
            incremental=True
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