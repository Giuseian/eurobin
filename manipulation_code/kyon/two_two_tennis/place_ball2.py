#!/usr/bin/env python3

# Simple script: lift dagana_2 along Z by +0.2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class Dagana2Lift(Node):

    def __init__(self):
        super().__init__('dagana2_lift')

        self.client = ActionClient(self, ReachPose, '/dagana_2_base/reach')

        # lift amount
        self.lift_z = 0.2
        # motion duration
        self.time_motion = 15.0

    def make_goal(self):

        goal = ReachPose.Goal()

        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = self.lift_z

        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        # pose.orientation.x = 0.0
        # pose.orientation.y = 0.70710678
        # pose.orientation.z = 0.0
        # pose.orientation.w = 0.70710678

        goal.frames = [pose]
        goal.time = [self.time_motion]

        # movimento incrementale rispetto alla posa attuale
        goal.incremental = True

        return goal

    def execute(self):

        self.get_logger().info("Waiting for action server...")
        self.client.wait_for_server()

        goal = self.make_goal()

        self.get_logger().info("Sending lift command (+0.2 m on Z)")

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Goal rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Lift completed")


def main():
    rclpy.init()

    node = Dagana2Lift()

    try:
        node.execute()
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()