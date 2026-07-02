#### V2 

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cartesian_interface_ros.action import ReachPose
from geometry_msgs.msg import Pose


class DualArmHoming(Node):

    def __init__(self):
        super().__init__('dual_arm_homing')

        self.client_1 = ActionClient(self, ReachPose, '/dagana_1_tcp/reach')
        self.client_2 = ActionClient(self, ReachPose, '/dagana_2_tcp/reach')

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

        self.declare_parameter('motion_time', 10.0)

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







#### V1 

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from cartesian_interface_ros.action import ReachPose
# from geometry_msgs.msg import Pose


# class DualArmHoming(Node):

#     def __init__(self):
#         super().__init__('dual_arm_homing')

#         self.client_1 = ActionClient(self, ReachPose, '/dagana_1_tcp/reach')
#         self.client_2 = ActionClient(self, ReachPose, '/dagana_2_tcp/reach')

#         # # ===== POSIZIONI ASSOLUTE HOMING =====
#         # self.declare_parameter('d1_x', 0.50693)
#         # self.declare_parameter('d1_y', 0.1753)
#         # self.declare_parameter('d1_z', 0.25352)

#         # self.declare_parameter('d2_x', 0.52485)
#         # self.declare_parameter('d2_y', -0.22406)
#         # self.declare_parameter('d2_z', 0.27441)

#         # # ===== ORIENTAZIONI HOMING =====
#         # self.declare_parameter('d1_qx', 0.24991)
#         # self.declare_parameter('d1_qy', 0.50097)
#         # self.declare_parameter('d1_qz', 0.82353)
#         # self.declare_parameter('d1_qw', -0.091496)

#         # self.declare_parameter('d2_qx', 0.091491)
#         # self.declare_parameter('d2_qy', 0.82353)
#         # self.declare_parameter('d2_qz', 0.50097)
#         # self.declare_parameter('d2_qw', -0.24991)

#         # # ===== POSIZIONI ASSOLUTE HOMING =====
#         # self.declare_parameter('d1_x', 0.5)
#         # self.declare_parameter('d1_y', 0.3)
#         # self.declare_parameter('d1_z', 0.2)

#         # self.declare_parameter('d2_x', 0.5)
#         # self.declare_parameter('d2_y', -0.3)
#         # self.declare_parameter('d2_z', 0.2)

#         # # ===== ORIENTAZIONI HOMING =====
#         # self.declare_parameter('d1_qx', 0.0)
#         # self.declare_parameter('d1_qy', 0.0)
#         # self.declare_parameter('d1_qz', -0.7)
#         # self.declare_parameter('d1_qw', 0.7)

#         # self.declare_parameter('d2_qx', 0.7)
#         # self.declare_parameter('d2_qy', 0.7)
#         # self.declare_parameter('d2_qz', 0.0)
#         # self.declare_parameter('d2_qw', 0.0)

#         # # ===== POSIZIONI ASSOLUTE HOMING =====
#         # self.declare_parameter('d1_x', 0.53)
#         # self.declare_parameter('d1_y', 0.3)
#         # self.declare_parameter('d1_z', 0.29)

#         # self.declare_parameter('d2_x', 0.53)
#         # self.declare_parameter('d2_y', -0.3)
#         # self.declare_parameter('d2_z', 0.29)

#         # # ===== ORIENTAZIONI HOMING =====
#         # self.declare_parameter('d1_qx', 0.5)
#         # self.declare_parameter('d1_qy', 0.5)
#         # self.declare_parameter('d1_qz', 0.5)
#         # self.declare_parameter('d1_qw', -0.5)

#         # self.declare_parameter('d2_qx', -0.5)
#         # self.declare_parameter('d2_qy', -0.5)
#         # self.declare_parameter('d2_qz', -0.5)
#         # self.declare_parameter('d2_qw', 0.5)

#         # ===== POSIZIONI ASSOLUTE HOMING TCP =====
#         self.declare_parameter('d1_x', 0.8)
#         self.declare_parameter('d1_y', 0.3)
#         self.declare_parameter('d1_z', 0.25)

#         self.declare_parameter('d2_x', 0.8)
#         self.declare_parameter('d2_y', -0.35)
#         self.declare_parameter('d2_z', 0.25)

#         # ===== ORIENTAZIONI HOMING TCP =====
#         self.declare_parameter('d1_qx', 0.0)
#         self.declare_parameter('d1_qy', 0.7)
#         self.declare_parameter('d1_qz', 0.0)
#         self.declare_parameter('d1_qw', 0.7)

#         self.declare_parameter('d2_qx', 0.0)
#         self.declare_parameter('d2_qy', 0.7)
#         self.declare_parameter('d2_qz', 0.0)
#         self.declare_parameter('d2_qw', 0.7)

#         self.declare_parameter('motion_time', 10.0)

#         self.d1_x = self.get_parameter('d1_x').value
#         self.d1_y = self.get_parameter('d1_y').value
#         self.d1_z = self.get_parameter('d1_z').value

#         self.d2_x = self.get_parameter('d2_x').value
#         self.d2_y = self.get_parameter('d2_y').value
#         self.d2_z = self.get_parameter('d2_z').value

#         self.d1_qx = self.get_parameter('d1_qx').value
#         self.d1_qy = self.get_parameter('d1_qy').value
#         self.d1_qz = self.get_parameter('d1_qz').value
#         self.d1_qw = self.get_parameter('d1_qw').value

#         self.d2_qx = self.get_parameter('d2_qx').value
#         self.d2_qy = self.get_parameter('d2_qy').value
#         self.d2_qz = self.get_parameter('d2_qz').value
#         self.d2_qw = self.get_parameter('d2_qw').value

#         self.motion_time = self.get_parameter('motion_time').value

#     def make_goal(self, x, y, z, qx, qy, qz, qw, time_s):
#         goal = ReachPose.Goal()

#         pose = Pose()
#         pose.position.x = float(x)
#         pose.position.y = float(y)
#         pose.position.z = float(z)

#         pose.orientation.x = float(qx)
#         pose.orientation.y = float(qy)
#         pose.orientation.z = float(qz)
#         pose.orientation.w = float(qw)

#         goal.frames = [pose]
#         goal.time = [float(time_s)]
#         goal.incremental = False

#         return goal

#     def send_two_goals_and_wait(self, goal1, goal2):
#         self.get_logger().info('Waiting for action servers...')
#         self.client_1.wait_for_server()
#         self.client_2.wait_for_server()

#         self.get_logger().info('Sending absolute homing goals...')

#         future_1 = self.client_1.send_goal_async(goal1)
#         future_2 = self.client_2.send_goal_async(goal2)

#         rclpy.spin_until_future_complete(self, future_1)
#         rclpy.spin_until_future_complete(self, future_2)

#         gh1 = future_1.result()
#         gh2 = future_2.result()

#         if gh1 is None or not gh1.accepted:
#             raise RuntimeError('Goal dagana_1_tcp rejected')
#         if gh2 is None or not gh2.accepted:
#             raise RuntimeError('Goal dagana_2_tcp rejected')

#         r1 = gh1.get_result_async()
#         r2 = gh2.get_result_async()

#         rclpy.spin_until_future_complete(self, r1)
#         rclpy.spin_until_future_complete(self, r2)

#         if r1.result() is None:
#             raise RuntimeError('No result from dagana_1_tcp')
#         if r2.result() is None:
#             raise RuntimeError('No result from dagana_2_tcp')

#         self.get_logger().info('Homing motion completed successfully.')

#     def execute(self):
#         self.get_logger().info(
#             f"dagana_1_tcp pos: ({self.d1_x}, {self.d1_y}, {self.d1_z}) | "
#             f"quat: ({self.d1_qx}, {self.d1_qy}, {self.d1_qz}, {self.d1_qw})"
#         )
#         self.get_logger().info(
#             f"dagana_2_tcp pos: ({self.d2_x}, {self.d2_y}, {self.d2_z}) | "
#             f"quat: ({self.d2_qx}, {self.d2_qy}, {self.d2_qz}, {self.d2_qw})"
#         )

#         goal1 = self.make_goal(
#             self.d1_x, self.d1_y, self.d1_z,
#             self.d1_qx, self.d1_qy, self.d1_qz, self.d1_qw,
#             self.motion_time
#         )

#         goal2 = self.make_goal(
#             self.d2_x, self.d2_y, self.d2_z,
#             self.d2_qx, self.d2_qy, self.d2_qz, self.d2_qw,
#             self.motion_time
#         )

#         self.send_two_goals_and_wait(goal1, goal2)


# def main():
#     rclpy.init()
#     node = DualArmHoming()

#     try:
#         node.execute()
#     except Exception as e:
#         node.get_logger().error(f'Execution failed: {e}')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()