"""
Running Instructions: 
python ./move_d435_head_joint.py -p 0.3
"""

#!/usr/bin/env python3

import argparse
import time

import rclpy
from rclpy.node import Node
from xbot_msgs.msg import JointCommand
from xbot_msgs.srv import SetControlMask


class D435HeadJointCommander(Node):
    def __init__(self):
        super().__init__("d435_head_joint_commander")

        # Publisher for joint commands
        self.pub = self.create_publisher(
            JointCommand,
            "/xbotcore/command",
            10,
        )

        # Service client to set control mask
        self.mask_client = self.create_client(
            SetControlMask,
            "/xbotcore/joint_master/set_control_mask",
        )

    def set_control_mask(self):
        """
        Enable POSITION + IMPEDANCE control mode.
        This is required for XBotCore to accept position commands.
        """
        req = SetControlMask.Request()

        # POSITION (1) + IMPEDANCE (24) = 25
        req.ctrl_mask = (
            SetControlMask.Request.POSITION
            + SetControlMask.Request.IMPEDANCE
        )

        if not self.mask_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Control mask service not available.")
            return False

        future = self.mask_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().warn("set_control_mask call failed.")
            return False

        self.get_logger().info(
            f"Control mask set: {future.result().success}, {future.result().message}"
        )
        return future.result().success

    def send_position(
        self,
        position,
        duration=2.0,
        rate_hz=50.0,
        stiffness=100.0,
        damping=5.0,
        ctrl_mode=1,
    ):
        """
        Send position commands to the d435_head_joint.

        Args:
            position (float): target joint position [rad]
            duration (float): how long to send the command [s]
            rate_hz (float): publishing frequency
            stiffness (float): joint stiffness
            damping (float): joint damping
            ctrl_mode (int): control mode (1 = position)
        """

        period = 1.0 / rate_hz
        steps = int(duration * rate_hz)

        msg = JointCommand()
        msg.name = ["d435_head_joint"]
        msg.position = [float(position)]
        msg.stiffness = [float(stiffness)]
        msg.damping = [float(damping)]
        msg.ctrl_mode = [int(ctrl_mode)]

        self.get_logger().info(
            f"Sending command: pos={position}, stiff={stiffness}, damp={damping}, mode={ctrl_mode}"
        )

        for _ in range(steps):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            time.sleep(period)


def main():
    parser = argparse.ArgumentParser(description="Control d435_head_joint")

    parser.add_argument(
        "--position",
        "-p",
        type=float,
        required=True,
        help="Target joint position in radians",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=2.0,
        help="Command duration in seconds",
    )

    parser.add_argument(
        "--rate",
        "-r",
        type=float,
        default=50.0,
        help="Publishing rate (Hz)",
    )

    parser.add_argument(
        "--stiffness",
        "-k",
        type=float,
        default=0.0,
        help="Joint stiffness",
    )

    parser.add_argument(
        "--damping",
        "-b",
        type=float,
        default=0.0,
        help="Joint damping",
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=int,
        default=1,
        help="Control mode (1 = position)",
    )

    args = parser.parse_args()

    rclpy.init()
    node = D435HeadJointCommander()

    # Enable control mask before sending commands
    node.set_control_mask()

    # Send command
    node.send_position(
        position=args.position,
        duration=args.duration,
        rate_hz=args.rate,
        stiffness=args.stiffness,
        damping=args.damping,
        ctrl_mode=args.mode,
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()