#!/usr/bin/env python3

# gz topic -e -t /world/default/dynamic_pose/info | grep -A 15 dagana

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class XbotHomingClient(Node):

    def __init__(self):
        super().__init__('xbot_homing_client')
        self.cli = self.create_client(SetBool, '/xbotcore/homing/switch')

    def send_homing_request(self, enable: bool = True):
        self.get_logger().info('Waiting for /xbotcore/homing/switch service...')

        if not self.cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Service /xbotcore/homing/switch not available')

        req = SetBool.Request()
        req.data = enable

        self.get_logger().info(f'Sending homing request: data={req.data}')
        future = self.cli.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None:
            raise RuntimeError('No response from /xbotcore/homing/switch')

        self.get_logger().info(
            f'Homing service response: success={result.success}, message="{result.message}"'
        )

        if not result.success:
            raise RuntimeError(f'Homing failed: {result.message}')


def main():
    rclpy.init()
    node = XbotHomingClient()

    try:
        node.send_homing_request(enable=True)
    except Exception as e:
        node.get_logger().error(f'Execution failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()