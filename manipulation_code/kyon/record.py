#!/usr/bin/env python3

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DualImageSaver(Node):
    def __init__(self):
        super().__init__('dual_image_saver')

        self.bridge = CvBridge()

        self.rgb_topic = '/zed_front_up/image'
        self.depth_topic = '/zed_front_up/depth_image'

        self.rgb_dir = Path('/home/user/image/rgb')
        self.depth_dir = Path('/home/user/image/depth')

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_count = 1
        self.depth_count = 1

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )

        self.get_logger().info(f'Subscribed to RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Subscribed to Depth topic: {self.depth_topic}')
        self.get_logger().info(f'RGB images will be saved in: {self.rgb_dir}')
        self.get_logger().info(f'Depth images will be saved in: {self.depth_dir}')

    def make_filename(self, directory: Path, index: int) -> str:
        return str(directory / f'{index:07d}.png')

    def rgb_callback(self, msg: Image) -> None:
        try:
            # Converte in OpenCV BGR8
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            filename = self.make_filename(self.rgb_dir, self.rgb_count)

            ok = cv2.imwrite(filename, cv_image)
            if not ok:
                self.get_logger().error(f'Failed to save RGB image: {filename}')
                return

            self.get_logger().info(f'Saved RGB: {filename}')
            self.rgb_count += 1

        except Exception as e:
            self.get_logger().error(f'RGB callback error: {e}')

    def depth_callback(self, msg: Image) -> None:
        try:
            # Mantiene il formato originale
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Caso 1: depth già uint16 -> salvataggio diretto
            if depth_image.dtype == np.uint16:
                depth_to_save = depth_image

            # Caso 2: depth float32/float64 -> conversione metri -> millimetri
            elif depth_image.dtype in (np.float32, np.float64):
                depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
                depth = np.clip(depth, 0.0, 65.535)   # max range representable in uint16 mm
                depth_to_save = (depth * 1000.0).astype(np.uint16)

            # Caso 3: altro tipo -> tentativo di conversione prudente
            else:
                self.get_logger().warn(
                    f'Unexpected depth dtype {depth_image.dtype}, converting to uint16.'
                )
                depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
                depth = np.clip(depth, 0, 65535)
                depth_to_save = depth.astype(np.uint16)

            filename = self.make_filename(self.depth_dir, self.depth_count)

            ok = cv2.imwrite(filename, depth_to_save)
            if not ok:
                self.get_logger().error(f'Failed to save depth image: {filename}')
                return

            self.get_logger().info(f'Saved Depth: {filename}')
            self.depth_count += 1

        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DualImageSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()