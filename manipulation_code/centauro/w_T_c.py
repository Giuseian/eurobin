#### V2 - ENGLISH VERSION

#!/usr/bin/env python3

"""
Transform FoundationPose object poses from the camera frame to pelvis and world.

By default the script reads the newest:

    /home/user/shared_data/realsense/outputs_by_object/<timestamp>/poses.json

and writes, in the same folder:

    poses_world.json

The TF chain follows the same convention used by grasp_centauro_test_world.py and
push_box_xyz.py:

    ci/world -> ci/pelvis -> pelvis -> camera_frame
"""

from pathlib import Path
import json
import math
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


class FoundationPoseWorldTransformer(Node):

    def __init__(self):
        super().__init__('foundation_pose_world_transformer')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter(
            'outputs_root',
            '/home/user/shared_data/realsense/outputs_by_object',
        )
        self.declare_parameter('input_dir', '')
        self.declare_parameter('input_filename', 'poses.json')
        self.declare_parameter('output_filename', 'poses_world.json')
        self.declare_parameter('cartesian_world_frame', 'ci/world')
        self.declare_parameter('cartesian_robot_base_frame', 'ci/pelvis')
        self.declare_parameter('robot_base_frame', 'pelvis')
        self.declare_parameter('camera_frame', 'D435_head_camera_link')
        self.declare_parameter('tf_lookup_timeout', 20.0)

        self.outputs_root = Path(
            str(self.get_parameter('outputs_root').value)
        ).expanduser()
        self.input_dir = str(self.get_parameter('input_dir').value).strip()
        self.input_filename = str(self.get_parameter('input_filename').value)
        self.output_filename = str(self.get_parameter('output_filename').value)
        self.cartesian_world_frame = str(
            self.get_parameter('cartesian_world_frame').value
        ).strip()
        self.cartesian_robot_base_frame = str(
            self.get_parameter('cartesian_robot_base_frame').value
        ).strip()
        self.robot_base_frame = str(
            self.get_parameter('robot_base_frame').value
        ).strip()
        self.camera_frame = str(self.get_parameter('camera_frame').value).strip()
        self.tf_lookup_timeout = float(self.get_parameter('tf_lookup_timeout').value)

    # ------------------------------------------------------------------
    # Quaternion and vector utilities, copied in spirit from grasp/push.
    # ------------------------------------------------------------------
    def quat_normalize(self, q):
        x, y, z, w = q
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 0.0:
            return IDENTITY_QUATERNION
        return (x / norm, y / norm, z / norm, w / norm)

    def quat_conjugate(self, q):
        x, y, z, w = q
        return (-x, -y, -z, w)

    def quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def quat_rotate_vector(self, q, v):
        q = self.quat_normalize(q)
        rotated = self.quat_multiply(
            self.quat_multiply(q, (v[0], v[1], v[2], 0.0)),
            self.quat_conjugate(q),
        )
        return (rotated[0], rotated[1], rotated[2])

    def quat_from_yaw(self, yaw):
        half_yaw = float(yaw) / 2.0
        return self.quat_normalize((0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)))

    def yaw_from_quat(self, q):
        x, y, z, w = self.quat_normalize(q)
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def vector_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def matrix_to_quat(self, matrix):
        m00, m01, m02 = matrix[0][:3]
        m10, m11, m12 = matrix[1][:3]
        m20, m21, m22 = matrix[2][:3]
        trace = m00 + m11 + m22

        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (m21 - m12) / scale
            qy = (m02 - m20) / scale
            qz = (m10 - m01) / scale
        elif m00 > m11 and m00 > m22:
            scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / scale
            qx = 0.25 * scale
            qy = (m01 + m10) / scale
            qz = (m02 + m20) / scale
        elif m11 > m22:
            scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / scale
            qx = (m01 + m10) / scale
            qy = 0.25 * scale
            qz = (m12 + m21) / scale
        else:
            scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / scale
            qx = (m02 + m20) / scale
            qy = (m12 + m21) / scale
            qz = 0.25 * scale

        return self.quat_normalize((qx, qy, qz, qw))

    def pose_from_transform(self, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (translation.x, translation.y, translation.z),
            self.quat_normalize((rotation.x, rotation.y, rotation.z, rotation.w)),
        )

    def compose_poses(
        self,
        first_position,
        first_orientation,
        second_position,
        second_orientation,
    ):
        rotated_second = self.quat_rotate_vector(first_orientation, second_position)
        position = self.vector_add(first_position, rotated_second)
        orientation = self.quat_normalize(
            self.quat_multiply(first_orientation, second_orientation)
        )
        return position, orientation

    def transform_pose(self, parent_child_position, parent_child_orientation, pose):
        position, orientation = pose
        return self.compose_poses(
            parent_child_position,
            parent_child_orientation,
            position,
            orientation,
        )

    # ------------------------------------------------------------------
    # TF input
    # ------------------------------------------------------------------
    def lookup_tf_pose(self, target_frame, source_frame, timeout_s, label):
        deadline = time.monotonic() + float(timeout_s)
        last_error = None

        self.get_logger().info(
            f'Looking up TF {target_frame} -> {source_frame} for {label}...'
        )

        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                pose = self.pose_from_transform(transform)
                self.get_logger().info(
                    f'TF found for {label}: {target_frame} -> {source_frame}'
                )
                return pose
            except TransformException as exc:
                last_error = exc
                rclpy.spin_once(self, timeout_sec=0.1)

        raise RuntimeError(
            f'TF {target_frame} -> {source_frame} not available after '
            f'{float(timeout_s):.1f} s: {last_error}'
        )

    def read_frame_transforms(self):
        world_pelvis_pose = self.lookup_tf_pose(
            self.cartesian_world_frame,
            self.cartesian_robot_base_frame,
            self.tf_lookup_timeout,
            'T_cartesio_world_pelvis',
        )
        pelvis_camera_pose = self.lookup_tf_pose(
            self.robot_base_frame,
            self.camera_frame,
            self.tf_lookup_timeout,
            'T_pelvis_camera',
        )
        world_camera_pose = self.compose_poses(
            world_pelvis_pose[0],
            world_pelvis_pose[1],
            pelvis_camera_pose[0],
            pelvis_camera_pose[1],
        )
        return world_pelvis_pose, pelvis_camera_pose, world_camera_pose

    # ------------------------------------------------------------------
    # JSON input/output
    # ------------------------------------------------------------------
    def resolve_input_dir(self):
        if self.input_dir:
            path = Path(self.input_dir).expanduser()
            if not path.is_absolute():
                path = self.outputs_root / path
            return path

        candidates = [
            path for path in self.outputs_root.iterdir()
            if path.is_dir() and (path / self.input_filename).is_file()
        ]
        if not candidates:
            raise FileNotFoundError(
                f'No directory containing {self.input_filename} found in {self.outputs_root}'
            )
        return max(candidates, key=lambda path: path.name)

    def parse_pose_entry(self, entry):
        if isinstance(entry, list):
            if len(entry) == 3 and all(isinstance(v, (int, float)) for v in entry):
                return (tuple(float(v) for v in entry), IDENTITY_QUATERNION)

            if len(entry) == 7 and all(isinstance(v, (int, float)) for v in entry):
                position = tuple(float(v) for v in entry[:3])
                orientation = self.quat_normalize(tuple(float(v) for v in entry[3:7]))
                return position, orientation

            if len(entry) == 4 and all(isinstance(v, (int, float)) for v in entry):
                position = tuple(float(v) for v in entry[:3])
                orientation = self.quat_from_yaw(float(entry[3]))
                return position, orientation

            if (
                len(entry) == 4
                and all(isinstance(row, list) and len(row) == 4 for row in entry)
            ):
                matrix = [[float(value) for value in row] for row in entry]
                position = (matrix[0][3], matrix[1][3], matrix[2][3])
                orientation = self.matrix_to_quat(matrix)
                return position, orientation

        if isinstance(entry, dict):
            position_raw = (
                entry.get('position')
                or entry.get('translation')
                or entry.get('xyz')
            )
            orientation_raw = (
                entry.get('orientation')
                or entry.get('quaternion')
                or entry.get('xyzw')
            )
            yaw_raw = entry.get('yaw_raw')
            yaw = entry.get('yaw')
            if position_raw is None and 'pose' in entry:
                return self.parse_pose_entry(entry['pose'])
            if (
                isinstance(position_raw, list)
                and len(position_raw) == 3
                and all(isinstance(v, (int, float)) for v in position_raw)
            ):
                position = tuple(float(v) for v in position_raw)
                orientation = IDENTITY_QUATERNION
                if (
                    isinstance(orientation_raw, list)
                    and len(orientation_raw) == 4
                    and all(isinstance(v, (int, float)) for v in orientation_raw)
                ):
                    orientation = self.quat_normalize(
                        tuple(float(v) for v in orientation_raw)
                    )
                elif isinstance(yaw_raw, (int, float)):
                    orientation = self.quat_from_yaw(float(yaw_raw))
                elif isinstance(yaw, (int, float)):
                    orientation = self.quat_from_yaw(float(yaw))
                return position, orientation

        raise ValueError(
            'Unsupported pose format. Expected [x, y, z], '
            '[x, y, z, qx, qy, qz, qw], a dictionary with position/orientation, '
            'or a 4x4 matrix.'
        )

    def pose_to_json(self, pose):
        position, orientation = pose
        return {
            'position': [float(v) for v in position],
            'orientation_xyzw': [float(v) for v in orientation],
            'yaw': float(self.yaw_from_quat(orientation)),
        }

    def build_output(self, poses_by_image, world_pelvis_pose, pelvis_camera_pose, world_camera_pose):
        output = {
            'metadata': {
                'source_file': self.input_filename,
                'frames': {
                    'camera': self.camera_frame,
                    'pelvis': self.robot_base_frame,
                    'world': self.cartesian_world_frame,
                    'cartesian_robot_base': self.cartesian_robot_base_frame,
                },
                'transforms': {
                    'world_T_pelvis': self.pose_to_json(world_pelvis_pose),
                    'pelvis_T_camera': self.pose_to_json(pelvis_camera_pose),
                    'world_T_camera': self.pose_to_json(world_camera_pose),
                },
            },
            'poses': {},
        }

        for image_name, objects in poses_by_image.items():
            if not isinstance(objects, dict):
                raise ValueError(f'{image_name}: expected an object-to-pose dictionary')

            output['poses'][image_name] = {}
            for object_name, camera_entry in objects.items():
                camera_pose = self.parse_pose_entry(camera_entry)
                pelvis_pose = self.transform_pose(
                    pelvis_camera_pose[0],
                    pelvis_camera_pose[1],
                    camera_pose,
                )
                world_pose = self.transform_pose(
                    world_camera_pose[0],
                    world_camera_pose[1],
                    camera_pose,
                )
                output['poses'][image_name][object_name] = {
                    'camera': self.pose_to_json(camera_pose),
                    'pelvis': self.pose_to_json(pelvis_pose),
                    'world': self.pose_to_json(world_pose),
                }

        return output

    def run(self):
        input_dir = self.resolve_input_dir()
        input_path = input_dir / self.input_filename
        output_path = input_dir / self.output_filename

        if not input_path.is_file():
            raise FileNotFoundError(f'Pose file not found: {input_path}')

        with input_path.open('r', encoding='utf-8') as handle:
            poses_by_image = json.load(handle)
        if not isinstance(poses_by_image, dict):
            raise ValueError(f'{input_path}: the content must be a JSON object')

        self.get_logger().info(f'Reading camera poses from: {input_path}')
        world_pelvis_pose, pelvis_camera_pose, world_camera_pose = (
            self.read_frame_transforms()
        )
        output = self.build_output(
            poses_by_image,
            world_pelvis_pose,
            pelvis_camera_pose,
            world_camera_pose,
        )

        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(output, handle, indent=2)
            handle.write('\n')

        self.get_logger().info(f'Written to: {output_path}')
        return output_path


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseWorldTransformer()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()








#### V1 - ITALIAN VERSION 


# #!/usr/bin/env python3

# """
# Transform FoundationPose object poses from the camera frame to pelvis and world.

# By default the script reads the newest:

#     /home/user/shared_data/realsense/outputs_by_object/<timestamp>/poses.json

# and writes, in the same folder:

#     poses_world.json

# The TF chain follows the same convention used by grasp_centauro_test_world.py and
# push_box_xyz.py:

#     ci/world -> ci/pelvis -> pelvis -> camera_frame
# """

# from pathlib import Path
# import json
# import math
# import time

# import rclpy
# from rclpy.duration import Duration
# from rclpy.node import Node
# from rclpy.time import Time
# from tf2_ros import Buffer, TransformException, TransformListener


# IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


# class FoundationPoseWorldTransformer(Node):

#     def __init__(self):
#         super().__init__('foundation_pose_world_transformer')

#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         self.declare_parameter(
#             'outputs_root',
#             '/home/user/shared_data/realsense/outputs_by_object',
#         )
#         self.declare_parameter('input_dir', '')
#         self.declare_parameter('input_filename', 'poses.json')
#         self.declare_parameter('output_filename', 'poses_world.json')
#         self.declare_parameter('cartesian_world_frame', 'ci/world')
#         self.declare_parameter('cartesian_robot_base_frame', 'ci/pelvis')
#         self.declare_parameter('robot_base_frame', 'pelvis')
#         self.declare_parameter('camera_frame', 'D435_head_camera_link')
#         self.declare_parameter('tf_lookup_timeout', 20.0)

#         self.outputs_root = Path(
#             str(self.get_parameter('outputs_root').value)
#         ).expanduser()
#         self.input_dir = str(self.get_parameter('input_dir').value).strip()
#         self.input_filename = str(self.get_parameter('input_filename').value)
#         self.output_filename = str(self.get_parameter('output_filename').value)
#         self.cartesian_world_frame = str(
#             self.get_parameter('cartesian_world_frame').value
#         ).strip()
#         self.cartesian_robot_base_frame = str(
#             self.get_parameter('cartesian_robot_base_frame').value
#         ).strip()
#         self.robot_base_frame = str(
#             self.get_parameter('robot_base_frame').value
#         ).strip()
#         self.camera_frame = str(self.get_parameter('camera_frame').value).strip()
#         self.tf_lookup_timeout = float(self.get_parameter('tf_lookup_timeout').value)

#     # ------------------------------------------------------------------
#     # Quaternion and vector utilities, copied in spirit from grasp/push.
#     # ------------------------------------------------------------------
#     def quat_normalize(self, q):
#         x, y, z, w = q
#         norm = math.sqrt(x*x + y*y + z*z + w*w)
#         if norm <= 0.0:
#             return IDENTITY_QUATERNION
#         return (x / norm, y / norm, z / norm, w / norm)

#     def quat_conjugate(self, q):
#         x, y, z, w = q
#         return (-x, -y, -z, w)

#     def quat_multiply(self, q1, q2):
#         x1, y1, z1, w1 = q1
#         x2, y2, z2, w2 = q2
#         return (
#             w1*x2 + x1*w2 + y1*z2 - z1*y2,
#             w1*y2 - x1*z2 + y1*w2 + z1*x2,
#             w1*z2 + x1*y2 - y1*x2 + z1*w2,
#             w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         )

#     def quat_rotate_vector(self, q, v):
#         q = self.quat_normalize(q)
#         rotated = self.quat_multiply(
#             self.quat_multiply(q, (v[0], v[1], v[2], 0.0)),
#             self.quat_conjugate(q),
#         )
#         return (rotated[0], rotated[1], rotated[2])

#     def vector_add(self, a, b):
#         return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

#     def matrix_to_quat(self, matrix):
#         m00, m01, m02 = matrix[0][:3]
#         m10, m11, m12 = matrix[1][:3]
#         m20, m21, m22 = matrix[2][:3]
#         trace = m00 + m11 + m22

#         if trace > 0.0:
#             scale = math.sqrt(trace + 1.0) * 2.0
#             qw = 0.25 * scale
#             qx = (m21 - m12) / scale
#             qy = (m02 - m20) / scale
#             qz = (m10 - m01) / scale
#         elif m00 > m11 and m00 > m22:
#             scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
#             qw = (m21 - m12) / scale
#             qx = 0.25 * scale
#             qy = (m01 + m10) / scale
#             qz = (m02 + m20) / scale
#         elif m11 > m22:
#             scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
#             qw = (m02 - m20) / scale
#             qx = (m01 + m10) / scale
#             qy = 0.25 * scale
#             qz = (m12 + m21) / scale
#         else:
#             scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
#             qw = (m10 - m01) / scale
#             qx = (m02 + m20) / scale
#             qy = (m12 + m21) / scale
#             qz = 0.25 * scale

#         return self.quat_normalize((qx, qy, qz, qw))

#     def pose_from_transform(self, transform):
#         translation = transform.transform.translation
#         rotation = transform.transform.rotation
#         return (
#             (translation.x, translation.y, translation.z),
#             self.quat_normalize((rotation.x, rotation.y, rotation.z, rotation.w)),
#         )

#     def compose_poses(
#         self,
#         first_position,
#         first_orientation,
#         second_position,
#         second_orientation,
#     ):
#         rotated_second = self.quat_rotate_vector(first_orientation, second_position)
#         position = self.vector_add(first_position, rotated_second)
#         orientation = self.quat_normalize(
#             self.quat_multiply(first_orientation, second_orientation)
#         )
#         return position, orientation

#     def transform_pose(self, parent_child_position, parent_child_orientation, pose):
#         position, orientation = pose
#         return self.compose_poses(
#             parent_child_position,
#             parent_child_orientation,
#             position,
#             orientation,
#         )

#     # ------------------------------------------------------------------
#     # TF input
#     # ------------------------------------------------------------------
#     def lookup_tf_pose(self, target_frame, source_frame, timeout_s, label):
#         deadline = time.monotonic() + float(timeout_s)
#         last_error = None

#         self.get_logger().info(
#             f'Cerco TF {target_frame} -> {source_frame} per {label}...'
#         )

#         while rclpy.ok() and time.monotonic() < deadline:
#             try:
#                 transform = self.tf_buffer.lookup_transform(
#                     target_frame,
#                     source_frame,
#                     Time(),
#                     timeout=Duration(seconds=0.2),
#                 )
#                 pose = self.pose_from_transform(transform)
#                 self.get_logger().info(
#                     f'TF trovato per {label}: {target_frame} -> {source_frame}'
#                 )
#                 return pose
#             except TransformException as exc:
#                 last_error = exc
#                 rclpy.spin_once(self, timeout_sec=0.1)

#         raise RuntimeError(
#             f'TF {target_frame} -> {source_frame} non disponibile dopo '
#             f'{float(timeout_s):.1f} s: {last_error}'
#         )

#     def read_frame_transforms(self):
#         world_pelvis_pose = self.lookup_tf_pose(
#             self.cartesian_world_frame,
#             self.cartesian_robot_base_frame,
#             self.tf_lookup_timeout,
#             'T_cartesio_world_pelvis',
#         )
#         pelvis_camera_pose = self.lookup_tf_pose(
#             self.robot_base_frame,
#             self.camera_frame,
#             self.tf_lookup_timeout,
#             'T_pelvis_camera',
#         )
#         world_camera_pose = self.compose_poses(
#             world_pelvis_pose[0],
#             world_pelvis_pose[1],
#             pelvis_camera_pose[0],
#             pelvis_camera_pose[1],
#         )
#         return world_pelvis_pose, pelvis_camera_pose, world_camera_pose

#     # ------------------------------------------------------------------
#     # JSON input/output
#     # ------------------------------------------------------------------
#     def resolve_input_dir(self):
#         if self.input_dir:
#             path = Path(self.input_dir).expanduser()
#             if not path.is_absolute():
#                 path = self.outputs_root / path
#             return path

#         candidates = [
#             path for path in self.outputs_root.iterdir()
#             if path.is_dir() and (path / self.input_filename).is_file()
#         ]
#         if not candidates:
#             raise FileNotFoundError(
#                 f'Nessuna cartella con {self.input_filename} in {self.outputs_root}'
#             )
#         return max(candidates, key=lambda path: path.name)

#     def parse_pose_entry(self, entry):
#         if isinstance(entry, list):
#             if len(entry) == 3 and all(isinstance(v, (int, float)) for v in entry):
#                 return (tuple(float(v) for v in entry), IDENTITY_QUATERNION)

#             if len(entry) == 7 and all(isinstance(v, (int, float)) for v in entry):
#                 position = tuple(float(v) for v in entry[:3])
#                 orientation = self.quat_normalize(tuple(float(v) for v in entry[3:7]))
#                 return position, orientation

#             if (
#                 len(entry) == 4
#                 and all(isinstance(row, list) and len(row) == 4 for row in entry)
#             ):
#                 matrix = [[float(value) for value in row] for row in entry]
#                 position = (matrix[0][3], matrix[1][3], matrix[2][3])
#                 orientation = self.matrix_to_quat(matrix)
#                 return position, orientation

#         if isinstance(entry, dict):
#             position_raw = (
#                 entry.get('position')
#                 or entry.get('translation')
#                 or entry.get('xyz')
#             )
#             orientation_raw = (
#                 entry.get('orientation')
#                 or entry.get('quaternion')
#                 or entry.get('xyzw')
#             )
#             if position_raw is None and 'pose' in entry:
#                 return self.parse_pose_entry(entry['pose'])
#             if (
#                 isinstance(position_raw, list)
#                 and len(position_raw) == 3
#                 and all(isinstance(v, (int, float)) for v in position_raw)
#             ):
#                 position = tuple(float(v) for v in position_raw)
#                 orientation = IDENTITY_QUATERNION
#                 if (
#                     isinstance(orientation_raw, list)
#                     and len(orientation_raw) == 4
#                     and all(isinstance(v, (int, float)) for v in orientation_raw)
#                 ):
#                     orientation = self.quat_normalize(
#                         tuple(float(v) for v in orientation_raw)
#                     )
#                 return position, orientation

#         raise ValueError(
#             'Formato posa non supportato. Attesi [x, y, z], '
#             '[x, y, z, qx, qy, qz, qw], dict con position/orientation, '
#             'oppure matrice 4x4.'
#         )

#     def pose_to_json(self, pose):
#         position, orientation = pose
#         return {
#             'position': [float(v) for v in position],
#             'orientation_xyzw': [float(v) for v in orientation],
#         }

#     def build_output(self, poses_by_image, world_pelvis_pose, pelvis_camera_pose, world_camera_pose):
#         output = {
#             'metadata': {
#                 'source_file': self.input_filename,
#                 'frames': {
#                     'camera': self.camera_frame,
#                     'pelvis': self.robot_base_frame,
#                     'world': self.cartesian_world_frame,
#                     'cartesian_robot_base': self.cartesian_robot_base_frame,
#                 },
#                 'transforms': {
#                     'world_T_pelvis': self.pose_to_json(world_pelvis_pose),
#                     'pelvis_T_camera': self.pose_to_json(pelvis_camera_pose),
#                     'world_T_camera': self.pose_to_json(world_camera_pose),
#                 },
#             },
#             'poses': {},
#         }

#         for image_name, objects in poses_by_image.items():
#             if not isinstance(objects, dict):
#                 raise ValueError(f'{image_name}: atteso dizionario oggetto -> posa')

#             output['poses'][image_name] = {}
#             for object_name, camera_entry in objects.items():
#                 camera_pose = self.parse_pose_entry(camera_entry)
#                 pelvis_pose = self.transform_pose(
#                     pelvis_camera_pose[0],
#                     pelvis_camera_pose[1],
#                     camera_pose,
#                 )
#                 world_pose = self.transform_pose(
#                     world_camera_pose[0],
#                     world_camera_pose[1],
#                     camera_pose,
#                 )
#                 output['poses'][image_name][object_name] = {
#                     'camera': self.pose_to_json(camera_pose),
#                     'pelvis': self.pose_to_json(pelvis_pose),
#                     'world': self.pose_to_json(world_pose),
#                 }

#         return output

#     def run(self):
#         input_dir = self.resolve_input_dir()
#         input_path = input_dir / self.input_filename
#         output_path = input_dir / self.output_filename

#         if not input_path.is_file():
#             raise FileNotFoundError(f'File pose non trovato: {input_path}')

#         with input_path.open('r', encoding='utf-8') as handle:
#             poses_by_image = json.load(handle)
#         if not isinstance(poses_by_image, dict):
#             raise ValueError(f'{input_path}: il contenuto deve essere un oggetto JSON')

#         self.get_logger().info(f'Leggo pose camera da: {input_path}')
#         world_pelvis_pose, pelvis_camera_pose, world_camera_pose = (
#             self.read_frame_transforms()
#         )
#         output = self.build_output(
#             poses_by_image,
#             world_pelvis_pose,
#             pelvis_camera_pose,
#             world_camera_pose,
#         )

#         with output_path.open('w', encoding='utf-8') as handle:
#             json.dump(output, handle, indent=2)
#             handle.write('\n')

#         self.get_logger().info(f'Scritto: {output_path}')
#         return output_path


# def main(args=None):
#     rclpy.init(args=args)
#     node = FoundationPoseWorldTransformer()
#     try:
#         node.run()
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
