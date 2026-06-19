from live_io import capture_and_estimate_pose

live_result = capture_and_estimate_pose(
    host_robot_compose_path="../robot_docker/compose.yml",
    robot_service="dev",
    robot_save_script="/home/user/PoseEstimation/pipeline/save_realsense_rgbd.py",
    robot_shared_data_root="/home/user/shared_data",
    perception_shared_data_root="/workspace/shared_data",
    sam_prompt="box",
    mesh_file="/workspace/shared_data/models/box.obj",
)

print("\nLIVE TEST OK")
print(f"timestamp: {live_result.frame.timestamp}")
print(f"rgb:       {live_result.frame.rgb_path}")
print(f"depth:     {live_result.frame.depth_npy_path}")
print(f"mask:      {live_result.mask_path}")
print(f"pose:      {live_result.pose_path}")
print(f"vis:       {live_result.visualization_path}")