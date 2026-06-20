from live_io import get_latest_live_frame, run_live_pose_estimation


def main() -> None:
    frame = get_latest_live_frame(
        perception_shared_data_root="/workspace/shared_data",
        frame_id="000000",
    )

    live_result = run_live_pose_estimation(
        frame=frame,
        sam_prompt="box",
        mesh_file="/workspace/shared_data/realsense/meshes/box_amazon_model/meshes/box_amazon_model.obj",
        sam_script_path="/workspace/PoseEstimation/pipeline/sam_script_fp.py",
        fp_script_path="/workspace/PoseEstimation/pipeline/run_fp_single_frame.py",
    )

    print("\nLIVE TEST OK")
    print(f"timestamp: {live_result.frame.timestamp}")
    print(f"rgb:       {live_result.frame.rgb_path}")
    print(f"depth:     {live_result.frame.depth_npy_path}")
    print(f"mask:      {live_result.mask_path}")
    print(f"pose:      {live_result.pose_path}")
    print(f"vis:       {live_result.visualization_path}")


if __name__ == "__main__":
    main()



# from live_io import capture_and_estimate_pose

# live_result = capture_and_estimate_pose(
#     host_robot_compose_path="../robot_docker/compose.yml",
#     robot_service="dev",
#     robot_save_script="/home/user/PoseEstimation/pipeline/save_realsense_rgbd.py",
#     robot_shared_data_root="/home/user/shared_data",
#     perception_shared_data_root="/workspace/shared_data",
#     sam_prompt="box",
#     mesh_file="/workspace/shared_data/models/box.obj",
# )

# print("\nLIVE TEST OK")
# print(f"timestamp: {live_result.frame.timestamp}")
# print(f"rgb:       {live_result.frame.rgb_path}")
# print(f"depth:     {live_result.frame.depth_npy_path}")
# print(f"mask:      {live_result.mask_path}")
# print(f"pose:      {live_result.pose_path}")
# print(f"vis:       {live_result.visualization_path}")