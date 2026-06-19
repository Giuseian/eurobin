from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LiveFrame:
    timestamp: str
    frame_id: str
    data_root: Path
    rgb_path: Path
    depth_npy_path: Path
    depth_png_path: Path
    intrinsics_path: Path


@dataclass
class LivePoseResult:
    frame: LiveFrame
    mask_path: Path
    pose_path: Path
    visualization_path: Path | None


def run_command(command: list[str], label: str) -> None:
    print(f"\n[LIVE_IO][{label}] Running:")
    print(" ".join(command))

    subprocess.run(command, check=True)


def get_latest_timestamp(rgb_root: Path) -> str:
    if not rgb_root.exists():
        raise FileNotFoundError(f"RGB root not found: {rgb_root}")

    timestamp_dirs = sorted(
        [p for p in rgb_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )

    if not timestamp_dirs:
        raise RuntimeError(f"No timestamp folders found in: {rgb_root}")

    return timestamp_dirs[-1].name


def build_live_frame(
    shared_data_root: str | Path,
    timestamp: str,
    frame_id: str = "000000",
) -> LiveFrame:
    shared_data_root = Path(shared_data_root)
    data_root = shared_data_root / "realsense"

    rgb_path = data_root / "rgb" / timestamp / f"{frame_id}.png"
    depth_npy_path = data_root / "depth" / timestamp / "npy" / f"{frame_id}.npy"
    depth_png_path = data_root / "depth" / timestamp / "png" / f"{frame_id}.png"
    intrinsics_path = data_root / "camera" / timestamp / "intrinsics.yaml"

    required_paths = [
        rgb_path,
        depth_npy_path,
        depth_png_path,
        intrinsics_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected live frame file not found: {path}")

    return LiveFrame(
        timestamp=timestamp,
        frame_id=frame_id,
        data_root=data_root,
        rgb_path=rgb_path,
        depth_npy_path=depth_npy_path,
        depth_png_path=depth_png_path,
        intrinsics_path=intrinsics_path,
    )


def capture_single_frame_from_robot(
    host_robot_compose_path: str | Path,
    robot_service: str = "dev",
    robot_save_script: str = "/home/user/PoseEstimation/pipeline/save_realsense_rgbd.py",
    robot_shared_data_root: str = "/home/user/shared_data",
    perception_shared_data_root: str | Path = "/workspace/shared_data",
    frame_id: str = "000000",
) -> LiveFrame:
    command = [
        "docker",
        "compose",
        "-f",
        str(host_robot_compose_path),
        "exec",
        "-T",
        robot_service,
        "python",
        robot_save_script,
        "--save_dir",
        robot_shared_data_root,
        "--max_frames",
        "1",
    ]

    run_command(command, label="capture_single_frame")

    perception_shared_data_root = Path(perception_shared_data_root)
    rgb_root = perception_shared_data_root / "realsense" / "rgb"

    timestamp = get_latest_timestamp(rgb_root)

    return build_live_frame(
        shared_data_root=perception_shared_data_root,
        timestamp=timestamp,
        frame_id=frame_id,
    )


def run_sam3_for_frame(
    frame: LiveFrame,
    sam_prompt: str,
    sam_script_path: str = "pipeline/sam_script_fp.py",
    score_threshold: float | None = None,
    mask_mode: str = "best",
) -> Path:
    command = [
        "run_sam3",
        "python",
        sam_script_path,
        "--data_root",
        str(frame.data_root),
        "--timestamp",
        frame.timestamp,
        "--prompt",
        sam_prompt,
        "--image_id",
        frame.frame_id,
        "--mask_mode",
        mask_mode,
    ]

    if score_threshold is not None:
        command.extend(["--score_threshold", str(score_threshold)])

    run_command(command, label="sam3")

    mask_path = frame.data_root / "masks" / frame.timestamp / f"{frame.frame_id}.png"

    if not mask_path.exists():
        raise FileNotFoundError(f"SAM3 mask was not created: {mask_path}")

    return mask_path


def run_foundationpose_for_frame(
    frame: LiveFrame,
    mesh_file: str | Path,
    fp_script_path: str = "pipeline/run_fp_single_frame.py",
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    debug: int = 1,
) -> Path:
    command = [
        "run_foundationpose",
        "python",
        fp_script_path,
        "--data_root",
        str(frame.data_root),
        "--timestamp",
        frame.timestamp,
        "--mesh_file",
        str(mesh_file),
        "--start_frame",
        frame.frame_id,
        "--max_frames",
        "1",
        "--est_refine_iter",
        str(est_refine_iter),
        "--track_refine_iter",
        str(track_refine_iter),
        "--debug",
        str(debug),
    ]

    run_command(command, label="foundationpose")

    pose_path = (
        frame.data_root
        / "outputs"
        / frame.timestamp
        / "ob_in_cam"
        / f"{frame.frame_id}.txt"
    )

    if not pose_path.exists():
        raise FileNotFoundError(f"FoundationPose pose was not created: {pose_path}")

    return pose_path


def run_live_pose_estimation(
    frame: LiveFrame,
    sam_prompt: str,
    mesh_file: str | Path,
    sam_script_path: str = "pipeline/sam_script_fp.py",
    fp_script_path: str = "pipeline/run_fp_single_frame.py",
    score_threshold: float | None = None,
    mask_mode: str = "best",
    est_refine_iter: int = 5,
    track_refine_iter: int = 2,
    debug: int = 1,
) -> LivePoseResult:
    mask_path = run_sam3_for_frame(
        frame=frame,
        sam_prompt=sam_prompt,
        sam_script_path=sam_script_path,
        score_threshold=score_threshold,
        mask_mode=mask_mode,
    )

    pose_path = run_foundationpose_for_frame(
        frame=frame,
        mesh_file=mesh_file,
        fp_script_path=fp_script_path,
        est_refine_iter=est_refine_iter,
        track_refine_iter=track_refine_iter,
        debug=debug,
    )

    visualization_path = (
        frame.data_root
        / "outputs"
        / frame.timestamp
        / "vis"
        / f"{frame.frame_id}.png"
    )

    if not visualization_path.exists():
        visualization_path = None

    return LivePoseResult(
        frame=frame,
        mask_path=mask_path,
        pose_path=pose_path,
        visualization_path=visualization_path,
    )


def capture_and_estimate_pose(
    host_robot_compose_path: str | Path,
    robot_service: str,
    robot_save_script: str,
    robot_shared_data_root: str,
    perception_shared_data_root: str | Path,
    sam_prompt: str,
    mesh_file: str | Path,
) -> LivePoseResult:
    frame = capture_single_frame_from_robot(
        host_robot_compose_path=host_robot_compose_path,
        robot_service=robot_service,
        robot_save_script=robot_save_script,
        robot_shared_data_root=robot_shared_data_root,
        perception_shared_data_root=perception_shared_data_root,
    )

    return run_live_pose_estimation(
        frame=frame,
        sam_prompt=sam_prompt,
        mesh_file=mesh_file,
    )