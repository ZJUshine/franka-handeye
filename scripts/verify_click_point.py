#!/usr/bin/env python3
"""
Franka Hand-Eye Calibration - Click-to-Move Verification

Click a pixel in the live camera feed to move the gripper to that 3D point.
Requires a RealSense depth stream and a valid calibration result.

Usage:
    python scripts/verify_click_point.py --host 172.16.0.2

Controls:
    Left click : pick a target point
    m          : move to selected point
    c          : clear selection
    q or ESC   : quit
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from franka_handeye import RobotController, CalibrationMode, load_calibration_result

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

WINDOW_NAME = "Click-to-Move Verification"


def start_realsense(resolution: tuple[int, int], fps: int, serial: str | None):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align, profile


def compute_point_base(
    point_cam: np.ndarray,
    T_calibration: np.ndarray,
    mode: CalibrationMode,
    T_gripper_base_current: np.ndarray,
) -> np.ndarray:
    point_cam_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])

    if mode == CalibrationMode.EYE_TO_HAND:
        point_base_h = T_calibration @ point_cam_h
    else:
        point_base_h = T_gripper_base_current @ (T_calibration @ point_cam_h)

    return point_base_h


def main():
    parser = argparse.ArgumentParser(
        description="Click a point in the camera image and move the gripper to that 3D location."
    )
    parser.add_argument("--host", default="172.16.0.2", help="Robot FCI IP address")
    parser.add_argument(
        "--calibration",
        default="data/hand-eye-calibration-output/calibration_result.json",
        help="Path to calibration result JSON",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["eye_in_hand", "eye_to_hand"],
        help="Calibration mode (auto-detected from calibration file if omitted)",
    )
    parser.add_argument("--serial", default=None, help="RealSense serial number (optional)")
    parser.add_argument("--width", type=int, default=1280, help="Color/depth width")
    parser.add_argument("--height", type=int, default=720, help="Color/depth height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Offset (m) to stop short along camera Z (positive moves closer to camera)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move the robot; only print target coordinates",
    )
    args = parser.parse_args()

    if rs is None:
        print("ERROR: pyrealsense2 is not installed.")
        return 1

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}")
        return 1

    T_calibration, detected_mode = load_calibration_result(calib_path)
    mode = CalibrationMode(args.mode) if args.mode else detected_mode

    print("=" * 70)
    print("Franka Hand-Eye Calibration - Click-to-Move Verification")
    print(f"Mode: {mode.value}")
    print(f"Calibration: {calib_path}")
    print("=" * 70)

    print("\nWARNING: This script will move the robot!")
    print("Ensure the workspace is clear and the E-stop is accessible.")
    try:
        input("\nPress ENTER to continue or Ctrl+C to abort: ")
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1

    # Connect robot
    try:
        robot = RobotController(args.host, dynamics_factor=0.1)
        print("✓ Robot connected")
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        return 1

    # Start camera
    try:
        pipeline, align, _ = start_realsense((args.width, args.height), args.fps, args.serial)
        print("✓ RealSense started")
    except Exception as e:
        print(f"ERROR: Failed to start RealSense: {e}")
        return 1

    shared = {
        "depth_frame": None,
        "intrinsics": None,
        "click_pixel": None,
        "point_cam": None,
        "point_base": None,
        "depth_m": None,
        "T_gripper_base": None,
        "quaternion": None,
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        depth_frame = shared["depth_frame"]
        if depth_frame is None:
            print("No depth frame available yet.")
            return

        depth_m = depth_frame.get_distance(x, y)
        if depth_m <= 0:
            print(f"No valid depth at pixel ({x}, {y}).")
            return

        intrinsics = shared["intrinsics"]
        if intrinsics is None:
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        point_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
        point_cam = np.array(point_cam, dtype=float)

        if args.offset != 0.0:
            point_cam[2] = max(0.0, point_cam[2] - args.offset)

        try:
            robot_state = robot.get_state()
            T_gripper_base_current = robot_state["O_T_EE"]
        except Exception as e:
            print(f"Failed to read robot state: {e}")
            return

        point_base_h = compute_point_base(point_cam, T_calibration, mode, T_gripper_base_current)
        quaternion = R.from_matrix(T_gripper_base_current[:3, :3]).as_quat().tolist()

        shared["click_pixel"] = (x, y)
        shared["point_cam"] = point_cam
        shared["point_base"] = point_base_h[:3]
        shared["depth_m"] = depth_m
        shared["T_gripper_base"] = T_gripper_base_current
        shared["quaternion"] = quaternion

        print("\n--- Target Selected ---")
        print(f"Pixel: ({x}, {y}), depth: {depth_m:.4f} m")
        print(f"Point (camera): {point_cam}")
        print(f"Point (base):   {point_base_h[:3]}")
        if args.offset != 0.0:
            print(f"Offset applied: {args.offset:.4f} m (camera Z)")
        print("Press 'm' to move, 'c' to clear, 'q' to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            shared["depth_frame"] = depth_frame
            shared["intrinsics"] = depth_frame.profile.as_video_stream_profile().intrinsics

            color_image = np.asanyarray(color_frame.get_data())

            if shared["click_pixel"] is not None:
                x, y = shared["click_pixel"]
                cv2.drawMarker(color_image, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                if shared["depth_m"] is not None:
                    label = f"{shared['depth_m']:.3f} m"
                    cv2.putText(
                        color_image,
                        label,
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow(WINDOW_NAME, color_image)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                shared["click_pixel"] = None
                shared["point_cam"] = None
                shared["point_base"] = None
                shared["depth_m"] = None
                shared["T_gripper_base"] = None
                shared["quaternion"] = None
                print("Selection cleared.")
            if key == ord("m"):
                if shared["point_base"] is None or shared["quaternion"] is None:
                    print("No target selected yet. Click a point first.")
                    continue
                if args.dry_run:
                    print(f"Dry run: would move to {shared['point_base']}")
                    continue

                try:
                    target_pos = shared["point_base"].tolist()
                    target_quat = shared["quaternion"]
                    print("🤖 Moving to selected point...")
                    robot.move_cartesian(target_pos, target_quat)
                    print("✓ Move complete")
                except Exception as e:
                    print(f"Move failed: {e}")

    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
