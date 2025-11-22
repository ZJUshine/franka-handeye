import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
from pathlib import Path

# Franky Remote imports
from franky import Robot, JointMotion

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get intrinsics
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            return None
            
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image

    def get_intrinsics_matrix(self):
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(self.intrinsics.coeffs)
        return K, dist_coeffs

    def stop(self):
        self.pipeline.stop()

class CharucoDetector:
    def __init__(self, params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            
        self.squares_x = params['board_size'][0]
        self.squares_y = params['board_size'][1]
        self.square_length = params['square_length']
        self.marker_length = params['marker_length']
        
        # Create Charuco board
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.dictionary
        )
        self.params = cv2.aruco.DetectorParameters()

    def detect(self, image, camera_matrix, dist_coeffs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )
        
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, camera_matrix, dist_coeffs, None, None
                )
                return valid, rvec, tvec, charuco_corners
                
        return False, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Franka Hand-Eye Calibration Data Capture")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--output", default="data/captured-data", help="Output directory")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configurations
    with open("config/joint_poses.yaml", 'r') as f:
        joint_poses = yaml.safe_load(f)['joint_poses']

    # Initialize Camera
    print("Initializing RealSense camera...")
    camera = RealSenseCamera()
    K, D = camera.get_intrinsics_matrix()
    print(f"Camera Intrinsics:\n{K}")

    # Initialize Charuco Detector
    detector = CharucoDetector("charuco/calibration_board_parameters.yaml")

    # Initialize Robot
    print(f"Connecting to robot at {args.host}...")
    # Note: FRANKY_SERVER_IP should be set in environment before running this script
    # or handled by the user. 
    if "FRANKY_SERVER_IP" not in os.environ:
        print("WARNING: FRANKY_SERVER_IP not set. Assuming localhost or already set.")
    
    try:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = 0.1
        
        print("Moving to initial home position...")
        # Move to first pose as home or just start loop
        
        for i, pose in enumerate(joint_poses):
            print(f"\n--- Pose {i+1}/{len(joint_poses)} ---")
            print(f"Moving to: {pose}")
            
            try:
                motion = JointMotion(pose)
                robot.move(motion)
            except Exception as e:
                print(f"Failed to move to pose {i}: {e}")
                continue

            # Wait for stabilization
            time.sleep(1.0)
            
            # Capture Data
            color_img = camera.get_frame()
            if color_img is None:
                print("Failed to get frame from camera!")
                continue

            # Get Robot State
            state = robot.state
            O_T_EE = state.O_T_EE # 4x4 homogeneous matrix (row-major or column-major? Franka is usually column-major in docs but numpy array structure matters. Franky/libfranka usually returns column-major flat array or 4x4. Let's check.)
            # In Franky, O_T_EE is usually a numpy array. 
            # Assuming it's a 4x4 matrix or 16-element array. 
            
            # Detect Board
            valid, rvec, tvec, corners = detector.detect(color_img, K, D)
            
            if valid:
                print("Charuco board detected!")
                # Draw axis for visualization (optional, strictly for debug)
                cv2.drawFrameAxes(color_img, K, D, rvec, tvec, 0.1)
            else:
                print("WARNING: Charuco board NOT detected in this pose.")
            
            # Save Data
            pose_dir = output_dir / f"pose_{i:02d}"
            pose_dir.mkdir(exist_ok=True)
            
            # Save Image
            cv2.imwrite(str(pose_dir / "image.png"), color_img)
            
            # Save Metadata
            data = {
                "joint_pose": pose,
                "O_T_EE": O_T_EE.tolist() if hasattr(O_T_EE, "tolist") else O_T_EE,
                "camera_intrinsics": K.tolist(),
                "dist_coeffs": D.tolist(),
                "charuco_detected": valid,
            }
            
            if valid:
                data["T_cam_target_rvec"] = rvec.tolist()
                data["T_cam_target_tvec"] = tvec.tolist()

            with open(pose_dir / "data.json", 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Saved data to {pose_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        camera.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    main()

