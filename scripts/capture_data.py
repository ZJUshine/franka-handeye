import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path

# Set default server IP (can be overridden by env var)
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from franky import Robot, JointMotion, Affine

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

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
        self.board = cv2.aruco.CharucoBoard(
            (params['board_size'][0], params['board_size'][1]),
            params['square_length'],
            params['marker_length'],
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        )
        self.dictionary = self.board.getDictionary()
        self.params = cv2.aruco.DetectorParameters()

    def detect(self, image, K, D):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, K, D, None, None)
                return valid, rvec, tvec, charuco_corners
        return False, None, None, None

class ManualCaptureApp:
    def __init__(self, root, robot, camera, detector, output_dir, K, D):
        self.root = root
        self.root.title("Franka Hand-Eye Capture (Manual Mode)")
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.output_dir = output_dir
        self.K = K
        self.D = D
        
        self.captured_count = 0
        self.captured_poses = []
        self.is_moving = False

        # UI Layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video Feed
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=5, pady=5)

        # Controls Frame
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Robot Control")
        self.controls_frame.grid(row=0, column=1, sticky="ns", padx=5)

        # Joint Sliders
        self.sliders = []
        self.joint_vars = []
        current_joints = self.robot.state.q
        
        for i in range(7):
            lbl = ttk.Label(self.controls_frame, text=f"Joint {i+1}")
            lbl.pack(pady=2)
            var = tk.DoubleVar(value=current_joints[i])
            self.joint_vars.append(var)
            scale = ttk.Scale(self.controls_frame, from_=-2.9, to=2.9, variable=var, orient="horizontal", length=200)
            scale.pack(pady=2)
            # Bind release event to move robot
            scale.bind("<ButtonRelease-1>", lambda e, idx=i: self.on_slider_release())
            self.sliders.append(scale)

        # Go Button (Explicit move)
        self.btn_move = ttk.Button(self.controls_frame, text="Move to Sliders", command=self.move_to_sliders)
        self.btn_move.pack(pady=10, fill=tk.X)

        # Capture Frame
        self.capture_frame = ttk.LabelFrame(self.main_frame, text="Capture")
        self.capture_frame.grid(row=1, column=1, sticky="ew", padx=5)
        
        self.lbl_count = ttk.Label(self.capture_frame, text="Captured: 0/12")
        self.lbl_count.pack(pady=5)
        
        self.btn_capture = ttk.Button(self.capture_frame, text="Capture Pose", command=self.capture_pose)
        self.btn_capture.pack(pady=5, fill=tk.X)
        
        self.btn_save = ttk.Button(self.capture_frame, text="Save Joint Poses", command=self.save_joint_poses)
        self.btn_save.pack(pady=5, fill=tk.X)

        # Start Loops
        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_slider_release(self):
        # Optional: Auto-move on release
        pass

    def move_to_sliders(self):
        if self.is_moving: return
        target_q = [v.get() for v in self.joint_vars]
        threading.Thread(target=self._move_thread, args=(target_q,)).start()

    def _move_thread(self, target_q):
        self.is_moving = True
        self.btn_move.state(['disabled'])
        try:
            self.robot.move(JointMotion(target_q))
        except Exception as e:
            print(f"Move failed: {e}")
        finally:
            self.is_moving = False
            # Update sliders to actual final position (in case of error or limit)
            final_q = self.robot.state.q
            for i, val in enumerate(final_q):
                self.joint_vars[i].set(val)
            self.root.after(0, lambda: self.btn_move.state(['!disabled']))

    def update_video(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # Detection
            valid, rvec, tvec, corners = self.detector.detect(frame, self.K, self.D)
            if valid:
                cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)
                self.current_detection = (True, rvec, tvec)
            else:
                self.current_detection = (False, None, None)

            # Convert to Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 360)) # Resize for GUI
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.last_frame = frame # Store for saving

        self.root.after(33, self.update_video)

    def capture_pose(self):
        if not hasattr(self, 'last_frame') or self.last_frame is None:
            return
            
        state = self.robot.state
        q = state.q
        O_T_EE = state.O_T_EE
        
        valid, rvec, tvec = self.current_detection
        
        # Save
        pose_idx = self.captured_count
        pose_dir = self.output_dir / f"pose_{pose_idx:02d}"
        pose_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(pose_dir / "image.png"), self.last_frame)
        
        data = {
            "joint_pose": list(q),
            "O_T_EE": O_T_EE.tolist() if hasattr(O_T_EE, "tolist") else O_T_EE,
            "camera_intrinsics": self.K.tolist(),
            "dist_coeffs": self.D.tolist(),
            "charuco_detected": valid,
        }
        
        if valid:
            data["T_cam_target_rvec"] = rvec.tolist()
            data["T_cam_target_tvec"] = tvec.tolist()
            
        with open(pose_dir / "data.json", 'w') as f:
            json.dump(data, f, indent=4)
            
        self.captured_poses.append(list(q))
        self.captured_count += 1
        self.lbl_count.configure(text=f"Captured: {self.captured_count}/12")
        print(f"Captured pose {pose_idx}")

    def save_joint_poses(self):
        path = "config/joint_poses.yaml"
        with open(path, 'w') as f:
            yaml.dump({"joint_poses": self.captured_poses}, f)
        messagebox.showinfo("Saved", f"Saved {len(self.captured_poses)} poses to {path}")

    def on_close(self):
        self.camera.stop()
        self.root.destroy()

def run_automatic(robot, camera, detector, joint_poses, output_dir, K, D):
    print(f"Starting Automatic Capture of {len(joint_poses)} poses...")
    
    # Optional: Move home first?
    # robot.move(JointMotion(joint_poses[0])) 
    
    for i, pose in enumerate(joint_poses):
        print(f"\n--- Pose {i+1}/{len(joint_poses)} ---")
        try:
            robot.move(JointMotion(pose))
            time.sleep(0.5) # Stabilization
            
            frame = camera.get_frame()
            if frame is None: continue
            
            state = robot.state
            O_T_EE = state.O_T_EE
            
            valid, rvec, tvec, _ = detector.detect(frame, K, D)
            
            if valid:
                print("Charuco detected.")
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.1)
            else:
                print("WARNING: Charuco NOT detected.")
                
            pose_dir = output_dir / f"pose_{i:02d}"
            pose_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(pose_dir / "image.png"), frame)
            
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
                
        except Exception as e:
            print(f"Error at pose {i}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Franka Hand-Eye Calibration Capture")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP")
    parser.add_argument("--output", default="data/captured-data", help="Output dir")
    parser.add_argument("--manual", action="store_true", help="Force manual mode")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Hardware
    print("Initializing Camera...")
    camera = RealSenseCamera()
    K, D = camera.get_intrinsics_matrix()
    
    print(f"Connecting to Robot at {args.host}...")
    try:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = 0.1
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        camera.stop()
        return

    detector = CharucoDetector("charuco/calibration_board_parameters.yaml")

    # Check Config
    poses_path = Path("config/joint_poses.yaml")
    joint_poses = []
    if poses_path.exists():
        with open(poses_path, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'joint_poses' in data and data['joint_poses']:
                joint_poses = data['joint_poses']
    
    # Determine Mode
    mode_manual = args.manual or not joint_poses
    
    if mode_manual:
        print("Starting Manual GUI Mode...")
        root = tk.Tk()
        app = ManualCaptureApp(root, robot, camera, detector, output_dir, K, D)
        root.mainloop()
    else:
        run_automatic(robot, camera, detector, joint_poses, output_dir, K, D)
        camera.stop()

if __name__ == "__main__":
    main()
