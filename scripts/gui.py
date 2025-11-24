import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
import shutil
from pathlib import Path
import dearpygui.dearpygui as dpg

# Set default server IP
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from franky import (
    Robot, 
    JointMotion, 
    CartesianVelocityMotion,
    Twist,
    Duration,
    Reaction,
    Measure,
    CartesianVelocityStopMotion
)

# --- Helper Classes (Copied/Adapted from capture_data.py) ---

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RealSenseCamera:
    def __init__(self, lazy=True):
        self.pipeline = None
        self.config = None
        self.profile = None
        if not lazy:
            self.initialize()

    def initialize(self):
        if self.pipeline:
            return True
        try:
            print("Attempting to start RealSense pipeline...")
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            self.color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
            print("RealSense pipeline started.")
            return True
        except RuntimeError as e:
            print(f"RealSense Init Error: {e}")
            if "Device or resource busy" in str(e):
                return False
            else:
                raise e

    def get_frame(self):
        if not self.pipeline:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        except Exception:
            return None


    def get_intrinsics_matrix(self):
        if not hasattr(self, 'intrinsics') or not self.intrinsics:
            return np.eye(3), np.zeros(5)
            
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(self.intrinsics.coeffs)
        return K, dist_coeffs

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

class CharucoDetector:
    def __init__(self, params_path):
        if not os.path.exists(params_path):
            # Fallback or dummy if file missing, but better to warn
            print(f"Warning: {params_path} not found.")
            self.board = None
            return

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
        if self.board is None:
            return False, None, None, None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, K, D, None, None)
                return valid, rvec, tvec, charuco_corners
        return False, None, None, None

# --- Application State ---

class AppState:
    def __init__(self):
        self.robot = None
        self.camera = None
        self.detector = None
        self.K = None
        self.D = None
        self._initialized = False  # Track initialization state
        
        self.output_dir = Path("data/captured-data")
        self.captured_count = 0
        self.target_captures = 12
        self.captured_poses = []
        
        self.last_frame = None
        self.current_detection = (False, None, None)
        
        self.jogging_active = False
        self.active_jog_button = None  # Track which button is currently active
        
        # Load previous config if exists to keep count? 
        # For now, we reset like the original script or check dirs.
        if self.output_dir.exists():
            # Count existing directories
            existing = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("pose_")]
            # self.captured_count = len(existing) # Optional: resume
            pass

    def initialize(self, host):
        # Prevent double initialization
        if self._initialized:
            print("Already initialized, skipping...")
            return True
            
        # Ensure any previous camera instance is stopped
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None

        try:
            print("Initializing Camera...")
            self.camera = RealSenseCamera(lazy=True)
            # Don't get intrinsics yet if lazy
            # self.K, self.D = self.camera.get_intrinsics_matrix()
            
            print(f"Connecting to Robot at {host}...")
            self.robot = Robot(host)
            self.robot.recover_from_errors()
            self.robot.relative_dynamics_factor = 0.05 # Low dynamics for manual
            
            self.detector = CharucoDetector("config/calibration_board_parameters.yaml")
            
            # Reset output dir for fresh session as per original script behavior
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"Initialization Error: {e}")
            return False

    def cleanup(self):
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None

state = AppState()

# --- Logic Functions ---

def get_video_frame():
    """Get current video frame and update detection state. Returns frame as numpy array."""
    if not state.camera:
        return None
        
    frame = state.camera.get_frame()
    if frame is None:
        return None
        
    # Detection
    if state.detector and state.K is not None and state.D is not None:
        valid, rvec, tvec, corners = state.detector.detect(frame, state.K, state.D)
        if valid:
            cv2.drawFrameAxes(frame, state.K, state.D, rvec, tvec, 0.1)
            state.current_detection = (True, rvec, tvec)
        else:
            state.current_detection = (False, None, None)
    else:
        state.current_detection = (False, None, None)
            
    state.last_frame = frame.copy()
    return frame

def jog(axis, direction, btn_tag):
    """Start jogging motion. Called when button is pressed."""
    if not state.robot:
        print(f"Jog blocked: no robot connection")
        return

    # If already jogging, stop the current motion first
    if state.jogging_active:
        print(f"Already jogging, stopping current motion before starting new one")
        stop_jog()
    
    print(f"Starting jog: axis={axis}, direction={direction}, button={btn_tag}")
    state.jogging_active = True
    state.active_jog_button = btn_tag
    
    try:
        state.robot.recover_from_errors()
        
        linear = [0.0, 0.0, 0.0]
        angular = [0.0, 0.0, 0.0]
        v_lin = 0.02
        v_ang = 0.1
        
        if axis < 3:
            linear[axis] = v_lin * direction
        else:
            angular[axis-3] = v_ang * direction
            
        twist = Twist(linear, angular)
        motion = CartesianVelocityMotion(twist, duration=Duration(10000))
        
        # Safety: Stop if any force component exceeds 5 N (absolute)
        force_threshold = 5.0
        force_x_exceeded = Measure.FORCE_X > force_threshold
        force_x_neg_exceeded = Measure.FORCE_X < -force_threshold
        force_y_exceeded = Measure.FORCE_Y > force_threshold
        force_y_neg_exceeded = Measure.FORCE_Y < -force_threshold
        force_z_exceeded = Measure.FORCE_Z > force_threshold
        force_z_neg_exceeded = Measure.FORCE_Z < -force_threshold
        
        force_exceeded = (force_x_exceeded | force_x_neg_exceeded | 
                          force_y_exceeded | force_y_neg_exceeded |
                          force_z_exceeded | force_z_neg_exceeded)
        
        reaction = Reaction(force_exceeded, CartesianVelocityStopMotion(relative_dynamics_factor=0.05))
        motion.add_reaction(reaction)
        
        state.robot.move(motion, asynchronous=True)
        print("Jog motion started successfully")
        
    except Exception as e:
        print(f"Jog Error: {e}")
        state.jogging_active = False

def stop_jog():
    """Stop jogging motion. Called when button is released."""
    if not state.jogging_active or not state.robot:
        return
    
    print("Stopping jog motion...")
    try:
        # Execute a CartesianVelocityStopMotion to smoothly stop the robot
        stop_motion = CartesianVelocityStopMotion(relative_dynamics_factor=0.9)
        state.robot.move(stop_motion)
        print("Stop motion executed")
        state.robot.recover_from_errors()
        print("Robot errors recovered")
    except Exception as e:
        print(f"Error stopping jog: {e}")
    finally:
        state.jogging_active = False
        state.active_jog_button = None
        print("Jog state cleared")

def capture_pose():
    if not state.robot or state.last_frame is None:
        print("Cannot capture: Robot not connected or no video")
        return

    try:
        robot_state = state.robot.state
        q_remote = robot_state.q
        q = [float(q_remote[i]) for i in range(len(q_remote))]
        
        O_T_EE_affine = robot_state.O_T_EE
        O_T_EE_matrix_remote = O_T_EE_affine.matrix
        O_T_EE = []
        for r in range(4):
            row_remote = O_T_EE_matrix_remote[r]
            row = [float(row_remote[c]) for c in range(4)]
            O_T_EE.extend(row)
            
        valid, rvec, tvec = state.current_detection
        
        pose_idx = state.captured_count
        pose_dir = state.output_dir / f"pose_{pose_idx:02d}"
        pose_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(pose_dir / "image.png"), state.last_frame)
        
        data = {
            "joint_pose": q,
            "O_T_EE": O_T_EE,
            "camera_intrinsics": state.K,
            "dist_coeffs": state.D,
            "charuco_detected": valid,
        }
        
        if valid:
            data["T_cam_target_rvec"] = rvec.tolist() if hasattr(rvec, 'tolist') else rvec
            data["T_cam_target_tvec"] = tvec.tolist() if hasattr(tvec, 'tolist') else tvec
            
        with open(pose_dir / "data.json", 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
            
        state.captured_poses.append(q)
        state.captured_count += 1
        
        # Save joint poses config
        path = "config/joint_poses.yaml"
        os.makedirs("config", exist_ok=True)
        yaml_content = "joint_poses:\n"
        for pose in state.captured_poses:
            pose_str = "  - [" + ", ".join([f"{x:.4f}" for x in pose]) + "]\n"
            yaml_content += pose_str
        with open(path, 'w') as f:
            f.write(yaml_content)
            
        print(f"Captured Pose {pose_idx}!")
        
        if state.captured_count >= state.target_captures:
            print("Target captured count reached!")
            
    except Exception as e:
        print(f"Capture Error: {e}")

def go_home():
    if not state.robot: return
    try:
        state.robot.recover_from_errors()
        state.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]), asynchronous=True)
        print("Moving Home...")
    except Exception as e:
        print(f"Home Error: {e}")

# --- UI Setup ---

# Global UI state
ui_state = {
    'video_texture': None,
    'video_width': 640,
    'video_height': 360,
    'camera_init_attempted': False,
    'jog_buttons_pressed': {}  # Track which jog buttons are currently pressed
}

def update_camera_texture(frame):
    """Update the camera texture with a new frame."""
    if frame is None or ui_state['video_texture'] is None:
        return
    
    # Resize frame for display
    display_frame = cv2.resize(frame, (ui_state['video_width'], ui_state['video_height']))
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    # Flip vertically (OpenCV uses top-left origin, DPG uses bottom-left)
    rgb_frame = np.flipud(rgb_frame)
    # Normalize to [0, 1] range
    normalized = rgb_frame.astype(np.float32) / 255.0
    # Flatten to 1D array (width * height * channels)
    flat = normalized.flatten()
    
    dpg.set_value(ui_state['video_texture'], flat)

def create_ui():
    """Create the DearPyGui interface."""
    dpg.create_context()
    dpg.create_viewport(title='Franka Hand-Eye Calibration', width=1280, height=900)
    
    # Register texture for video display
    with dpg.texture_registry(show=False):
        # Create a placeholder texture (will be updated each frame)
        ui_state['video_texture'] = dpg.add_raw_texture(
            width=ui_state['video_width'],
            height=ui_state['video_height'],
            default_value=np.zeros((ui_state['video_height'], ui_state['video_width'], 3), dtype=np.float32).flatten(),
            format=dpg.mvFormat_Float_rgb
        )
    
    with dpg.window(label="Franka Hand-Eye Calibration", tag="primary_window"):
        # Header
        with dpg.group(horizontal=True):
            dpg.add_text("Franka Hand-Eye Calibration")
            dpg.add_spacing(count=1)
            status_label = dpg.add_text("Connecting...", tag="status_label")
        
        dpg.add_separator()
        
        # Main content area
        with dpg.group(horizontal=True):
            # Left Column: Video Feed
            with dpg.group():
                dpg.add_text("Camera Feed")
                dpg.add_image(ui_state['video_texture'], width=640, height=360, tag="video_image")
                detection_label = dpg.add_text("Waiting for Charuco...", tag="detection_label")
            
            dpg.add_spacing(count=1)
            
            # Right Column: Controls
            with dpg.group():
                # Capture Status
                with dpg.group():
                    dpg.add_text("Capture Status")
                    count_label = dpg.add_text(f"Captured: 0 / {state.target_captures}", tag="count_label")
                    progress_bar = dpg.add_progress_bar(default_value=0.0, tag="progress_bar", width=300)
                    dpg.add_button(label="Capture Pose", callback=capture_pose, width=300)
                
                dpg.add_spacing(count=2)
                
                # Jog Controls
                with dpg.group():
                    dpg.add_text("Jog Controls")
                    dpg.add_text("Hold to move", color=[150, 150, 150])
                    
                    # Translation tab
                    with dpg.group():
                        dpg.add_text("Translation", color=[100, 150, 255])
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('X+', 0, 1), ('X-', 0, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('Y+', 1, 1), ('Y-', 1, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('Z+', 2, 1), ('Z-', 2, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                    
                    dpg.add_spacing(count=1)
                    
                    # Rotation tab
                    with dpg.group():
                        dpg.add_text("Rotation", color=[100, 200, 150])
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('RX+', 3, 1), ('RX-', 3, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('RY+', 4, 1), ('RY-', 4, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                        with dpg.group(horizontal=True):
                            for label, axis, sign in [('RZ+', 5, 1), ('RZ-', 5, -1)]:
                                btn_tag = f"jog_btn_{axis}_{sign}"
                                dpg.add_button(label=label, tag=btn_tag, width=140, height=40)
                                ui_state['jog_buttons_pressed'][btn_tag] = False
                
                dpg.add_spacing(count=2)
                
                # Robot State
                with dpg.group():
                    dpg.add_text("Robot State")
                    pos_label = dpg.add_text("Pos: ---", tag="pos_label")
                    joints_label = dpg.add_text("Joints: ---", tag="joints_label")
                    dpg.add_button(label="GO HOME", callback=go_home, width=300)
    
    # Don't use callbacks - we'll check button state in the update loop instead
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

def update_ui():
    """Update loop called each frame."""
    # Try to initialize camera if needed
    if state.camera and not state.camera.pipeline and not ui_state['camera_init_attempted']:
        ui_state['camera_init_attempted'] = True
        if state.camera.initialize():
            state.K, state.D = state.camera.get_intrinsics_matrix()
    
    # Update video frame
    frame = get_video_frame()
    if frame is not None:
        update_camera_texture(frame)
    
    # Update detection status
    is_det, _, _ = state.current_detection
    detection_text = "Charuco DETECTED" if is_det else "Charuco NOT Detected"
    detection_color = [0, 255, 0] if is_det else [255, 0, 0]
    dpg.set_value("detection_label", detection_text)
    dpg.configure_item("detection_label", color=detection_color)
    
    # Update capture count
    dpg.set_value("count_label", f"Captured: {state.captured_count} / {state.target_captures}")
    dpg.set_value("progress_bar", state.captured_count / state.target_captures)
    
    # Update robot state
    if state.robot:
        try:
            s = state.robot.state
            O_T_EE = s.O_T_EE
            trans = O_T_EE.translation
            dpg.set_value("pos_label", f"Pos: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
            
            q = s.q
            q_str = ", ".join([f"{x:.2f}" for x in q])
            dpg.set_value("joints_label", f"Joints: [{q_str}]")
            
            dpg.set_value("status_label", "Connected")
            dpg.configure_item("status_label", color=[0, 255, 0])
        except:
            dpg.set_value("status_label", "Connection Lost")
            dpg.configure_item("status_label", color=[255, 0, 0])
    else:
        dpg.set_value("status_label", "Disconnected")
        dpg.configure_item("status_label", color=[255, 0, 0])
    
    # Handle jog button states - check each button to see if it's being held down
    any_button_pressed = False
    
    for btn_tag in ui_state['jog_buttons_pressed'].keys():
        # Check if this button is hovered and left mouse button is down
        is_hovered = dpg.is_item_hovered(btn_tag)
        is_mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        button_held = is_hovered and is_mouse_down
        
        if button_held:
            any_button_pressed = True
            # Button is being held down
            if not ui_state['jog_buttons_pressed'][btn_tag]:
                # Button just pressed - start jogging
                parts = btn_tag.split('_')
                axis = int(parts[2])
                sign = int(parts[3])
                ui_state['jog_buttons_pressed'][btn_tag] = True
                jog(axis, sign, btn_tag)
        else:
            # Button is not being held
            if ui_state['jog_buttons_pressed'][btn_tag]:
                # Button was just released
                ui_state['jog_buttons_pressed'][btn_tag] = False
    
    # If no button is pressed and we're still jogging, stop
    if state.jogging_active and not any_button_pressed:
        print("No jog button held - stopping jog")
        stop_jog()

# --- Startup ---

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP")
    args, _ = parser.parse_known_args()
    
    if state.initialize(args.host):
        print("System Initialized.")
    else:
        print("System Initialization Failed.")

    create_ui()
    
    # Main render loop
    while dpg.is_dearpygui_running():
        update_ui()
        dpg.render_dearpygui_frame()
    
    # Cleanup
    state.cleanup()
    dpg.destroy_context()

if __name__ in {"__main__", "__mp_main__"}:
    run()

