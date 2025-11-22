import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
import shutil
import base64
from pathlib import Path
from nicegui import ui, app

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
            
            self.detector = CharucoDetector("charuco/calibration_board_parameters.yaml")
            
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

def get_video_frame_base64():
    if not state.camera:
        return None
        
    frame = state.camera.get_frame()
    if frame is None:
        return None
        
    # Detection
    if state.detector:
        valid, rvec, tvec, corners = state.detector.detect(frame, state.K, state.D)
        if valid:
            cv2.drawFrameAxes(frame, state.K, state.D, rvec, tvec, 0.1)
            state.current_detection = (True, rvec, tvec)
        else:
            state.current_detection = (False, None, None)
            
    state.last_frame = frame.copy()
    
    # Resize for web display (save bandwidth)
    display_frame = cv2.resize(frame, (640, 360))
    _, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'

async def jog(axis, direction):
    if state.jogging_active or not state.robot:
        return

    state.jogging_active = True
    
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
        motion = CartesianVelocityMotion(twist, duration=Duration(5000))
        
        # Safety
        force_norm_cond = (Measure.FORCE_X * Measure.FORCE_X + 
                           Measure.FORCE_Y * Measure.FORCE_Y + 
                           Measure.FORCE_Z * Measure.FORCE_Z) > 100.0
        reaction = Reaction(force_norm_cond, CartesianVelocityStopMotion(relative_dynamics_factor=0.05))
        motion.add_reaction(reaction)
        
        state.robot.move(motion, asynchronous=True)
        
    except Exception as e:
        ui.notify(f"Jog Error: {e}", type='negative')
        state.jogging_active = False

async def stop_jog():
    if not state.jogging_active or not state.robot:
        return
    try:
        state.robot.stop()
        state.robot.recover_from_errors()
    except:
        pass
    finally:
        state.jogging_active = False

def capture_pose():
    if not state.robot or state.last_frame is None:
        ui.notify("Cannot capture: Robot not connected or no video", type='warning')
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
            
        ui.notify(f"Captured Pose {pose_idx}!", type='positive')
        
        if state.captured_count >= state.target_captures:
            ui.notify("Target captured count reached!", type='positive', close_button="OK")
            
    except Exception as e:
        ui.notify(f"Capture Error: {e}", type='negative')

def go_home():
    if not state.robot: return
    try:
        state.robot.recover_from_errors()
        state.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]), asynchronous=True)
        ui.notify("Moving Home...", type='info')
    except Exception as e:
        ui.notify(f"Home Error: {e}", type='negative')

# --- UI Setup ---

@ui.page('/')
def main_page():
    ui.colors(primary='#5898d4', secondary='#26a69a', accent='#9c27b0', positive='#21ba45', negative='#c10015', info='#31ccec', warning='#f2c037')

    with ui.header().classes(replace='row items-center') as header:
        ui.icon('precision_manufacturing', size='32px').classes('q-mr-sm text-white')
        ui.label('Franka Hand-Eye Calibration').classes('text-h6 text-white')
        ui.space()
        status_label = ui.label('Connecting...').classes('text-white')

    with ui.row().classes('w-full q-pa-md'):
        # Left Column: Video
        with ui.card().classes('w-2/3'):
            ui.label('Camera Feed').classes('text-h6 q-mb-md')
            video_image = ui.image().props('fit=scale-down').style('width: 100%; height: 500px; background-color: #000')
            
            with ui.row().classes('items-center q-mt-md'):
                ui.icon('grid_on').classes('q-mr-sm')
                detection_label = ui.label('Waiting for Charuco...').classes('text-subtitle1')
                ui.space()
                
                def toggle_camera():
                    if state.camera and state.camera.pipeline:
                        state.camera.stop()
                        cam_btn.props('icon=videocam color=positive')
                        cam_btn.text = 'Start Camera'
                    elif state.camera:
                        # It will try to init in the loop
                        pass 
                        
                cam_btn = ui.button('Start Camera', on_click=lambda: None).props('icon=videocam_off color=negative')
                # We'll handle button state in loop actually or make it simpler
                
        # Right Column: Controls
        with ui.column().classes('w-1/3 q-pl-md'):
            
            # Capture Status
            with ui.card().classes('w-full q-mb-md'):
                ui.label('Capture Status').classes('text-h6')
                progress = ui.linear_progress(value=0).classes('q-my-md')
                count_label = ui.label(f'Captured: 0 / {state.target_captures}').classes('text-subtitle1 text-bold')
                
                ui.button('Capture Pose', on_click=capture_pose, icon='camera').props('color=positive size=lg').classes('w-full')

            # Jog Controls
            with ui.card().classes('w-full q-mb-md'):
                ui.label('Jog Controls').classes('text-h6 q-mb-sm')
                ui.label('Hold to move').classes('text-caption text-grey q-mb-md')
                
                with ui.tabs().classes('w-full') as tabs:
                    t_trans = ui.tab('Translation')
                    t_rot = ui.tab('Rotation')

                with ui.tab_panels(tabs, value=t_trans).classes('w-full'):
                    with ui.tab_panel(t_trans):
                        with ui.grid(columns=2).classes('w-full gap-2'):
                            for label, axis, sign in [('X+', 0, 1), ('X-', 0, -1), 
                                                    ('Y+', 1, 1), ('Y-', 1, -1), 
                                                    ('Z+', 2, 1), ('Z-', 2, -1)]:
                                btn = ui.button(label).props('push')
                                # Bind press/release
                                btn.on('mousedown', lambda _, a=axis, s=sign: jog(a, s))
                                btn.on('mouseup', stop_jog)
                                btn.on('mouseleave', stop_jog) # Safety if mouse leaves button while down

                    with ui.tab_panel(t_rot):
                        with ui.grid(columns=2).classes('w-full gap-2'):
                            for label, axis, sign in [('RX+', 3, 1), ('RX-', 3, -1), 
                                                    ('RY+', 4, 1), ('RY-', 4, -1), 
                                                    ('RZ+', 5, 1), ('RZ-', 5, -1)]:
                                btn = ui.button(label).props('push color=secondary')
                                btn.on('mousedown', lambda _, a=axis, s=sign: jog(a, s))
                                btn.on('mouseup', stop_jog)
                                btn.on('mouseleave', stop_jog)

            # Robot State
            with ui.card().classes('w-full'):
                ui.label('Robot State').classes('text-h6')
                pos_label = ui.label('Pos: ---').classes('font-mono text-xs q-mb-xs')
                joints_label = ui.label('Joints: ---').classes('font-mono text-xs q-mb-md')
                ui.button('GO HOME', on_click=go_home, icon='home').props('color=warning outline').classes('w-full')

    # Update Loop
    camera_init_attempted = False
    
    async def update_loop():
        nonlocal camera_init_attempted
        
        # Update Video
        # Try to initialize camera if needed (only once)
        if state.camera and not state.camera.pipeline and not camera_init_attempted:
            camera_init_attempted = True
            if state.camera.initialize():
                # Once initialized, get intrinsics
                state.K, state.D = state.camera.get_intrinsics_matrix()
        
        src = get_video_frame_base64()
        if src:
            video_image.set_source(src)
            
        # Update Detection Status
        is_det, _, _ = state.current_detection
        detection_label.text = "Charuco DETECTED" if is_det else "Charuco NOT Detected"
        detection_label.classes(remove='text-red text-green', add='text-green' if is_det else 'text-red')
        
        # Update Counts
        count_label.text = f'Captured: {state.captured_count} / {state.target_captures}'
        progress.value = state.captured_count / state.target_captures
        
        # Update Robot State
        if state.robot:
            try:
                s = state.robot.state
                
                # Pos
                O_T_EE = s.O_T_EE
                trans = O_T_EE.translation
                pos_label.text = f"Pos: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]"
                
                # Joints
                q = s.q
                q_str = ", ".join([f"{x:.2f}" for x in q])
                joints_label.text = f"Joints: [{q_str}]"
                
                status_label.text = "Connected"
                status_label.classes(remove='text-red', add='text-green')
            except:
                status_label.text = "Connection Lost"
                status_label.classes(remove='text-green', add='text-red')
        else:
            status_label.text = "Disconnected"
            status_label.classes(remove='text-green', add='text-red')

    ui.timer(0.05, update_loop)

# --- Startup ---

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP")
    args, _ = parser.parse_known_args()
    
    if state.initialize(args.host):
        print("System Initialized.")
    else:
        print("System Initialization Failed.")

    app.on_shutdown(state.cleanup)
    
    # Native mode disabled due to missing system dependencies
    # ui.run(title="Franka Calibration", favicon="🤖", native=True, window_size=(1280, 900))
    ui.run(title="Franka Calibration", favicon="🤖")

if __name__ in {"__main__", "__mp_main__"}:
    run()

