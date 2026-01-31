"""
Hand-eye calibration computation and verification.

Supports two calibration modes:
- EYE_IN_HAND: Camera mounted on robot gripper, target fixed in world
- EYE_TO_HAND: Camera fixed in world, target mounted on robot gripper
"""

import json
import numpy as np
import cv2
from enum import Enum
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class CalibrationMode(str, Enum):
    """Calibration mode for hand-eye calibration."""
    EYE_IN_HAND = "eye_in_hand"
    EYE_TO_HAND = "eye_to_hand"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_captured_data(data_dir: str | Path) -> tuple[list, list, list, list]:
    """
    Load captured calibration data from directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing pose_XX subdirectories with data.json files.
    
    Returns
    -------
    tuple
        (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        Lists of rotation matrices and translation vectors.
    """
    data_dir = Path(data_dir)
    pose_dirs = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and d.name.startswith("pose_")
    ])
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    valid_poses = 0
    
    for p_dir in pose_dirs:
        json_path = p_dir / "data.json"
        if not json_path.exists():
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if not data.get("charuco_detected", False):
            print(f"Skipping {p_dir.name}: Charuco not detected.")
            continue
            
        # Extract Robot Pose (T_gripper2base = O_T_EE from Franka)
        O_T_EE = np.array(data["O_T_EE"])
        T_g2b = O_T_EE.reshape(4, 4)
        
        # Verify it's a valid transformation matrix
        if not np.allclose(T_g2b[3, :], [0, 0, 0, 1]):
            print(f"Warning: {p_dir.name} has invalid homogeneous row")
            continue
        
        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3]
        
        # Extract Target Pose from OpenCV solvePnP
        rvec = np.array(data["T_cam_target_rvec"]).flatten()
        tvec = np.array(data["T_cam_target_tvec"]).flatten()
        
        R_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c = tvec
        
        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)
        
        valid_poses += 1
    
    print(f"Loaded {valid_poses} valid poses for calibration.")
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def compute_hand_eye_calibration(
    R_gripper2base: list,
    t_gripper2base: list,
    R_target2cam: list,
    t_target2cam: list,
    method: int = cv2.CALIB_HAND_EYE_DANIILIDIS,
    mode: CalibrationMode = CalibrationMode.EYE_IN_HAND
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute hand-eye calibration using OpenCV.

    Parameters
    ----------
    R_gripper2base : list
        List of 3x3 rotation matrices (gripper to base).
    t_gripper2base : list
        List of translation vectors (gripper to base).
    R_target2cam : list
        List of 3x3 rotation matrices (target to camera).
    t_target2cam : list
        List of translation vectors (target to camera).
    method : int
        OpenCV calibration method. Default is DANIILIDIS.
    mode : CalibrationMode
        EYE_IN_HAND: camera on gripper, solves T_cam2gripper.
        EYE_TO_HAND: camera fixed, solves T_cam2base.

    Returns
    -------
    tuple
        EYE_IN_HAND: (R_cam2gripper, t_cam2gripper)
        EYE_TO_HAND: (R_cam2base, t_cam2base)
    """
    if len(R_gripper2base) < 3:
        raise ValueError("Need at least 3 poses for calibration.")

    mode = CalibrationMode(mode)

    if mode == CalibrationMode.EYE_TO_HAND:
        # For eye-to-hand, OpenCV expects R_base2gripper and t_base2gripper
        # (the inverse of T_gripper2base)
        R_base2gripper = [Rg.T for Rg in R_gripper2base]
        t_base2gripper = [
            -Rg.T @ tg.flatten() for Rg, tg in zip(R_gripper2base, t_gripper2base)
        ]

        R_result, t_result = cv2.calibrateHandEye(
            R_gripper2base=R_base2gripper,
            t_gripper2base=t_base2gripper,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=method
        )
    else:
        R_result, t_result = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=method
        )

    return R_result, t_result


def compute_consistency_metrics(
    R_gripper2base: list,
    t_gripper2base: list,
    R_target2cam: list,
    t_target2cam: list,
    T_calibration: np.ndarray,
    mode: CalibrationMode = CalibrationMode.EYE_IN_HAND
) -> tuple[float, float]:
    """
    Compute consistency/repeatability metrics for calibration.

    For EYE_IN_HAND: the target (charuco board) position in base frame should
    be constant across all poses if calibration is accurate.
    For EYE_TO_HAND: the target position in gripper frame should be constant.

    Parameters
    ----------
    R_gripper2base : list
        List of gripper-to-base rotation matrices.
    t_gripper2base : list
        List of gripper-to-base translations.
    R_target2cam : list
        List of target-to-camera rotation matrices.
    t_target2cam : list
        List of target-to-camera translations.
    T_calibration : np.ndarray
        4x4 transformation matrix.
        EYE_IN_HAND: T_cam2gripper
        EYE_TO_HAND: T_cam2base
    mode : CalibrationMode
        Calibration mode.

    Returns
    -------
    tuple
        (mean_error, std_error) position errors in meters.
    """
    mode = CalibrationMode(mode)
    target_positions = []

    for i in range(len(R_gripper2base)):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_gripper2base[i]
        T_g2b[:3, 3] = t_gripper2base[i].flatten()

        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_target2cam[i]
        T_t2c[:3, 3] = t_target2cam[i].flatten()

        if mode == CalibrationMode.EYE_TO_HAND:
            # T_cam2base is fixed, check T_target2gripper consistency
            # T_target2base = T_cam2base @ T_target2cam
            T_target2base = T_calibration @ T_t2c
            # T_target2gripper = T_base2gripper @ T_target2base
            T_base2gripper = np.linalg.inv(T_g2b)
            T_target2gripper = T_base2gripper @ T_target2base
            target_positions.append(T_target2gripper[:3, 3])
        else:
            # T_target2base = T_gripper2base * T_cam2gripper * T_target2cam
            T_t2b = T_g2b @ T_calibration @ T_t2c
            target_positions.append(T_t2b[:3, 3])

    target_positions = np.array(target_positions)
    mean_pos = np.mean(target_positions, axis=0)
    pos_errors = np.linalg.norm(target_positions - mean_pos, axis=1)

    return float(np.mean(pos_errors)), float(np.std(pos_errors))


def save_calibration_result(
    output_path: str | Path,
    R_result: np.ndarray,
    t_result: np.ndarray,
    consistency_error_mean: float = None,
    consistency_error_std: float = None,
    mode: CalibrationMode = CalibrationMode.EYE_IN_HAND
) -> dict:
    """
    Save calibration result to JSON file.

    Parameters
    ----------
    output_path : str or Path
        Output file path.
    R_result : np.ndarray
        3x3 rotation matrix.
    t_result : np.ndarray
        Translation vector.
    consistency_error_mean : float, optional
        Mean consistency error.
    consistency_error_std : float, optional
        Std of consistency error.
    mode : CalibrationMode
        Calibration mode.

    Returns
    -------
    dict
        The saved result dictionary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = CalibrationMode(mode)

    # Create homogeneous matrix
    T_result = np.eye(4)
    T_result[:3, :3] = R_result
    T_result[:3, 3] = t_result.flatten()

    # Convert to quaternion
    quat = R.from_matrix(R_result).as_quat()  # [x, y, z, w]

    # Use mode-specific key for the transformation matrix
    if mode == CalibrationMode.EYE_TO_HAND:
        matrix_key = "T_cam_base"
    else:
        matrix_key = "T_cam_gripper"

    result = {
        "mode": mode.value,
        matrix_key: T_result.tolist(),
        "xyz": t_result.flatten().tolist(),
        "quaternion_xyzw": quat.tolist(),
    }

    if consistency_error_mean is not None:
        result["consistency_error_mean"] = consistency_error_mean
    if consistency_error_std is not None:
        result["consistency_error_std"] = consistency_error_std

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    return result


def load_calibration_result(calib_path: str | Path) -> tuple[np.ndarray, CalibrationMode]:
    """
    Load calibration result from JSON file.

    Parameters
    ----------
    calib_path : str or Path
        Path to calibration JSON file.

    Returns
    -------
    tuple
        (T_result, mode) where T_result is a 4x4 transformation matrix and
        mode is the CalibrationMode. For legacy files without a mode field,
        defaults to EYE_IN_HAND.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    mode_str = calib.get('mode', CalibrationMode.EYE_IN_HAND.value)
    mode = CalibrationMode(mode_str)

    if mode == CalibrationMode.EYE_TO_HAND:
        T = np.array(calib['T_cam_base'])
    else:
        T = np.array(calib['T_cam_gripper'])

    return T, mode


def compute_alignment_pose(
    T_gripper_base_current: np.ndarray,
    T_calibration: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    offset_distance: float = 0.1,
    target_point_in_board: list = None,
    mode: CalibrationMode = CalibrationMode.EYE_IN_HAND
) -> np.ndarray:
    """
    Compute desired gripper pose to align with charuco board.

    Parameters
    ----------
    T_gripper_base_current : np.ndarray
        Current gripper pose in base frame (4x4).
    T_calibration : np.ndarray
        EYE_IN_HAND: Camera to gripper transform (4x4).
        EYE_TO_HAND: Camera to base transform (4x4).
    rvec : np.ndarray
        Rotation vector of target in camera frame.
    tvec : np.ndarray
        Translation vector of target in camera frame.
    offset_distance : float
        Distance to maintain from board (meters).
    target_point_in_board : list
        [x, y, z] point in board frame to align with. Default: origin.
    mode : CalibrationMode
        Calibration mode.

    Returns
    -------
    np.ndarray
        Desired gripper pose in base frame (4x4).
    """
    if target_point_in_board is None:
        target_point_in_board = [0, 0, 0]

    mode = CalibrationMode(mode)

    # Convert rvec, tvec to transformation matrix
    R_target_cam, _ = cv2.Rodrigues(rvec)
    t_target_cam = tvec.flatten()

    T_target_cam = np.eye(4)
    T_target_cam[:3, :3] = R_target_cam
    T_target_cam[:3, 3] = t_target_cam

    if mode == CalibrationMode.EYE_TO_HAND:
        # Camera is fixed in world; T_calibration = T_cam2base
        T_cam_base = T_calibration
    else:
        # Camera is on gripper; T_calibration = T_cam2gripper
        T_cam_base = T_gripper_base_current @ T_calibration

    # Target (charuco board) pose in base frame
    T_target_base = T_cam_base @ T_target_cam

    # Desired gripper pose: aligned with charuco board, offset along Z axis
    T_gripper_target_desired = np.eye(4)
    T_gripper_target_desired[:3, 3] = [
        target_point_in_board[0],
        target_point_in_board[1],
        target_point_in_board[2] - offset_distance
    ]

    # Desired gripper pose in base frame
    T_gripper_base_desired = T_target_base @ T_gripper_target_desired

    return T_gripper_base_desired

