import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def load_captured_data(data_dir):
    data_dir = Path(data_dir)
    pose_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("pose_")])
    
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
            
        # Extract Robot Pose (T_gripper2base)
        # O_T_EE is typically column-major in Franka. 
        # If it's a list of 16, we reshape(4,4).T (to get row-major standard matrix)
        # If it's already 4x4, we check orthogonality.
        
        O_T_EE = np.array(data["O_T_EE"])
        
        if O_T_EE.shape == (16,):
            # Column-major flat array -> Reshape to 4x4 -> Transpose to get standard Row-major
            T_g2b = O_T_EE.reshape(4, 4).T
        elif O_T_EE.shape == (4, 4):
            # Assume it's already in correct format, but check if it needs transpose (Franka raw is column major)
            # We'll assume the capture script or franky wrapper provides a usable 4x4. 
            # If it looks like [ [R, t], [0, 1] ], it's good.
            # If the last row is [0,0,0,1], it is row-major storage of the matrix.
            if np.allclose(O_T_EE[3, :], [0, 0, 0, 1]):
                T_g2b = O_T_EE
            elif np.allclose(O_T_EE[:, 3], [0, 0, 0, 1]):
                 # It's transposed (column vectors are rows)
                 T_g2b = O_T_EE.T
            else:
                 T_g2b = O_T_EE # Fallback
        else:
             print(f"Skipping {p_dir.name}: Invalid O_T_EE shape {O_T_EE.shape}")
             continue

        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3]
        
        # Extract Target Pose (T_target2cam)
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

def main():
    parser = argparse.ArgumentParser(description="Compute Hand-Eye Calibration")
    parser.add_argument("--data", default="data/captured-data", help="Directory with captured data")
    parser.add_argument("--method", default="daniilidis", help="Calibration method (daniilidis, tsai, etc.)")
    args = parser.parse_args()

    R_g2b, t_g2b, R_t2c, t_t2c = load_captured_data(args.data)

    if len(R_g2b) < 3:
        print("Error: Need at least 3 poses for calibration.")
        return

    print("Running calibration...")
    
    method_dict = {
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andrei": cv2.CALIB_HAND_EYE_ANDREI
    }
    
    method = method_dict.get(args.method.lower(), cv2.CALIB_HAND_EYE_DANIILIDIS)
    
    # cv2.calibrateHandEye inputs:
    # R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
    # It solves for X where AX = XB (Eye-in-hand) or AX = ZB (Eye-to-hand)?
    # OpenCV documentation:
    # Solves AX = XB. 
    # A = T_gripper2base_i+1 * inv(T_gripper2base_i) (relative motion of gripper)
    # B = T_target2cam_i+1 * inv(T_target2cam_i) (relative motion of target in cam frame)
    # Wait, calibrateHandEye takes absolute poses as lists?
    # Yes, "R_gripper2base: vector of rotation matrices..."
    # So we pass the lists directly.
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=method
    )

    print("\nCalibration Result (T_cam_gripper):")
    print("Rotation Matrix (R_cam2gripper):")
    print(R_cam2gripper)
    print("\nTranslation Vector (t_cam2gripper):")
    print(t_cam2gripper.flatten())
    
    # Create Homogeneous Matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    print("\nHomogeneous Matrix T_cam_gripper:")
    print(T_cam2gripper)
    
    # Convert to Quaternion
    quat = R.from_matrix(R_cam2gripper).as_quat() # [x, y, z, w]
    print("\nQuaternion [x, y, z, w]:")
    print(quat)
    
    # Save result
    result = {
        "T_cam_gripper": T_cam2gripper.tolist(),
        "xyz": t_cam2gripper.flatten().tolist(),
        "quaternion_xyzw": quat.tolist()
    }
    
    output_dir = Path("data/hand-eye-calibration-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_result.json"

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved result to {output_path}")

if __name__ == "__main__":
    main()

