# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Franka Hand-Eye Calibration - GUI application for hand-eye calibration of a Franka robot using an Intel RealSense camera and ChArUco board. Computes the camera-to-gripper transformation matrix.

## Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Running the Application
```bash
# GUI application (main)
python franka-handeye-app.py --host <ROBOT_IP>

# Console entry points (after pip install -e .)
franka-handeye-app --host 172.16.0.2
franka-capture --host 172.16.0.2
franka-calibrate --data data/captured-data
franka-verify --host 172.16.0.2
```

### Headless Scripts
```bash
python scripts/capture_data.py --host 172.16.0.2 --output data/captured-data
python scripts/compute_calibration.py --data data/captured-data
python scripts/verify_calibration.py --host 172.16.0.2 --offset 0.06
```

## Architecture

### Core Package (franka_handeye/)
- **camera.py**: `RealSenseCamera` class - Intel RealSense interface with lazy initialization, frame capture, and intrinsics extraction
- **detector.py**: `CharucoDetector` class - ChArUco board detection using OpenCV ArUco DICT_6X6_250, returns rotation/translation vectors via solvePnP
- **robot.py**: `RobotController` class - Franka robot control via franky-remote server, handles jogging, joint poses, and end-effector transforms
- **calibration.py**: Hand-eye calibration computation (OpenCV Daniilidis method), data loading from pose_XX/data.json files, consistency metrics, result serialization

### GUI Application (franka-handeye-app.py)
DearPyGui-based interface with three main tabs:
1. **Capture**: Live camera feed with ChArUco overlay, auto-run through 12 preset poses or manual jog controls
2. **Calibration**: Compute and display calibration results with 3D visualization
3. **Verify**: Physical verification by tracing board corners with robot

Uses `AppState` class for state management and multi-threading for hardware operations.

### Configuration (config/)
- **calibration_board_parameters.yaml**: Board dimensions (5x7, square=0.034m, marker=0.028m) - must match printed board
- **default_joint_poses.yaml**: Default 12+ robot joint poses for auto-capture
- **joint_poses.yaml**: Custom poses (auto-created/updated by manual capture)

### Data Flow
1. Capture: Robot moves to poses → camera captures → ChArUco detected → data saved to `data/captured-data/pose_XX/data.json`
2. Calibrate: Load all poses → compute hand-eye transform → save to `data/hand-eye-calibration-output/`
3. Verify: Load calibration → robot traces corners → validate accuracy

## Hardware Requirements
- Intel RealSense camera
- Franka Emika Panda or Franka Research 3
- franky-remote server running on robot PC
