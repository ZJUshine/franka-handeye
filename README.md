# Franka Hand-Eye Calibration

A complete GUI application for hand-eye calibration of the Franka robot using a RealSense camera and Charuco board.

![Calibration Verification](data/media/franka-handeye-verification.gif)

## Installation

### 1. Server Side (Robot PC)
The robot control runs on a real-time machine. We recommend using [ServoBox](https://www.servobox.dev/) to deploy the [franky-remote](https://github.com/kvasios/franky-remote) server.

Install and run the package for your robot generation:
```bash
# For Franka Emika Panda (Gen1)
servobox pkg-install franky-remote-gen1
servobox run franky-remote-gen1

# OR for Franka Research 3 (FR3)
servobox pkg-install franky-remote-fr3
servobox run franky-remote-fr3
```

### 2. Client Side (Workstation)
Clone and install dependencies:
```bash
git clone https://github.com/kvasios/franka-handeye.git
cd franka-handeye

# Create environment (optional but recommended)
micromamba create -n franka-handeye python=3.10
micromamba activate franka-handeye

# Install
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run the GUI application:
```bash
python franka-handeye-app.py --host <ROBOT_IP>
```
*Default host is `172.16.0.2`*

### Workflow
1.  **Capture**:
    *   **Auto Run**: Uses 12 recommended preset poses to automatically capture the dataset.
    *   **Manual**: Alternatively, use **Jog Controls** to move to your own poses and click **Capture**.
    *   Ensure ChArUco board is detected (green status) for each pose.
2.  **Calibrate**:
    *   Click **Run Calibration**.
    *   Check the result and 3D plot.
3.  **Verify**:
    *   **Check Frames**: Visualizes the computed frame alignment.
    *   **Visit Corners**: Robot physically traces the board corners to verify accuracy.

## Configuration
*   **Board**: Update `config/calibration_board_parameters.yaml` with your physical board measurements.
*   **Poses**: `config/joint_poses.yaml` contains the 12 poses used for Auto Capture.

## Headless / Scripts
Individual scripts are available in `scripts/` for CI or headless operation:
*   `scripts/capture_data.py`: Capture dataset.
*   `scripts/compute_calibration.py`: Compute from existing data.
*   `scripts/verify_calibration.py`: Run physical verification.
