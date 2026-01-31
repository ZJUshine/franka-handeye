# Repository Guidelines

## Project Structure & Module Organization
- `franka_handeye/` core library: `camera.py`, `detector.py`, `robot.py`, `calibration.py`.
- `franka-handeye-app.py` GUI entry (DearPyGui).
- `scripts/` headless CLI helpers (`capture_data.py`, `compute_calibration.py`, `verify_calibration.py`).
- `config/` board and robot pose configuration (`calibration_board_parameters.yaml`, `default_joint_poses.yaml`, `joint_poses.yaml`, `charuco_board_5x7.png`).
- `data/` sample captures and outputs (`captured-data/pose_XX/data.json`, `hand-eye-calibration-output/`), plus media.

## Build, Test, and Development Commands
Use Python 3.10+.
```
pip install -r requirements.txt
pip install -e .
```
Run the GUI:
```
python franka-handeye-app.py --host 172.16.0.2
```
Console entry points (after editable install):
```
franka-handeye-app --host 172.16.0.2
franka-capture --host 172.16.0.2 --output data/captured-data
franka-calibrate --data data/captured-data
franka-verify --host 172.16.0.2
```

## Coding Style & Naming Conventions
- Indentation: 4 spaces; keep lines at 100 chars (Black config).
- Formatting/linting: `python -m black .` and `python -m ruff check .`.
- Naming: `snake_case` for modules/functions, `CamelCase` for classes, `UPPER_CASE` for constants.
- Data captures follow `pose_XX` directory naming.

## Testing Guidelines
- `pytest` is available in dev deps, but tests are not yet committed.
- Add new tests under `tests/` with `test_*.py`.
- Hardware-dependent tests should be clearly marked/guarded so CI or offline runs can skip them.

## Commit & Pull Request Guidelines
- History shows short, plain messages (e.g., "readme update"). Keep commit subjects concise; no strict convention.
- PRs should include: summary, how to run/reproduce, hardware assumptions (robot model, RealSense), and config changes.
- Include screenshots/GIFs for GUI changes.
- Avoid committing large raw data unless needed; if you add captures or calibration outputs, explain why.

## Configuration & Hardware Notes
- Always update `config/calibration_board_parameters.yaml` to match your printed board measurements.
- Robot control requires a running `franky-remote` server on the robot PC.

## Agent Notes
- For automated agents, also review `CLAUDE.md` for architecture and command examples.

Use chinese to answer.
