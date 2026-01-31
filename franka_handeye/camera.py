"""
RealSense camera interface for hand-eye calibration.
"""

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class RealSenseCamera:
    """
    RealSense camera wrapper with lazy initialization support.

    Parameters
    ----------
    lazy : bool
        If True, delays pipeline initialization until initialize() is called.
        Useful for GUI applications where camera might not be immediately needed.
    resolution : tuple
        Camera resolution (width, height). Default is (1280, 720).
    fps : int
        Frame rate. Default is 30.
    serial_number : str or None
        Serial number of the camera to use. If None, uses the first available.
    """

    def __init__(self, lazy: bool = False, resolution: tuple = (1280, 720), fps: int = 30, serial_number: str | None = None):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed")

        self.pipeline = None
        self.config = None
        self.profile = None
        self.intrinsics = None
        self.resolution = resolution
        self.fps = fps
        self.serial_number = serial_number

        if not lazy:
            self.initialize()

    @staticmethod
    def list_devices() -> list[dict]:
        """
        List all connected RealSense devices.

        Returns
        -------
        list of dict
            Each dict has 'name' and 'serial_number' keys.
        """
        if not REALSENSE_AVAILABLE:
            return []
        ctx = rs.context()
        devices = []
        for dev in ctx.query_devices():
            devices.append({
                'name': dev.get_info(rs.camera_info.name),
                'serial_number': dev.get_info(rs.camera_info.serial_number),
            })
        return devices

    def initialize(self) -> bool:
        """
        Initialize the camera pipeline.

        Returns
        -------
        bool
            True if initialization succeeded, False otherwise.
        """
        if self.pipeline is not None:
            return True

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            if self.serial_number:
                self.config.enable_device(self.serial_number)
            self.config.enable_stream(
                rs.stream.color,
                self.resolution[0],
                self.resolution[1],
                rs.format.bgr8,
                self.fps
            )
            self.profile = self.pipeline.start(self.config)
            self.color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
            return True
        except RuntimeError as e:
            self.pipeline = None
            # Handle transient errors gracefully - GUI will retry
            if "Device or resource busy" in str(e) or "Couldn't resolve requests" in str(e):
                return False
            raise
    
    @property
    def is_initialized(self) -> bool:
        """Check if camera is initialized."""
        return self.pipeline is not None
    
    def get_frame(self) -> np.ndarray | None:
        """
        Capture a single color frame.
        
        Returns
        -------
        np.ndarray or None
            BGR image array, or None if capture failed.
        """
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
    
    def get_intrinsics_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get camera intrinsics matrix and distortion coefficients.
        
        Returns
        -------
        tuple
            (K, dist_coeffs) where K is 3x3 intrinsics matrix and 
            dist_coeffs is the distortion coefficient array.
        """
        if not self.intrinsics:
            return np.eye(3), np.zeros(5)
            
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(self.intrinsics.coeffs)
        return K, dist_coeffs
    
    def stop(self):
        """Stop the camera pipeline."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
    
    def __del__(self):
        self.stop()
    
    def __enter__(self):
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

