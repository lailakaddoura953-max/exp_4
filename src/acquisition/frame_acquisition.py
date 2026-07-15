"""
Frame Acquisition Module

Handles synchronization of frames from multiple cameras with bounded buffering.

Properties validated:
- Property 1: Frame Synchronization (50ms tolerance)
- Property 10: Frame Buffer Bounds (max size enforced)
- Property 11: Complete Synchronized Batch (all 4 cameras present)
- Property 17: Buffer FIFO Eviction (oldest frames removed first)
"""

import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Union
from collections import deque
from dataclasses import dataclass, field

from src.models.core import SynchronizedFrameBatch


@dataclass
class CameraSource:
    """
    Camera source configuration and state
    
    Supports file sources, RTSP streams, USB cameras, and mock sources
    """
    camera_id: int
    source: Union[str, int, List[np.ndarray]]  # File path, RTSP URL, device ID, or mock frames
    resolution: tuple  # (width, height)
    fps: int
    
    # Internal state
    capture: Optional[cv2.VideoCapture] = None
    is_mock: bool = False
    mock_frames: List[np.ndarray] = field(default_factory=list)
    mock_frame_index: int = 0
    is_connected: bool = False
    last_frame_time: int = 0  # Microseconds
    
    def __post_init__(self):
        """Validate camera source configuration"""
        if not 0 <= self.camera_id <= 3:
            raise ValueError(f"camera_id must be in range [0, 3], got {self.camera_id}")
        
        # Check if this is a mock source (list of frames)
        if isinstance(self.source, list):
            self.is_mock = True
            self.mock_frames = self.source
            self.is_connected = True


class FrameAcquisitionModule:
    """
    Frame acquisition and synchronization module
    
    Manages frame capture from 4 cameras with synchronization within 50ms tolerance.
    Implements bounded frame buffers with FIFO eviction.
    """
    
    def __init__(
        self,
        sync_tolerance_ms: float = 50.0,
        buffer_size_per_camera: int = 100,
        auto_reconnect: bool = True
    ):
        """
        Initialize frame acquisition module
        
        Args:
            sync_tolerance_ms: Maximum time difference between frames in a batch (Property 1)
            buffer_size_per_camera: Maximum frames to buffer per camera (Property 10)
            auto_reconnect: Whether to automatically reconnect disconnected cameras
        """
        if sync_tolerance_ms <= 0:
            raise ValueError(f"sync_tolerance_ms must be positive, got {sync_tolerance_ms}")
        
        if buffer_size_per_camera <= 0:
            raise ValueError(f"buffer_size_per_camera must be positive, got {buffer_size_per_camera}")
        
        self.sync_tolerance_ms = sync_tolerance_ms
        self.buffer_size_per_camera = buffer_size_per_camera
        self.auto_reconnect = auto_reconnect
        
        # Camera sources
        self.cameras: Dict[int, CameraSource] = {}
        
        # Frame buffers (bounded FIFO queues)
        self.frame_buffers: Dict[int, deque] = {}
        
        # Sequence number for batches
        self.sequence_number = 0
        
        # Statistics
        self.frames_acquired = {i: 0 for i in range(4)}
        self.sync_failures = 0
        self.buffer_overflows = {i: 0 for i in range(4)}
    
    def initialize_cameras(self, camera_sources: List[CameraSource]):
        """
        Initialize camera connections
        
        Args:
            camera_sources: List of CameraSource objects for all cameras
        
        Raises:
            ValueError: If not exactly 4 cameras provided
        """
        if len(camera_sources) != 4:
            raise ValueError(f"Must provide exactly 4 cameras, got {len(camera_sources)}")
        
        # Verify camera IDs are 0-3
        camera_ids = {cam.camera_id for cam in camera_sources}
        if camera_ids != {0, 1, 2, 3}:
            raise ValueError(f"Camera IDs must be [0, 1, 2, 3], got {sorted(camera_ids)}")
        
        for camera_source in camera_sources:
            self.cameras[camera_source.camera_id] = camera_source
            self.frame_buffers[camera_source.camera_id] = deque(maxlen=self.buffer_size_per_camera)
            
            # Connect to camera if not mock
            if not camera_source.is_mock:
                self._connect_camera(camera_source)
    
    def _connect_camera(self, camera_source: CameraSource) -> bool:
        """
        Connect to a camera source
        
        Args:
            camera_source: CameraSource to connect
        
        Returns:
            True if connection successful
        """
        try:
            if isinstance(camera_source.source, int):
                # USB camera (device ID)
                camera_source.capture = cv2.VideoCapture(camera_source.source)
            elif isinstance(camera_source.source, str):
                # File or RTSP stream
                camera_source.capture = cv2.VideoCapture(camera_source.source)
            else:
                return False
            
            if camera_source.capture and camera_source.capture.isOpened():
                # Set resolution
                camera_source.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_source.resolution[0])
                camera_source.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_source.resolution[1])
                camera_source.is_connected = True
                return True
            else:
                camera_source.is_connected = False
                return False
        except Exception:
            camera_source.is_connected = False
            return False
    
    def _disconnect_camera(self, camera_id: int):
        """Disconnect a camera and release resources"""
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if camera.capture:
                camera.capture.release()
                camera.capture = None
            camera.is_connected = False
    
    def acquire_frame(self, camera_id: int) -> Optional[tuple]:
        """
        Acquire a single frame from a camera
        
        Args:
            camera_id: Camera to acquire from
        
        Returns:
            Tuple of (frame, timestamp_us) or None if acquisition failed
        """
        if camera_id not in self.cameras:
            return None
        
        camera = self.cameras[camera_id]
        
        # Handle mock source
        if camera.is_mock:
            if camera.mock_frame_index >= len(camera.mock_frames):
                return None  # No more frames
            
            frame = camera.mock_frames[camera.mock_frame_index]
            camera.mock_frame_index += 1
            
            # Generate mock timestamp (simulate 30 fps = 33333 microseconds per frame)
            timestamp_us = camera.last_frame_time + 33333
            camera.last_frame_time = timestamp_us
            
            return (frame, timestamp_us)
        
        # Handle real camera source
        if not camera.is_connected:
            if self.auto_reconnect:
                self._connect_camera(camera)
            if not camera.is_connected:
                return None
        
        try:
            ret, frame = camera.capture.read()
            if ret and frame is not None:
                timestamp_us = int(time.time() * 1_000_000)  # Current time in microseconds
                camera.last_frame_time = timestamp_us
                return (frame, timestamp_us)
            else:
                # Frame acquisition failed
                camera.is_connected = False
                return None
        except Exception:
            camera.is_connected = False
            return None
    
    def buffer_frame(self, camera_id: int, frame: np.ndarray, timestamp_us: int):
        """
        Add frame to camera's buffer
        
        Implements bounded FIFO buffer (Property 10, Property 17).
        When buffer is full, oldest frame is automatically evicted.
        
        Args:
            camera_id: Camera ID
            frame: Frame data
            timestamp_us: Timestamp in microseconds
        """
        if camera_id not in self.frame_buffers:
            return
        
        buffer = self.frame_buffers[camera_id]
        
        # Check if buffer is full (will evict oldest)
        if len(buffer) >= self.buffer_size_per_camera:
            self.buffer_overflows[camera_id] += 1
        
        # Add to buffer (deque with maxlen handles FIFO eviction)
        buffer.append((frame, timestamp_us))
        self.frames_acquired[camera_id] += 1
    
    def acquire_and_buffer_all(self):
        """
        Acquire frames from all cameras and add to buffers
        
        Attempts to acquire from all connected cameras and buffers frames.
        """
        for camera_id in range(4):
            result = self.acquire_frame(camera_id)
            if result:
                frame, timestamp_us = result
                self.buffer_frame(camera_id, frame, timestamp_us)
    
    def get_synchronized_frames(self) -> Optional[SynchronizedFrameBatch]:
        """
        Get synchronized frame batch from all cameras
        
        Finds frames from all 4 cameras within sync_tolerance_ms of each other.
        Validates Property 1 (Frame Synchronization) and Property 11 (Complete Batch).
        
        Returns:
            SynchronizedFrameBatch if sync successful, None otherwise
        """
        # Check if all cameras have at least one frame
        for camera_id in range(4):
            if len(self.frame_buffers[camera_id]) == 0:
                return None
        
        # Get oldest frame from each camera
        oldest_frames = {}
        oldest_timestamps = {}
        
        for camera_id in range(4):
            buffer = self.frame_buffers[camera_id]
            if len(buffer) > 0:
                frame, timestamp = buffer[0]  # Peek at oldest
                oldest_frames[camera_id] = frame
                oldest_timestamps[camera_id] = timestamp
        
        # Check if we have all 4 cameras
        if len(oldest_timestamps) != 4:
            return None
        
        # Check synchronization (Property 1)
        timestamps_list = list(oldest_timestamps.values())
        min_ts = min(timestamps_list)
        max_ts = max(timestamps_list)
        time_diff_ms = (max_ts - min_ts) / 1000.0
        
        if time_diff_ms <= self.sync_tolerance_ms:
            # Frames are synchronized! Remove from buffers and create batch
            frames = {}
            timestamps = {}
            
            for camera_id in range(4):
                frame, timestamp = self.frame_buffers[camera_id].popleft()
                frames[camera_id] = frame
                timestamps[camera_id] = timestamp
            
            # Create synchronized batch
            batch = SynchronizedFrameBatch(
                frames=frames,
                timestamps=timestamps,
                sequence_number=self.sequence_number,
                is_complete=True  # Property 11: All 4 cameras present
            )
            
            self.sequence_number += 1
            return batch
        else:
            # Not synchronized - remove oldest frame and try again
            # Find camera with oldest timestamp
            oldest_camera_id = min(oldest_timestamps.keys(), key=lambda k: oldest_timestamps[k])
            self.frame_buffers[oldest_camera_id].popleft()
            self.sync_failures += 1
            return None
    
    def get_camera_status(self, camera_id: int) -> Dict:
        """
        Get status information for a camera
        
        Args:
            camera_id: Camera to query
        
        Returns:
            Dictionary with status information
        """
        if camera_id not in self.cameras:
            return {'connected': False, 'buffer_size': 0}
        
        camera = self.cameras[camera_id]
        buffer = self.frame_buffers[camera_id]
        
        return {
            'camera_id': camera_id,
            'connected': camera.is_connected,
            'buffer_size': len(buffer),
            'buffer_capacity': self.buffer_size_per_camera,
            'frames_acquired': self.frames_acquired[camera_id],
            'buffer_overflows': self.buffer_overflows[camera_id]
        }
    
    def get_system_status(self) -> Dict:
        """
        Get overall system status
        
        Returns:
            Dictionary with system-wide statistics
        """
        return {
            'cameras_connected': sum(1 for cam in self.cameras.values() if cam.is_connected),
            'total_cameras': len(self.cameras),
            'sync_failures': self.sync_failures,
            'sequence_number': self.sequence_number,
            'camera_statuses': {i: self.get_camera_status(i) for i in range(4)}
        }
    
    def shutdown(self):
        """Shutdown module and release all camera resources"""
        for camera_id in range(4):
            self._disconnect_camera(camera_id)
        
        # Clear buffers
        for buffer in self.frame_buffers.values():
            buffer.clear()


def create_mock_frame(
    width: int = 640,
    height: int = 480,
    color: tuple = (100, 150, 200),
    text: str = ""
) -> np.ndarray:
    """
    Create a mock frame for testing
    
    Args:
        width: Frame width
        height: Frame height
        color: BGR color tuple
        text: Optional text to overlay
    
    Returns:
        Mock frame as numpy array
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    
    if text:
        cv2.putText(frame, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2)
    
    return frame
