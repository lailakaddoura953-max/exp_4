"""
Data models for the Hazard Detection System.

This module defines all data structures used throughout the hazard detection
pipeline, including validation logic to ensure data integrity.

Binary hazard classification: each detection is either is_hazard=True
(confirmed hazard, alert dispatched) or is_hazard=False (not a hazard, logged only).
No severity levels are used.

Properties validated:
- Property 13: Unique event identifiers (UUID generation)
- Property 14: Hazard event structural completeness (mandatory field validation)

Requirements covered:
- 8.1: Hazard event structure with all mandatory fields
- 8.5: Unique event identifier generation
- 8.6: Rejection of events missing mandatory fields
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import uuid

import numpy as np


# Valid hazard types emitted by the system
VALID_HAZARD_TYPES = {
    "zone_violation",
    "ppe_violation",
    "container_misalignment",
    "container_door_open",
    "container_flipped",
    "container_dangling",
}


@dataclass
class BBox:
    """
    Normalized bounding box with coordinates relative to frame dimensions.

    All values are in the range [0.0, 1.0].
    Format follows YOLO convention: x_center, y_center, width, height.
    """

    x_center: float  # [0.0, 1.0] relative to frame width
    y_center: float  # [0.0, 1.0] relative to frame height
    width: float  # [0.0, 1.0]
    height: float  # [0.0, 1.0]

    def __post_init__(self):
        """Validate that all coordinates are in [0.0, 1.0]."""
        for attr_name in ("x_center", "y_center", "width", "height"):
            value = getattr(self, attr_name)
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"BBox.{attr_name} must be a number, got {type(value).__name__}"
                )
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"BBox.{attr_name} must be in [0.0, 1.0], got {value}"
                )

    @property
    def center(self) -> Tuple[float, float]:
        """Return the center point as (x, y)."""
        return (self.x_center, self.y_center)

    @property
    def aspect_ratio(self) -> float:
        """Return height-to-width ratio. Returns 0.0 if width is 0."""
        if self.width == 0.0:
            return 0.0
        return self.height / self.width


@dataclass
class Detection:
    """
    A single object detection from the YOLO model.

    Produced by the YOLO_Detector for each detected object in a frame.
    """

    bbox: BBox
    class_label: str  # One of the 17 Roboflow classes
    confidence: float  # [0.0, 1.0]

    def __post_init__(self):
        """Validate detection fields."""
        if not isinstance(self.bbox, BBox):
            raise TypeError(
                f"Detection.bbox must be a BBox instance, got {type(self.bbox).__name__}"
            )
        if not isinstance(self.class_label, str) or not self.class_label.strip():
            raise ValueError("Detection.class_label must be a non-empty string")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(
                f"Detection.confidence must be a number, got {type(self.confidence).__name__}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Detection.confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass
class DiagnosticMetadata:
    """
    Diagnostic metadata attached to each HazardEvent.

    Contains additional context for debugging and downstream processing.
    """

    frame_index: int  # Index within Frame_Sequence (0-based)
    detection_class: str  # Raw YOLO class label
    frames_detected: int  # Number of frames the hazard appeared in
    flow_consistency_score: Optional[float] = None  # From OpticalFlowAnalyzer

    def __post_init__(self):
        """Validate metadata fields."""
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError(
                f"DiagnosticMetadata.frame_index must be a non-negative integer, "
                f"got {self.frame_index}"
            )
        if not isinstance(self.detection_class, str) or not self.detection_class.strip():
            raise ValueError(
                "DiagnosticMetadata.detection_class must be a non-empty string"
            )
        if not isinstance(self.frames_detected, int) or self.frames_detected < 0:
            raise ValueError(
                f"DiagnosticMetadata.frames_detected must be a non-negative integer, "
                f"got {self.frames_detected}"
            )
        if self.flow_consistency_score is not None:
            if not isinstance(self.flow_consistency_score, (int, float)):
                raise TypeError(
                    "DiagnosticMetadata.flow_consistency_score must be a number or None"
                )


@dataclass
class FrameSequence:
    """
    An ordered collection of 5-8 frames captured from the same camera feed.

    Represents a single sampling window for one camera.
    """

    frames: List[np.ndarray]  # 5-8 frames as numpy arrays (BGR images)
    camera_id: str  # Camera identifier from config
    timestamps: List[float]  # Capture timestamps per frame

    def __post_init__(self):
        """Validate frame sequence."""
        if not isinstance(self.frames, list):
            raise TypeError("FrameSequence.frames must be a list")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("FrameSequence.camera_id must be a non-empty string")
        if not isinstance(self.timestamps, list):
            raise TypeError("FrameSequence.timestamps must be a list")
        if len(self.frames) != len(self.timestamps):
            raise ValueError(
                f"FrameSequence.frames length ({len(self.frames)}) must match "
                f"timestamps length ({len(self.timestamps)})"
            )

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the sequence."""
        return len(self.frames)


@dataclass
class HazardEvent:
    """
    A structured record emitted when a detection is processed.

    Contains all information needed for downstream alert dispatch and logging.
    Implements validation per Requirement 8.6 to reject events missing mandatory fields.

    Binary hazard classification:
    - is_hazard=True: confirmed hazard, alert dispatched
    - is_hazard=False: not a hazard, logged only

    Mandatory fields: event_id, hazard_type, camera_id, timestamp, is_hazard,
                      confidence, bbox, metadata
    """

    event_id: str  # UUID, unique across system lifetime
    hazard_type: str  # zone_violation | ppe_violation | container_misalignment |
    #                    container_door_open | container_flipped | container_dangling
    camera_id: str  # Camera identifier from config
    timestamp: str  # ISO 8601 UTC (e.g., "2024-01-15T10:30:00Z")
    is_hazard: bool  # True = confirmed hazard (dispatch alert), False = logged only
    confidence: float  # [0.0, 1.0]
    bbox: BBox  # Normalized bounding box
    metadata: DiagnosticMetadata  # Frame index, detection class, frames detected, optional flow score

    def validate(self) -> bool:
        """
        Validate that all mandatory fields are present and well-formed.

        Returns True if the event is valid. Raises ValueError if any
        mandatory field is missing or invalid.

        Requirement 8.6: Reject events missing mandatory fields.
        """
        # Check event_id
        if not self.event_id or not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("HazardEvent validation failed: event_id is missing or empty")

        # Check hazard_type
        if not self.hazard_type or not isinstance(self.hazard_type, str) or not self.hazard_type.strip():
            raise ValueError("HazardEvent validation failed: hazard_type is missing or empty")
        if self.hazard_type not in VALID_HAZARD_TYPES:
            raise ValueError(
                f"HazardEvent validation failed: hazard_type '{self.hazard_type}' "
                f"is not a recognized type. Must be one of {VALID_HAZARD_TYPES}"
            )

        # Check camera_id
        if not self.camera_id or not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("HazardEvent validation failed: camera_id is missing or empty")

        # Check timestamp (ISO 8601 UTC format)
        if not self.timestamp or not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("HazardEvent validation failed: timestamp is missing or empty")
        # Basic ISO 8601 format validation
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise ValueError(
                f"HazardEvent validation failed: timestamp '{self.timestamp}' "
                f"is not valid ISO 8601 format"
            )

        # Check is_hazard
        if not isinstance(self.is_hazard, bool):
            raise ValueError(
                f"HazardEvent validation failed: is_hazard must be a bool, "
                f"got {type(self.is_hazard).__name__}"
            )

        # Check confidence
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(
                "HazardEvent validation failed: confidence must be a number"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"HazardEvent validation failed: confidence must be in [0.0, 1.0], "
                f"got {self.confidence}"
            )

        # Check bbox
        if not isinstance(self.bbox, BBox):
            raise ValueError(
                "HazardEvent validation failed: bbox must be a BBox instance"
            )

        # Check metadata
        if not isinstance(self.metadata, DiagnosticMetadata):
            raise ValueError(
                "HazardEvent validation failed: metadata must be a DiagnosticMetadata instance"
            )

        return True

    @staticmethod
    def generate_event_id() -> str:
        """
        Generate a unique event identifier using UUID4.

        Requirement 8.5: No two events share the same identifier across
        the system's operational lifetime.
        """
        return str(uuid.uuid4())

    @staticmethod
    def generate_timestamp() -> str:
        """Generate an ISO 8601 UTC timestamp for the current time."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ZonePolygon:
    """
    A polygonal zone region defined by normalized coordinates.

    Used by ZoneMap to define no-people, operation, and dropoff zones.
    """

    vertices: List[Tuple[float, float]]  # Normalized coordinates, min 3 vertices
    zone_type: str  # "no_people" | "operation" | "dropoff"
    camera_id: str  # Associated camera identifier

    # Valid zone types matching Roboflow annotations
    VALID_ZONE_TYPES = {"no_people", "operation", "dropoff"}

    def __post_init__(self):
        """Validate zone polygon."""
        # Validate zone type
        if self.zone_type not in self.VALID_ZONE_TYPES:
            raise ValueError(
                f"ZonePolygon.zone_type must be one of {self.VALID_ZONE_TYPES}, "
                f"got '{self.zone_type}'"
            )

        # Validate camera_id
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("ZonePolygon.camera_id must be a non-empty string")

        # Validate minimum vertex count
        if not isinstance(self.vertices, list) or len(self.vertices) < 3:
            raise ValueError(
                f"ZonePolygon must have at least 3 vertices, got {len(self.vertices) if isinstance(self.vertices, list) else 0}"
            )

        # Validate each vertex is a tuple of normalized coordinates
        for i, vertex in enumerate(self.vertices):
            if not isinstance(vertex, (tuple, list)) or len(vertex) != 2:
                raise ValueError(
                    f"ZonePolygon vertex {i} must be a (x, y) pair, got {vertex}"
                )
            x, y = vertex
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                raise ValueError(
                    f"ZonePolygon vertex {i} coordinates must be in [0.0, 1.0], "
                    f"got ({x}, {y})"
                )


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class FrameSamplerConfig:
    """Configuration for the Frame_Sampler component."""

    frame_count: int = 6  # Number of frames to capture per camera (5-8)
    timeout_ms: int = 2000  # Timeout in milliseconds for feed availability
    max_retries: int = 3  # Max retries per frame acquisition failure

    def __post_init__(self):
        """Validate frame sampler configuration."""
        if not isinstance(self.frame_count, int) or not (5 <= self.frame_count <= 8):
            raise ValueError(
                f"FrameSamplerConfig.frame_count must be an integer in [5, 8], "
                f"got {self.frame_count}"
            )
        if not isinstance(self.timeout_ms, int) or self.timeout_ms <= 0:
            raise ValueError(
                f"FrameSamplerConfig.timeout_ms must be a positive integer, "
                f"got {self.timeout_ms}"
            )
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError(
                f"FrameSamplerConfig.max_retries must be a non-negative integer, "
                f"got {self.max_retries}"
            )


@dataclass
class YOLOConfig:
    """Configuration for the YOLO_Detector component."""

    checkpoint_path: str = "checkpoints/yolov12_best.pt"
    device: str = "cuda"  # "cuda" or "cpu"
    input_resolution: int = 640  # Square resolution (320-750)
    confidence_threshold: float = 0.5  # Minimum confidence for detections
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self):
        """Validate YOLO configuration."""
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path.strip():
            raise ValueError("YOLOConfig.checkpoint_path must be a non-empty string")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(
                f"YOLOConfig.device must be 'cuda' or 'cpu', got '{self.device}'"
            )
        if not isinstance(self.input_resolution, int) or not (320 <= self.input_resolution <= 750):
            raise ValueError(
                f"YOLOConfig.input_resolution must be an integer in [320, 750], "
                f"got {self.input_resolution}"
            )
        if not isinstance(self.confidence_threshold, (int, float)) or not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"YOLOConfig.confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )


@dataclass
class HumanDetectorConfig:
    """Configuration for the Human_Detector component."""

    confidence_threshold: float = 0.5  # Minimum confidence for human detections

    def __post_init__(self):
        """Validate human detector configuration."""
        if not isinstance(self.confidence_threshold, (int, float)) or not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"HumanDetectorConfig.confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )


@dataclass
class ContainerAnalyzerConfig:
    """Configuration for the Container_Analyzer component."""

    confidence_threshold: float = 0.5  # Minimum confidence for container detections
    flipped_aspect_ratio_threshold: float = 1.5  # Height/width ratio for flipped detection
    safe_overlap_threshold: float = 0.3  # IoA threshold for safe crane operations
    ground_level_threshold: float = 0.4  # Vertical midpoint threshold for dangling
    motion_threshold: float = 0.7  # Flow variance threshold for motion indicator
    iou_threshold: float = 0.5  # IoU threshold for overlapping classification

    def __post_init__(self):
        """Validate container analyzer configuration."""
        if not isinstance(self.confidence_threshold, (int, float)) or not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"ContainerAnalyzerConfig.confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold}"
            )
        if not isinstance(self.flipped_aspect_ratio_threshold, (int, float)) or self.flipped_aspect_ratio_threshold <= 0:
            raise ValueError(
                f"ContainerAnalyzerConfig.flipped_aspect_ratio_threshold must be positive, "
                f"got {self.flipped_aspect_ratio_threshold}"
            )
        if not isinstance(self.safe_overlap_threshold, (int, float)) or not (0.0 <= self.safe_overlap_threshold <= 1.0):
            raise ValueError(
                f"ContainerAnalyzerConfig.safe_overlap_threshold must be in [0.0, 1.0], "
                f"got {self.safe_overlap_threshold}"
            )
        if not isinstance(self.ground_level_threshold, (int, float)) or not (0.0 <= self.ground_level_threshold <= 1.0):
            raise ValueError(
                f"ContainerAnalyzerConfig.ground_level_threshold must be in [0.0, 1.0], "
                f"got {self.ground_level_threshold}"
            )
        if not isinstance(self.motion_threshold, (int, float)) or self.motion_threshold < 0:
            raise ValueError(
                f"ContainerAnalyzerConfig.motion_threshold must be non-negative, "
                f"got {self.motion_threshold}"
            )
        if not isinstance(self.iou_threshold, (int, float)) or not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"ContainerAnalyzerConfig.iou_threshold must be in [0.0, 1.0], "
                f"got {self.iou_threshold}"
            )


@dataclass
class AlertDispatcherConfig:
    """Configuration for the Alert_Dispatcher component."""

    rate_limit_seconds: int = 60  # Rate limit window in seconds (10-300)
    channels: List[str] = field(default_factory=lambda: ["email", "dashboard"])

    def __post_init__(self):
        """Validate alert dispatcher configuration."""
        if not isinstance(self.rate_limit_seconds, int) or not (10 <= self.rate_limit_seconds <= 300):
            raise ValueError(
                f"AlertDispatcherConfig.rate_limit_seconds must be an integer in [10, 300], "
                f"got {self.rate_limit_seconds}"
            )
        if not isinstance(self.channels, list) or len(self.channels) == 0:
            raise ValueError(
                "AlertDispatcherConfig.channels must be a non-empty list"
            )


@dataclass
class CameraSwitcherConfig:
    """Configuration for the Camera_Switcher stub component."""

    camera_list: List[str] = field(default_factory=list)  # Up to 16 camera entries
    connection_types: Dict[str, str] = field(default_factory=dict)  # Placeholders
    transition_params: Dict[str, str] = field(default_factory=dict)  # Placeholders

    def __post_init__(self):
        """Validate camera switcher configuration."""
        if not isinstance(self.camera_list, list):
            raise TypeError("CameraSwitcherConfig.camera_list must be a list")
        if len(self.camera_list) > 16:
            raise ValueError(
                f"CameraSwitcherConfig.camera_list supports at most 16 entries, "
                f"got {len(self.camera_list)}"
            )


@dataclass
class PipelineConfig:
    """
    Top-level configuration for the Detection Pipeline.

    Aggregates all component configurations and pipeline-level parameters.
    """

    camera_sequence: List[str] = field(default_factory=list)  # Ordered camera processing list
    per_camera_timeout_seconds: int = 30  # Per-camera processing timeout
    frame_sampler: FrameSamplerConfig = field(default_factory=FrameSamplerConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    human_detector: HumanDetectorConfig = field(default_factory=HumanDetectorConfig)
    container_analyzer: ContainerAnalyzerConfig = field(default_factory=ContainerAnalyzerConfig)
    alert_dispatcher: AlertDispatcherConfig = field(default_factory=AlertDispatcherConfig)
    camera_switcher: CameraSwitcherConfig = field(default_factory=CameraSwitcherConfig)

    def __post_init__(self):
        """Validate pipeline configuration."""
        if not isinstance(self.camera_sequence, list):
            raise TypeError("PipelineConfig.camera_sequence must be a list")
        if not isinstance(self.per_camera_timeout_seconds, int) or self.per_camera_timeout_seconds <= 0:
            raise ValueError(
                f"PipelineConfig.per_camera_timeout_seconds must be a positive integer, "
                f"got {self.per_camera_timeout_seconds}"
            )

