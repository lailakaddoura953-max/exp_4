"""
Core data models shared by src/cv/, src/acquisition/, and src/alerting/.

This module was missing from the checkout (an orphaned import target
referenced by src/cv/flow_analyzer.py, src/acquisition/frame_acquisition.py,
and src/alerting/alert_system.py, but never actually committed to the
repo). Reconstructed here from those three modules' own field usage and
docstrings, so their existing code — which was otherwise complete and
correct — has something real to import.

FlowResult and SynchronizedFrameBatch are actively constructed and
consumed by their respective modules; their fields and validation below
match exactly what those call sites pass in and read back.

MisalignmentEvent and Severity are referenced only as type hints and via
attribute access (event.severity, event.camera_id, event.timestamp,
event.event_id) in src/alerting/alert_system.py — no call site in this
repo actually constructs a MisalignmentEvent, so their shape here is a
reasonable minimal reconstruction covering every attribute alert_system.py
touches, not a guess at unobserved fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict

import numpy as np


@dataclass
class FlowResult:
    """
    Result of a dense optical flow computation between two frames.

    Constructed by src.cv.flow_analyzer.OpticalFlowAnalyzer.compute_flow()
    and filter_outliers(); consumed by segment_dynamic_regions() and
    get_flow_consistency_score().

    Validates (per src/cv/flow_analyzer.py's docstrings):
    - Property 8: flow_vectors' spatial dimensions match frame_shape
    - Property 4: confidence values are within [0.0, 1.0]
    """

    flow_vectors: np.ndarray  # (H, W, 2) — per-pixel (dx, dy) flow
    confidence: np.ndarray  # (H, W) — per-pixel confidence in [0.0, 1.0]
    mean_magnitude: float
    mean_direction: float
    frame_shape: tuple  # (height, width)

    def __post_init__(self):
        """Validate Property 8 (spatial dims) and Property 4 (confidence bounds)."""
        expected_h, expected_w = self.frame_shape
        flow_h, flow_w = self.flow_vectors.shape[0], self.flow_vectors.shape[1]
        if (flow_h, flow_w) != (expected_h, expected_w):
            raise ValueError(
                f"FlowResult.flow_vectors spatial dims {(flow_h, flow_w)} "
                f"must match frame_shape {self.frame_shape} (Property 8)"
            )

        conf_min = float(np.min(self.confidence))
        conf_max = float(np.max(self.confidence))
        if conf_min < 0.0 or conf_max > 1.0:
            raise ValueError(
                f"FlowResult.confidence must be within [0.0, 1.0], "
                f"got range [{conf_min}, {conf_max}] (Property 4)"
            )


@dataclass
class SynchronizedFrameBatch:
    """
    A batch of frames from multiple cameras, synchronized within a
    configured time tolerance.

    Constructed by
    src.acquisition.frame_acquisition.FrameAcquisitionModule.get_synchronized_frames()
    once all cameras have a frame within sync_tolerance_ms of each other.

    Validates (per src/acquisition/frame_acquisition.py's docstrings):
    - Property 11: is_complete implies all 4 cameras are present
    """

    frames: Dict[int, np.ndarray]  # camera_id -> frame
    timestamps: Dict[int, int]  # camera_id -> timestamp in microseconds
    sequence_number: int
    is_complete: bool = False

    def __post_init__(self):
        """Validate Property 11: a complete batch has all 4 cameras."""
        if self.is_complete and set(self.frames.keys()) != {0, 1, 2, 3}:
            raise ValueError(
                f"SynchronizedFrameBatch marked is_complete=True must contain "
                f"all 4 cameras (0-3), got {sorted(self.frames.keys())} (Property 11)"
            )
        if set(self.frames.keys()) != set(self.timestamps.keys()):
            raise ValueError(
                "SynchronizedFrameBatch.frames and .timestamps must have the "
                "same set of camera_id keys"
            )


class Severity(Enum):
    """
    Misalignment event severity levels.

    Referenced by src.alerting.alert_system.AlertSystem.process_event(),
    which only dispatches alerts for HIGH and CRITICAL severity (Property 14).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MisalignmentEvent:
    """
    A detected camera/container misalignment event.

    Fields cover every attribute accessed by
    src.alerting.alert_system.AlertSystem (event.event_id, event.camera_id,
    event.severity, event.timestamp) — no call site in this repo
    constructs one, so this is a minimal, defensively-shaped
    reconstruction rather than a guess at fields never actually observed
    in use.
    """

    event_id: str
    camera_id: int
    severity: Severity
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
