"""
Hazard Detection System

Multi-camera hazard detection for industrial yard environments.
Detects human zone violations, container misalignment/door state issues,
and unsafe container orientations using YOLOv12 object detection.

Binary hazard classification: each detection is either is_hazard=True
(confirmed hazard, alert dispatched) or is_hazard=False (not a hazard, logged only).
No severity levels are used.
"""

from hazard_detection.models import (
    BBox,
    Detection,
    DiagnosticMetadata,
    FrameSequence,
    HazardEvent,
    ZonePolygon,
    FrameSamplerConfig,
    YOLOConfig,
    HumanDetectorConfig,
    ContainerAnalyzerConfig,
    AlertDispatcherConfig,
    CameraSwitcherConfig,
    PipelineConfig,
    VALID_HAZARD_TYPES,
)
from hazard_detection.config import (
    ConfigurationManager,
    ConfigurationError,
    DOCUMENTED_DEFAULTS,
    load_config,
)

# Pipeline components are imported lazily to avoid pulling in hardware
# dependencies (FrameAcquisitionModule, OpticalFlowAnalyzer, etc.) when
# only the data models or training pipeline are needed.
# Import them directly where required, e.g.:
#   from hazard_detection.frame_sampler import FrameSampler

__all__ = [
    "BBox",
    "Detection",
    "DiagnosticMetadata",
    "FrameSequence",
    "HazardEvent",
    "ZonePolygon",
    "FrameSamplerConfig",
    "YOLOConfig",
    "HumanDetectorConfig",
    "ContainerAnalyzerConfig",
    "AlertDispatcherConfig",
    "CameraSwitcherConfig",
    "PipelineConfig",
    "VALID_HAZARD_TYPES",
    "ConfigurationManager",
    "ConfigurationError",
    "DOCUMENTED_DEFAULTS",
    "load_config",
]
