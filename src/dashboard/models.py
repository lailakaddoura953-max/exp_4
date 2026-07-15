"""
Data models for the Yard Hazard Inference Dashboard.

Defines the dashboard-specific data structures used throughout the inference
pipeline, REST API, and HazardStore.  These models complement (and in some
cases wrap) the existing data models in src/hazard_detection/models.py without
modifying them.

Requirements covered:
- 1.6:  InferenceEngine config validation at construction time
- 16.1: HazardResult dataclass fields
- 16.2: HazardResult confidence clamped to [0.0, 1.0]
- 16.3: hazard_reason is empty string when is_hazard=False
- 16.4: HazardResult serialisation uses snake_case with nested bbox
- 17.1: InferenceEngineConfig fields and defaults
- 17.3: ValueError at construction when confidence_threshold is out of range
         or checkpoint_path is empty/None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# BBox is defined in the existing hazard-detection models — do NOT redefine it.
# src/ is on sys.path (installed as editable package), so we import directly.
from hazard_detection.models import BBox


# ---------------------------------------------------------------------------
# HazardResult
# ---------------------------------------------------------------------------


@dataclass
class HazardResult:
    """
    A record produced by the Inference_Engine for one YOLO detection.

    One HazardResult is emitted per detection that meets or exceeds the
    configured Confidence_Threshold.  The confidence value is always clamped
    to [0.0, 1.0] regardless of what the underlying model emits.

    Requirements: 16.1, 16.2, 16.3
    """

    class_label: str
    confidence: float       # clamped to [0.0, 1.0]
    bbox: BBox              # from src/hazard_detection/models.py
    is_hazard: bool
    hazard_reason: str      # "" when is_hazard is False (Requirement 16.3)
    camera_id: str

    def __post_init__(self) -> None:
        """Clamp confidence to [0.0, 1.0] at construction time."""
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-compatible dict using snake_case keys.

        The ``bbox`` field is represented as a nested object with the four
        YOLO-format fields: x_center, y_center, width, height.

        Requirement 16.4
        """
        return {
            "class_label": self.class_label,
            "confidence": self.confidence,
            "bbox": {
                "x_center": self.bbox.x_center,
                "y_center": self.bbox.y_center,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "is_hazard": self.is_hazard,
            "hazard_reason": self.hazard_reason,
            "camera_id": self.camera_id,
        }


# ---------------------------------------------------------------------------
# LocationContext
# ---------------------------------------------------------------------------


@dataclass
class LocationContext:
    """
    Stub location metadata attached to every HazardEvent.

    All fields are stubs until the real camera registry and yard map are
    integrated.  Use ``from_camera_id()`` to derive a stub location from a
    camera identifier string.

    Fields
    ------
    facility : str
        Always "Railyard" for the current deployment scope.
    berth : str
        One of "Berth 403", "Berth 405", "Berth 406" (stub; determined by
        camera_id mapping).
        # TODO: expand berth list from yard operations team
    crane : str
        "Crane 01" through "Crane 14" (14 cranes total; stub assignment based
        on camera_id mapping).
        # TODO: replace with real crane-camera mapping table
    camera : str
        "Camera 01" through "Camera 15" (15 cameras; stub number derived from
        camera_id string).
        # TODO: replace with real camera registry
    landmark : str
        Reserved for additional yard landmarks; empty for now.
        # TODO: populate once yard map is finalised
        # Reference map images in maps/ folder
        # Interactive map: https://3d.stowlog.com/apmt-pier-400/main
    """

    facility: str = "Railyard"
    berth: str = ""
    crane: str = ""
    camera: str = ""
    landmark: str = ""

    # ------------------------------------------------------------------
    # Stub camera_id → location mapping
    # ------------------------------------------------------------------
    #
    # Source: design.md §3.2a — "Stub camera_id → location mapping" table
    # and the CAMERA_PINS constant in terminal_map.js.
    #
    # | camera_id     | berth     | crane     | camera     |
    # |---------------|-----------|-----------|------------|
    # | cam_stub_01   | Berth 403 | Crane 01  | Camera 01  |
    # | cam_stub_02   | Berth 403 | Crane 02  | Camera 02  |
    # | cam_stub_03   | Berth 405 | Crane 07  | Camera 03  |
    # | cam_stub_04   | Berth 405 | Crane 08  | Camera 04  |
    # | cam_stub_05   | Berth 406 | Crane 13  | Camera 05  |
    # | cam_stub_06   | Berth 405 | Crane 06  | Camera 06  |
    # | cam_stub_07   | Berth 405 | Crane 07  | Camera 07  |
    # | cam_stub_08   | Berth 405 | Crane 08  | Camera 08  |
    # | cam_stub_09   | Berth 405 | Crane 09  | Camera 09  |
    # | cam_stub_10   | Berth 406 | Crane 10  | Camera 10  |
    # | cam_stub_11   | Berth 406 | Crane 11  | Camera 11  |
    # | cam_stub_12   | Berth 406 | Crane 12  | Camera 12  |
    # | cam_stub_13   | Berth 406 | Crane 13  | Camera 13  |
    # | cam_stub_14   | Berth 406 | Crane 14  | Camera 14  |
    # | cam_stub_15   | (gate)    |           | Camera 15  |
    #
    # Unrecognised camera_id → berth="", crane="", camera=camera_id

    @classmethod
    def from_camera_id(cls, camera_id: str) -> "LocationContext":
        """
        Derive stub location from a camera_id string.

        Stub logic: consult the hard-coded mapping table defined in the design
        document (design.md §3.2a).  Unrecognised camera IDs fall back to
        berth="", crane="", camera=camera_id.

        Replace this method body entirely when the real camera registry is
        available.
        # TODO: replace stub mapping with lookup against camera registry config
        """
        # --- Stub mapping table -------------------------------------------
        # Keys are the known cam_stub_XX identifiers; values are
        # (berth, crane, camera) triples.
        # TODO: replace stub mapping with lookup against camera registry config
        _table: dict[str, tuple[str, str, str]] = {
            "cam_stub_01": ("Berth 403", "Crane 01", "Camera 01"),
            "cam_stub_02": ("Berth 403", "Crane 02", "Camera 02"),
            "cam_stub_03": ("Berth 405", "Crane 07", "Camera 03"),
            "cam_stub_04": ("Berth 405", "Crane 08", "Camera 04"),
            "cam_stub_05": ("Berth 406", "Crane 13", "Camera 05"),
            "cam_stub_06": ("Berth 405", "Crane 06", "Camera 06"),
            "cam_stub_07": ("Berth 405", "Crane 07", "Camera 07"),
            "cam_stub_08": ("Berth 405", "Crane 08", "Camera 08"),
            "cam_stub_09": ("Berth 405", "Crane 09", "Camera 09"),
            "cam_stub_10": ("Berth 406", "Crane 10", "Camera 10"),
            "cam_stub_11": ("Berth 406", "Crane 11", "Camera 11"),
            "cam_stub_12": ("Berth 406", "Crane 12", "Camera 12"),
            "cam_stub_13": ("Berth 406", "Crane 13", "Camera 13"),
            "cam_stub_14": ("Berth 406", "Crane 14", "Camera 14"),
            # Camera 15 covers the gate area only — no berth/crane assigned
            # TODO (camera-mapping-3): Camera 15 berth/crane assignment unknown
            "cam_stub_15": ("", "", "Camera 15"),
        }
        # --- End stub mapping table ---------------------------------------

        if camera_id in _table:
            berth, crane, camera = _table[camera_id]
        else:
            # Unrecognised camera_id — use camera_id as the camera label so
            # the event remains traceable even without a formal registry entry.
            berth, crane, camera = "", "", camera_id

        return cls(
            facility="Railyard",
            berth=berth,
            crane=crane,
            camera=camera,
            landmark="",
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "facility": self.facility,
            "berth": self.berth,
            "crane": self.crane,
            "camera": self.camera,
            "landmark": self.landmark,
        }


# ---------------------------------------------------------------------------
# HazardEvent
# ---------------------------------------------------------------------------


@dataclass
class HazardEvent:
    """
    A structured record stored in HazardStore when a hazardous detection occurs.

    Created by the Flask backend (app.py) after a successful inference call
    that produced at least one HazardResult with is_hazard=True.

    Fields
    ------
    event_id : str
        UUID4 string — unique across the system's operational lifetime.
    hazard_type : str
        Equal to the hazard_result.hazard_reason that triggered this event.
    camera_id : str
        Camera identifier supplied to the inference call.
    timestamp : str
        ISO 8601 UTC timestamp (e.g. "2025-07-10T14:32:00Z").
    confidence : float
        YOLO confidence score for the triggering detection, clamped to [0,1].
    bbox : BBox
        Normalised bounding box of the triggering detection.
    annotated_image : Optional[str]
        Base64-encoded PNG of the annotated frame; stored in memory for modal
        display.  None when annotation failed.
    location : LocationContext
        Stub location metadata derived from camera_id.

    Requirement: 10.4
    """

    event_id: str                        # UUID4
    hazard_type: str                     # == hazard_result.hazard_reason
    camera_id: str
    timestamp: str                       # ISO 8601 UTC
    confidence: float
    bbox: BBox
    annotated_image: Optional[str]       # base64 PNG or None
    location: LocationContext            # stub location metadata

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for API responses."""
        return {
            "event_id": self.event_id,
            "hazard_type": self.hazard_type,
            "is_hazard": self.hazard_type not in ("no_hazard", "", None),
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "bbox": {
                "x_center": self.bbox.x_center,
                "y_center": self.bbox.y_center,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "annotated_image": self.annotated_image,
            "location": self.location.to_dict(),
        }


# ---------------------------------------------------------------------------
# InferenceEngineConfig
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig:
    """
    Configuration for the InferenceEngine component.

    Raises ``ValueError`` at construction time if:
    - ``confidence_threshold`` is outside [0.0, 1.0], or
    - ``checkpoint_path`` is empty or None.

    Requirements: 1.6, 17.1, 17.3

    Fields
    ------
    checkpoint_path : str
        Path to the YOLO model checkpoint (.pt file).
    confidence_threshold : float
        Minimum confidence for a detection to be included in hazard
        classification.  Must be in [0.0, 1.0].  Default 0.5.
    device : str
        Inference device: "cuda" or "cpu".  Default "cpu".
    flipped_aspect_ratio_threshold : float
        Height/width ratio above which a container bbox is classified as
        flipped.  Default 1.5.
    iou_threshold : float
        IoU threshold used by the Container - Open loading suppression rule.
        Default 0.5.
    """

    checkpoint_path: str
    confidence_threshold: float = 0.5
    device: str = "cpu"
    flipped_aspect_ratio_threshold: float = 1.5
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration fields at construction time."""
        # Validate checkpoint_path — must be a non-empty string
        if not self.checkpoint_path or not isinstance(self.checkpoint_path, str):
            raise ValueError(
                "InferenceEngineConfig.checkpoint_path must be a non-empty string, "
                f"got {self.checkpoint_path!r}"
            )
        if not self.checkpoint_path.strip():
            raise ValueError(
                "InferenceEngineConfig.checkpoint_path must be a non-empty string, "
                f"got {self.checkpoint_path!r}"
            )

        # Validate confidence_threshold — must be in [0.0, 1.0]
        if not isinstance(self.confidence_threshold, (int, float)):
            raise ValueError(
                "InferenceEngineConfig.confidence_threshold must be a float in "
                f"[0.0, 1.0], got {self.confidence_threshold!r}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                "InferenceEngineConfig.confidence_threshold must be in [0.0, 1.0], "
                f"got {self.confidence_threshold!r}"
            )
