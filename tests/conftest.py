"""
Shared pytest fixtures for Hazard Detection System tests.

Provides reusable fixtures for:
- Mock camera feeds (numpy arrays)
- Sample Detection objects
- Sample zone configurations (polygons)
- Output directory setup
- Sample HazardEvent factory
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

# Add src to path so we can import from the actual hazard_detection package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hazard_detection.models import (
    BBox,
    Detection,
    DiagnosticMetadata,
    FrameSequence,
    HazardEvent,
    ZonePolygon,
)


# ---------------------------------------------------------------------------
# Output directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Ensure tests/output/ directory exists for visual diagnostic artifacts."""
    out = Path(__file__).parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Mock camera feeds
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_frame() -> np.ndarray:
    """Single 640x640 RGB frame (uint8) with random pixel data."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_frame_sequence() -> FrameSequence:
    """6-frame sequence (default config) from cam_01 with sequential timestamps."""
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8) for _ in range(6)]
    timestamps = [1700000000.0 + i * 0.5 for i in range(6)]
    return FrameSequence(frames=frames, camera_id="cam_01", timestamps=timestamps)


@pytest.fixture
def mock_frame_sequence_factory():
    """Factory fixture to create frame sequences with custom parameters."""
    def _create(
        num_frames: int = 6,
        camera_id: str = "cam_01",
        resolution: Tuple[int, int] = (640, 640),
        seed: int = 42,
    ) -> FrameSequence:
        rng = np.random.default_rng(seed)
        frames = [
            rng.integers(0, 256, size=(resolution[0], resolution[1], 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]
        timestamps = [1700000000.0 + i * 0.5 for i in range(num_frames)]
        return FrameSequence(frames=frames, camera_id=camera_id, timestamps=timestamps)

    return _create


# ---------------------------------------------------------------------------
# Sample Detection objects
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_human_detection() -> Detection:
    """Human detection in center of frame with high confidence."""
    return Detection(
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.3),
        class_label="Human",
        confidence=0.85,
    )


@pytest.fixture
def sample_human_no_ppe_detection() -> Detection:
    """Human without safety clothes detection."""
    return Detection(
        bbox=BBox(x_center=0.3, y_center=0.6, width=0.08, height=0.25),
        class_label="Human - No Safety Clothes",
        confidence=0.78,
    )


@pytest.fixture
def sample_container_misaligned_detection() -> Detection:
    """Misaligned container detection."""
    return Detection(
        bbox=BBox(x_center=0.6, y_center=0.4, width=0.25, height=0.15),
        class_label="Container - Misaligned",
        confidence=0.72,
    )


@pytest.fixture
def sample_container_open_detection() -> Detection:
    """Open container door detection."""
    return Detection(
        bbox=BBox(x_center=0.4, y_center=0.5, width=0.2, height=0.18),
        class_label="Container - Open",
        confidence=0.68,
    )


@pytest.fixture
def sample_container_stacked_detection() -> Detection:
    """Stacked container detection (normal state)."""
    return Detection(
        bbox=BBox(x_center=0.6, y_center=0.35, width=0.24, height=0.14),
        class_label="Container - Stacked",
        confidence=0.91,
    )


@pytest.fixture
def sample_container_picked_detection() -> Detection:
    """Picked container (being moved by crane)."""
    return Detection(
        bbox=BBox(x_center=0.5, y_center=0.2, width=0.2, height=0.12),
        class_label="Container - Picked",
        confidence=0.82,
    )


@pytest.fixture
def sample_crane_detection() -> Detection:
    """Crane detection overlapping with picked container."""
    return Detection(
        bbox=BBox(x_center=0.5, y_center=0.15, width=0.3, height=0.25),
        class_label="Crane",
        confidence=0.95,
    )


@pytest.fixture
def sample_detection_factory():
    """Factory fixture to create Detection objects with custom parameters."""
    def _create(
        class_label: str = "Human",
        confidence: float = 0.8,
        x_center: float = 0.5,
        y_center: float = 0.5,
        width: float = 0.1,
        height: float = 0.2,
    ) -> Detection:
        return Detection(
            bbox=BBox(x_center=x_center, y_center=y_center, width=width, height=height),
            class_label=class_label,
            confidence=confidence,
        )

    return _create


# ---------------------------------------------------------------------------
# Sample zone configurations
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_no_people_zone() -> ZonePolygon:
    """No-people zone covering the left half of the frame."""
    return ZonePolygon(
        vertices=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        zone_type="no_people",
        camera_id="cam_01",
    )


@pytest.fixture
def sample_operation_zone() -> ZonePolygon:
    """Operation zone covering the right half of the frame."""
    return ZonePolygon(
        vertices=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
        zone_type="operation",
        camera_id="cam_01",
    )


@pytest.fixture
def sample_dropoff_zone() -> ZonePolygon:
    """Dropoff zone in the bottom-right corner."""
    return ZonePolygon(
        vertices=[(0.7, 0.7), (1.0, 0.7), (1.0, 1.0), (0.7, 1.0)],
        zone_type="dropoff",
        camera_id="cam_01",
    )


@pytest.fixture
def sample_zone_config() -> dict:
    """Full zone configuration for cam_01 with all zone types."""
    return {
        "cam_01": {
            "zones": [
                {
                    "vertices": [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
                    "zone_type": "no_people",
                },
                {
                    "vertices": [(0.4, 0.0), (0.8, 0.0), (0.8, 1.0), (0.4, 1.0)],
                    "zone_type": "operation",
                },
                {
                    "vertices": [(0.8, 0.6), (1.0, 0.6), (1.0, 1.0), (0.8, 1.0)],
                    "zone_type": "dropoff",
                },
            ]
        }
    }


@pytest.fixture
def sample_zone_config_multi_camera() -> dict:
    """Zone configuration spanning multiple cameras."""
    return {
        "cam_01": {
            "zones": [
                {
                    "vertices": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
                    "zone_type": "no_people",
                },
                {
                    "vertices": [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
                    "zone_type": "operation",
                },
            ]
        },
        "cam_02": {
            "zones": [
                {
                    "vertices": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                    "zone_type": "no_people",
                },
                {
                    "vertices": [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)],
                    "zone_type": "dropoff",
                },
            ]
        },
    }


# ---------------------------------------------------------------------------
# HazardEvent factory
# ---------------------------------------------------------------------------

@pytest.fixture
def hazard_event_factory():
    """Factory fixture to create HazardEvent objects with sensible defaults."""
    def _create(
        hazard_type: str = "zone_violation",
        camera_id: str = "cam_01",
        is_hazard: bool = True,
        confidence: float = 0.85,
        x_center: float = 0.5,
        y_center: float = 0.5,
        width: float = 0.1,
        height: float = 0.3,
        frame_index: int = 0,
        detection_class: str = "Human",
        frames_detected: int = 3,
        flow_consistency_score: Optional[float] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> HazardEvent:
        return HazardEvent(
            event_id=event_id or str(uuid.uuid4()),
            hazard_type=hazard_type,
            camera_id=camera_id,
            timestamp=timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            is_hazard=is_hazard,
            confidence=confidence,
            bbox=BBox(x_center=x_center, y_center=y_center, width=width, height=height),
            metadata=DiagnosticMetadata(
                frame_index=frame_index,
                detection_class=detection_class,
                frames_detected=frames_detected,
                flow_consistency_score=flow_consistency_score,
            ),
        )

    return _create
