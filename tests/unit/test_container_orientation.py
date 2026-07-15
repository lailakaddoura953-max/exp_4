"""
Unit tests for Container Door State and Orientation Detection.

Tests cover:
- Property 9: Container door temporal confirmation with loading suppression
- Property 10: Container flipped detection via aspect ratio
- Property 11: Container dangling detection (Picked without adequate Crane overlap)
- Loading operation suppression logic
- Unconfirmed detections (<2 frames) are not hazards

Visual outputs:
- tests/output/container_orientation_frames.png
- tests/output/container_aspect_ratios.png

**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 5.2, 5.3, 5.4, 5.6, 5.7**
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.container_analyzer import ContainerAnalyzer
from hazard_detection.models import (
    BBox,
    ContainerAnalyzerConfig,
    Detection,
    FrameSequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Default ContainerAnalyzerConfig for orientation tests."""
    return ContainerAnalyzerConfig(
        confidence_threshold=0.5,
        flipped_aspect_ratio_threshold=1.5,
        safe_overlap_threshold=0.3,
        ground_level_threshold=0.4,
        motion_threshold=0.7,
        iou_threshold=0.5,
    )


@pytest.fixture
def mock_flow_analyzer():
    """Mock OpticalFlowAnalyzer that returns consistent flow data."""
    mock = MagicMock()
    mock.compute_flow.return_value = MagicMock(
        flow_vectors=np.zeros((100, 100, 2), dtype=np.float32),
        confidence=np.ones((100, 100), dtype=np.float32) * 0.5,
        mean_magnitude=0.1,
        mean_direction=0.0,
        frame_shape=(100, 100),
    )
    mock.get_flow_consistency_score.return_value = 0.3
    return mock


@pytest.fixture
def analyzer(mock_flow_analyzer, default_config):
    """ContainerAnalyzer instance with mocked flow analyzer."""
    return ContainerAnalyzer(
        flow_analyzer=mock_flow_analyzer,
        config=default_config,
    )


@pytest.fixture
def frame_sequence_3():
    """A 3-frame sequence for testing."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    return FrameSequence(
        frames=frames, camera_id="cam_01", timestamps=[1.0, 2.0, 3.0]
    )


@pytest.fixture
def frame_sequence_5():
    """A 5-frame sequence for extended temporal tests."""
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    return FrameSequence(
        frames=frames,
        camera_id="cam_02",
        timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
    )


def _det(class_label: str, confidence: float, bbox: BBox) -> Detection:
    """Helper to create a Detection object."""
    return Detection(bbox=bbox, class_label=class_label, confidence=confidence)


# ===========================================================================
# Property 9: Container door temporal confirmation with loading suppression
# ===========================================================================


class TestDoorTemporalConfirmationWithSuppression:
    """
    **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

    Property 9: For any "Container - Open" detection above threshold,
    a Hazard_Event with hazard_type "container_door_open" SHALL only be
    emitted when confirmed in >=2 frames. If the detection spatially overlaps
    (IoU >= 0.5) with "Container - Picked" or "Crane", the hazard SHALL
    be suppressed.
    """

    def test_door_open_confirmed_2_frames_is_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.2: Open in >=2 frames with no overlap => is_hazard=True."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        det = _det("Container - Open", 0.75, bbox)
        detections = [[det], [det], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 1
        assert door_events[0].is_hazard is True
        assert door_events[0].metadata.frames_detected >= 2

    def test_door_open_confirmed_3_frames_is_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.2: Open in all 3 frames => is_hazard=True."""
        bbox = BBox(x_center=0.4, y_center=0.6, width=0.25, height=0.15)
        det = _det("Container - Open", 0.82, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 1
        assert door_events[0].is_hazard is True
        assert door_events[0].metadata.frames_detected == 3

    def test_door_open_single_frame_not_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.3: Open in <2 frames => is_hazard=False (logged only)."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        det = _det("Container - Open", 0.7, bbox)
        detections = [[det], [], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 1
        assert door_events[0].is_hazard is False
        assert door_events[0].metadata.frames_detected == 1

    def test_door_open_suppressed_by_picked_overlap(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.4: Overlap with Picked (IoU>=0.5) suppresses door event."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _det("Container - Open", 0.8, bbox)
        picked = _det("Container - Picked", 0.75, bbox)  # Same bbox => IoU=1.0
        detections = [
            [door_open, picked],
            [door_open, picked],
            [],
        ]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        # Door open is suppressed because of loading operation
        assert len(door_events) == 0

    def test_door_open_suppressed_by_crane_overlap(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.4: Overlap with Crane (IoU>=0.5) suppresses door event."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _det("Container - Open", 0.8, bbox)
        crane = _det("Crane", 0.92, bbox)
        detections = [
            [door_open, crane],
            [door_open, crane],
            [],
        ]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 0

    def test_door_open_not_suppressed_by_non_overlapping_picked(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.4: No overlap with Picked => door event is NOT suppressed."""
        door_bbox = BBox(x_center=0.2, y_center=0.5, width=0.2, height=0.15)
        picked_bbox = BBox(x_center=0.8, y_center=0.5, width=0.2, height=0.15)
        door_open = _det("Container - Open", 0.8, door_bbox)
        picked = _det("Container - Picked", 0.75, picked_bbox)
        detections = [
            [door_open, picked],
            [door_open, picked],
            [],
        ]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 1
        assert door_events[0].is_hazard is True

    def test_door_open_below_confidence_no_event(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.5: Below threshold => no event emitted at all."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        det = _det("Container - Open", 0.3, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 0

    def test_door_open_suppressed_in_some_frames_not_others(
        self, analyzer, frame_sequence_5
    ):
        """Loading overlap in some frames may prevent temporal confirmation."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _det("Container - Open", 0.8, bbox)
        picked = _det("Container - Picked", 0.7, bbox)

        # Frame 0: door + picked (suppressed), Frame 1: door only, rest empty
        detections = [
            [door_open, picked],  # suppressed
            [door_open],          # not suppressed -> 1 frame only
            [],
            [],
            [],
        ]

        events = analyzer.analyze(detections, frame_sequence_5)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        # Only 1 unsuppressed frame => not confirmed
        assert len(door_events) == 1
        assert door_events[0].is_hazard is False


# ===========================================================================
# Property 10: Container flipped detection
# ===========================================================================


class TestFlippedDetection:
    """
    **Validates: Requirements 5.2, 5.7**

    Property 10: For any container detection whose bounding box
    height-to-width ratio exceeds the configured flipped_aspect_ratio_threshold
    and is confirmed in >=2 frames, the system SHALL emit a Hazard_Event
    with hazard_type "container_flipped".
    """

    def test_flipped_confirmed_2_frames_is_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.2: Aspect ratio > threshold in >=2 frames => is_hazard=True."""
        # h/w = 0.5/0.2 = 2.5 > 1.5 threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.5)
        det = _det("Container - Stacked", 0.85, bbox)
        detections = [[det], [det], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is True
        assert flipped_events[0].metadata.frames_detected >= 2

    def test_flipped_single_frame_not_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.7: Flipped in <2 frames => is_hazard=False (logged only)."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.5)
        det = _det("Container - Separate", 0.8, bbox)
        detections = [[det], [], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is False
        assert flipped_events[0].metadata.frames_detected == 1

    def test_normal_aspect_ratio_no_flipped_event(
        self, analyzer, frame_sequence_3
    ):
        """Normal container (wider than tall) does not trigger flipped."""
        # h/w = 0.15/0.4 = 0.375 < 1.5 threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.15)
        det = _det("Container - Stacked", 0.9, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 0

    def test_flipped_at_exact_threshold_not_flipped(
        self, analyzer, frame_sequence_3
    ):
        """At exactly 1.5 ratio (not >) => not flipped."""
        # h/w = 0.3/0.2 = 1.5 exactly => not > threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3)
        det = _det("Container - Reefer", 0.8, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 0

    def test_flipped_slightly_above_threshold(
        self, analyzer, frame_sequence_3
    ):
        """Ratio just above threshold triggers flipped detection."""
        # h/w = 0.32/0.2 = 1.6 > 1.5 threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.32)
        det = _det("Container - Open", 0.7, bbox)
        detections = [[det], [det], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is True

    def test_flipped_below_confidence_no_event(
        self, analyzer, frame_sequence_3
    ):
        """Below confidence threshold => no flipped event even with high ratio."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.15, height=0.45)
        det = _det("Container - Stacked", 0.3, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 0

    def test_flipped_multiple_containers_different_ratios(
        self, analyzer, frame_sequence_3
    ):
        """Multiple containers: only flipped ones emit events."""
        normal_bbox = BBox(x_center=0.3, y_center=0.5, width=0.3, height=0.15)
        flipped_bbox = BBox(x_center=0.7, y_center=0.5, width=0.15, height=0.4)
        normal = _det("Container - Stacked", 0.8, normal_bbox)
        flipped = _det("Container - Stacked", 0.8, flipped_bbox)
        detections = [[normal, flipped], [normal, flipped], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]

        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is True


# ===========================================================================
# Property 11: Container dangling detection
# ===========================================================================


class TestDanglingDetection:
    """
    **Validates: Requirements 5.3, 5.4, 5.6, 5.7**

    Property 11: For any "Container - Picked" detection, if either (a) no
    Crane detection exists in the same frame, or (b) the intersection-over-area
    ratio with the nearest Crane is below safe_overlap_threshold and the
    container's vertical midpoint is above ground_level_threshold, and this
    is confirmed in >=2 frames, then emit "container_dangling".
    """

    def test_dangling_no_crane_confirmed_is_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.4: Picked with no Crane in >=2 frames => is_hazard=True."""
        bbox = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.15)
        det = _det("Container - Picked", 0.85, bbox)
        detections = [[det], [det], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is True
        assert dangling_events[0].metadata.frames_detected >= 2

    def test_dangling_no_crane_single_frame_not_hazard(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.7: Dangling in <2 frames => is_hazard=False."""
        bbox = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.15)
        det = _det("Container - Picked", 0.8, bbox)
        detections = [[det], [], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is False

    def test_dangling_insufficient_crane_overlap_above_ground(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.3: Picked with low crane overlap + high position => dangling."""
        # Picked is high (y=0.2 < ground=0.4), crane is far away
        picked_bbox = BBox(x_center=0.3, y_center=0.2, width=0.2, height=0.1)
        crane_bbox = BBox(x_center=0.8, y_center=0.8, width=0.15, height=0.15)
        picked = _det("Container - Picked", 0.8, picked_bbox)
        crane = _det("Crane", 0.9, crane_bbox)
        detections = [[picked, crane], [picked, crane], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is True

    def test_safe_crane_overlap_not_dangling(
        self, analyzer, frame_sequence_3
    ):
        """Req 5.6: Adequate crane overlap => no dangling event."""
        picked_bbox = BBox(x_center=0.5, y_center=0.3, width=0.2, height=0.12)
        # Crane fully covers the picked container
        crane_bbox = BBox(x_center=0.5, y_center=0.3, width=0.5, height=0.5)
        picked = _det("Container - Picked", 0.85, picked_bbox)
        crane = _det("Crane", 0.95, crane_bbox)
        detections = [[picked, crane], [picked, crane], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 0

    def test_dangling_below_ground_not_dangling(
        self, analyzer, frame_sequence_3
    ):
        """Picked below ground-level threshold is not dangling."""
        # y=0.6 > ground_level=0.4 => below ground, not dangerous
        picked_bbox = BBox(x_center=0.5, y_center=0.6, width=0.2, height=0.1)
        crane_bbox = BBox(x_center=0.8, y_center=0.8, width=0.1, height=0.1)
        picked = _det("Container - Picked", 0.8, picked_bbox)
        crane = _det("Crane", 0.9, crane_bbox)
        detections = [[picked, crane], [picked, crane], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 0

    def test_dangling_below_confidence_no_event(
        self, analyzer, frame_sequence_3
    ):
        """Below confidence threshold => no dangling event."""
        bbox = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.15)
        det = _det("Container - Picked", 0.3, bbox)
        detections = [[det], [det], [det]]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(dangling_events) == 0

    def test_dangling_multiple_cranes_best_overlap_used(
        self, analyzer, frame_sequence_3
    ):
        """Multiple cranes: best IoA is used for dangling check."""
        picked_bbox = BBox(x_center=0.5, y_center=0.3, width=0.2, height=0.12)
        # Far crane (no overlap)
        crane_far = BBox(x_center=0.9, y_center=0.9, width=0.1, height=0.1)
        # Close crane (covers picked)
        crane_close = BBox(x_center=0.5, y_center=0.3, width=0.4, height=0.4)
        picked = _det("Container - Picked", 0.8, picked_bbox)
        crane1 = _det("Crane", 0.9, crane_far)
        crane2 = _det("Crane", 0.9, crane_close)
        detections = [[picked, crane1, crane2], [picked, crane1, crane2], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        # Close crane provides adequate overlap => not dangling
        assert len(dangling_events) == 0


# ===========================================================================
# Loading Operation Suppression Logic (combined scenarios)
# ===========================================================================


class TestLoadingOperationSuppression:
    """
    Tests for loading operation suppression across different scenarios.

    **Validates: Requirements 4.4, 4.5**
    """

    def test_picked_prioritized_over_open_iou_above_threshold(
        self, analyzer, frame_sequence_3
    ):
        """Req 4.5: Same region IoU>=0.5, Picked prioritized."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.25, height=0.18)
        door_open = _det("Container - Open", 0.85, bbox)
        picked = _det("Container - Picked", 0.6, bbox)
        detections = [[door_open, picked], [door_open, picked], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 0

    def test_partial_iou_overlap_below_threshold_not_suppressed(
        self, analyzer, frame_sequence_3
    ):
        """Overlap with IoU < 0.5 does NOT suppress door open."""
        door_bbox = BBox(x_center=0.3, y_center=0.5, width=0.2, height=0.15)
        picked_bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.15)
        door_open = _det("Container - Open", 0.75, door_bbox)
        picked = _det("Container - Picked", 0.7, picked_bbox)
        detections = [[door_open, picked], [door_open, picked], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        # Low IoU => not suppressed, and confirmed in 2 frames
        assert len(door_events) == 1
        assert door_events[0].is_hazard is True

    def test_crane_overlap_suppresses_even_high_confidence_door(
        self, analyzer, frame_sequence_3
    ):
        """Crane with IoU>=0.5 suppresses even high-confidence door open."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _det("Container - Open", 0.95, bbox)
        crane = _det("Crane", 0.6, bbox)
        detections = [[door_open, crane], [door_open, crane], []]

        events = analyzer.analyze(detections, frame_sequence_3)
        door_events = [e for e in events if e.hazard_type == "container_door_open"]

        assert len(door_events) == 0


# ===========================================================================
# Unconfirmed Detections (<2 frames) are not hazards
# ===========================================================================


class TestUnconfirmedDetections:
    """
    Tests verifying that unconfirmed detections (present in fewer than 2
    frames) are logged but NOT emitted as is_hazard=True.

    **Validates: Requirements 4.3, 5.7**
    """

    def test_all_types_single_frame_not_hazard(
        self, analyzer, frame_sequence_5
    ):
        """All hazard types in single frame => is_hazard=False."""
        door_bbox = BBox(x_center=0.3, y_center=0.5, width=0.2, height=0.15)
        flipped_bbox = BBox(x_center=0.6, y_center=0.5, width=0.1, height=0.3)
        dangling_bbox = BBox(x_center=0.8, y_center=0.2, width=0.15, height=0.1)

        door_open = _det("Container - Open", 0.8, door_bbox)
        flipped = _det("Container - Stacked", 0.8, flipped_bbox)
        dangling = _det("Container - Picked", 0.8, dangling_bbox)

        # Each detection appears in only 1 frame
        detections = [[door_open, flipped, dangling], [], [], [], []]

        events = analyzer.analyze(detections, frame_sequence_5)

        for event in events:
            assert event.is_hazard is False
            assert event.metadata.frames_detected < 2

    def test_confirmed_detections_are_hazards(
        self, analyzer, frame_sequence_5
    ):
        """Verified: >=2 frames for each type => is_hazard=True."""
        door_bbox = BBox(x_center=0.3, y_center=0.5, width=0.2, height=0.15)
        flipped_bbox = BBox(x_center=0.6, y_center=0.5, width=0.1, height=0.3)
        dangling_bbox = BBox(x_center=0.8, y_center=0.2, width=0.15, height=0.1)

        door_open = _det("Container - Open", 0.8, door_bbox)
        flipped = _det("Container - Stacked", 0.8, flipped_bbox)
        dangling = _det("Container - Picked", 0.8, dangling_bbox)

        # Each detection in 2+ frames
        detections = [
            [door_open, flipped, dangling],
            [door_open, flipped, dangling],
            [],
            [],
            [],
        ]

        events = analyzer.analyze(detections, frame_sequence_5)

        door_events = [e for e in events if e.hazard_type == "container_door_open"]
        flipped_events = [e for e in events if e.hazard_type == "container_flipped"]
        dangling_events = [e for e in events if e.hazard_type == "container_dangling"]

        assert len(door_events) == 1
        assert door_events[0].is_hazard is True
        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is True
        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is True


# ===========================================================================
# Visual Output Generation
# ===========================================================================


class TestVisualOutputs:
    """
    Generate visual diagnostic outputs for container orientation detection.
    Saved to tests/output/ directory.
    """

    def test_generate_orientation_frames_visual(
        self, analyzer, output_dir
    ):
        """Generate annotated frames showing flipped/dangling states."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray frame

        # --- Panel 1: Flipped Container ---
        ax = axes[0]
        ax.imshow(frame)
        # Normal container
        rect_normal = mpatches.FancyBboxPatch(
            (50, 300), 200, 80, boxstyle="round,pad=0",
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect_normal)
        ax.text(55, 295, "Normal (w>h)", fontsize=9, color="green")

        # Flipped container (taller than wide)
        rect_flipped = mpatches.FancyBboxPatch(
            (350, 100), 80, 280, boxstyle="round,pad=0",
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect_flipped)
        ax.text(355, 90, "FLIPPED (h/w > 1.5)", fontsize=9, color="red")
        ax.set_title("Flipped Detection", fontweight="bold")
        ax.axis("off")

        # --- Panel 2: Dangling Container ---
        ax = axes[1]
        ax.imshow(frame)
        # Crane at top
        rect_crane = mpatches.FancyBboxPatch(
            (200, 20), 250, 100, boxstyle="round,pad=0",
            linewidth=2, edgecolor="purple", facecolor="purple", alpha=0.2
        )
        ax.add_patch(rect_crane)
        ax.text(210, 10, "Crane", fontsize=9, color="purple")

        # Dangling container (far from crane)
        rect_dangling = mpatches.FancyBboxPatch(
            (50, 80), 120, 60, boxstyle="round,pad=0",
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect_dangling)
        ax.text(55, 70, "DANGLING (low crane overlap)", fontsize=9, color="red")

        # Safe container (within crane)
        rect_safe = mpatches.FancyBboxPatch(
            (250, 50), 120, 60, boxstyle="round,pad=0",
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect_safe)
        ax.text(255, 40, "Safe (adequate overlap)", fontsize=9, color="green")
        ax.set_title("Dangling Detection", fontweight="bold")
        ax.axis("off")

        # --- Panel 3: Door Open with Suppression ---
        ax = axes[2]
        ax.imshow(frame)
        # Door open alone
        rect_door = mpatches.FancyBboxPatch(
            (50, 200), 150, 100, boxstyle="round,pad=0",
            linewidth=2, edgecolor="orange", facecolor="none"
        )
        ax.add_patch(rect_door)
        ax.text(55, 190, "Door Open (HAZARD)", fontsize=9, color="orange")

        # Door open + Picked overlay (suppressed)
        rect_suppressed = mpatches.FancyBboxPatch(
            (350, 200), 150, 100, boxstyle="round,pad=0",
            linewidth=2, edgecolor="gray", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect_suppressed)
        ax.text(
            355, 190, "Door Open + Picked (SUPPRESSED)",
            fontsize=8, color="gray"
        )
        ax.set_title("Door Open Suppression", fontweight="bold")
        ax.axis("off")

        plt.suptitle(
            "Container Orientation & Door State Detection",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        out_path = output_dir / "container_orientation_frames.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert out_path.exists()

    def test_generate_aspect_ratio_distribution(self, output_dir):
        """Generate aspect ratio distribution plot for containers."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)

        # Simulate normal containers (wider than tall, ratio < 1.0)
        normal_ratios = rng.uniform(0.3, 0.8, size=60)
        # Simulate borderline containers (near threshold)
        borderline_ratios = rng.uniform(1.2, 1.6, size=15)
        # Simulate flipped containers (taller than wide, ratio > 1.5)
        flipped_ratios = rng.uniform(1.6, 3.5, size=25)

        all_ratios = np.concatenate([normal_ratios, borderline_ratios, flipped_ratios])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            all_ratios, bins=30, alpha=0.7, color="#3498db",
            edgecolor="black", linewidth=0.5
        )
        ax.axvline(
            1.5, color="red", linestyle="--", linewidth=2,
            label="Flipped Threshold (1.5)"
        )
        ax.axvline(
            np.mean(all_ratios), color="green", linestyle="--",
            linewidth=1.5, label=f"Mean ({np.mean(all_ratios):.2f})"
        )

        # Shade regions
        ax.axvspan(0, 1.5, alpha=0.05, color="green", label="Normal Zone")
        ax.axvspan(1.5, 4.0, alpha=0.05, color="red", label="Flipped Zone")

        ax.set_xlabel("Height/Width Aspect Ratio", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "Container Aspect Ratio Distribution",
            fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / "container_aspect_ratios.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert out_path.exists()
