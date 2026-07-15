"""
Unit tests for the Container Analyzer module.

Tests cover:
- IoU and IoA geometric calculations
- Misalignment detection with disambiguation
- Door open detection with loading operation suppression
- Flipped container detection via aspect ratio
- Dangling container detection
- Temporal confirmation (>=2 frames required for is_hazard=True)
- Flow consistency scoring (Property 8)
- Visual output generation (IoU distribution, annotated frames)

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.container_analyzer import ContainerAnalyzer
from hazard_detection.models import (
    BBox,
    ContainerAnalyzerConfig,
    Detection,
    FrameSequence,
    HazardEvent,
)
from cv.flow_analyzer import OpticalFlowAnalyzer, FlowConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Default ContainerAnalyzerConfig."""
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
    """Mock OpticalFlowAnalyzer that doesn't require real frames."""
    mock = MagicMock(spec=OpticalFlowAnalyzer)
    mock.compute_flow.return_value = MagicMock(
        flow_vectors=np.zeros((100, 100, 2), dtype=np.float32),
        confidence=np.ones((100, 100), dtype=np.float32) * 0.5,
        mean_magnitude=0.1,
        mean_direction=0.0,
        frame_shape=(100, 100),
    )
    mock.get_flow_consistency_score.return_value = 0.8
    return mock


@pytest.fixture
def analyzer(mock_flow_analyzer, default_config):
    """ContainerAnalyzer instance with mocked flow analyzer."""
    return ContainerAnalyzer(
        flow_analyzer=mock_flow_analyzer,
        config=default_config,
    )


@pytest.fixture
def frame_sequence():
    """A simple 3-frame sequence for testing."""
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)
    ]
    return FrameSequence(
        frames=frames,
        camera_id="cam_01",
        timestamps=[1.0, 2.0, 3.0],
    )


def _make_detection(class_label: str, confidence: float, bbox: BBox) -> Detection:
    """Helper to create a Detection."""
    return Detection(bbox=bbox, class_label=class_label, confidence=confidence)


# ---------------------------------------------------------------------------
# Tests: IoU Calculation
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """Tests for _compute_iou geometric calculation."""

    def test_identical_boxes_return_1(self, analyzer):
        box = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        assert analyzer._compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_return_0(self, analyzer):
        box_a = BBox(x_center=0.2, y_center=0.5, width=0.2, height=0.2)
        box_b = BBox(x_center=0.8, y_center=0.5, width=0.2, height=0.2)
        assert analyzer._compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self, analyzer):
        box_a = BBox(x_center=0.4, y_center=0.5, width=0.4, height=0.4)
        box_b = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        iou = analyzer._compute_iou(box_a, box_b)
        # Partial overlap: intersection < union
        assert 0.0 < iou < 1.0

    def test_iou_is_symmetric(self, analyzer):
        box_a = BBox(x_center=0.3, y_center=0.4, width=0.3, height=0.2)
        box_b = BBox(x_center=0.4, y_center=0.5, width=0.2, height=0.3)
        assert analyzer._compute_iou(box_a, box_b) == pytest.approx(
            analyzer._compute_iou(box_b, box_a)
        )


# ---------------------------------------------------------------------------
# Tests: IoA Calculation
# ---------------------------------------------------------------------------


class TestComputeIoA:
    """Tests for _compute_ioa geometric calculation."""

    def test_inner_fully_inside_outer(self, analyzer):
        inner = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        outer = BBox(x_center=0.5, y_center=0.5, width=0.6, height=0.6)
        assert analyzer._compute_ioa(inner, outer) == pytest.approx(1.0)

    def test_no_overlap_returns_0(self, analyzer):
        inner = BBox(x_center=0.2, y_center=0.5, width=0.1, height=0.1)
        outer = BBox(x_center=0.8, y_center=0.5, width=0.1, height=0.1)
        assert analyzer._compute_ioa(inner, outer) == pytest.approx(0.0)

    def test_partial_coverage(self, analyzer):
        inner = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        outer = BBox(x_center=0.6, y_center=0.5, width=0.4, height=0.4)
        ioa = analyzer._compute_ioa(inner, outer)
        assert 0.0 < ioa < 1.0


# ---------------------------------------------------------------------------
# Tests: Flipped Detection
# ---------------------------------------------------------------------------


class TestIsFlipped:
    """Tests for _is_flipped aspect ratio check."""

    def test_normal_container_not_flipped(self, analyzer):
        # Width > height => ratio < 1.5
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        assert analyzer._is_flipped(bbox) is False

    def test_flipped_container_detected(self, analyzer):
        # Height >> width => ratio > 1.5
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.4)
        assert analyzer._is_flipped(bbox) is True

    def test_exactly_at_threshold_not_flipped(self, analyzer):
        # ratio = 1.5 exactly => not > threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3)
        assert analyzer._is_flipped(bbox) is False

    def test_zero_width_not_flipped(self, analyzer):
        # Edge case: zero width should not crash
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.0, height=0.3)
        assert analyzer._is_flipped(bbox) is False


# ---------------------------------------------------------------------------
# Tests: Dangling Detection
# ---------------------------------------------------------------------------


class TestIsDangling:
    """Tests for _is_dangling crane overlap check."""

    def test_no_crane_in_frame_is_dangling(self, analyzer):
        picked = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.2)
        assert analyzer._is_dangling(picked, []) is True

    def test_sufficient_crane_overlap_not_dangling(self, analyzer):
        picked = BBox(x_center=0.5, y_center=0.3, width=0.3, height=0.2)
        crane = BBox(x_center=0.5, y_center=0.3, width=0.5, height=0.5)
        assert analyzer._is_dangling(picked, [crane]) is False

    def test_insufficient_overlap_above_ground_is_dangling(self, analyzer):
        # Picked is high (y=0.2 < ground_level=0.4) with minimal crane overlap
        picked = BBox(x_center=0.3, y_center=0.2, width=0.2, height=0.1)
        crane = BBox(x_center=0.8, y_center=0.8, width=0.2, height=0.2)
        assert analyzer._is_dangling(picked, [crane]) is True

    def test_insufficient_overlap_below_ground_not_dangling(self, analyzer):
        # Picked is low (y=0.6 > ground_level=0.4) — not dangling
        picked = BBox(x_center=0.3, y_center=0.6, width=0.2, height=0.1)
        crane = BBox(x_center=0.8, y_center=0.8, width=0.2, height=0.2)
        assert analyzer._is_dangling(picked, [crane]) is False


# ---------------------------------------------------------------------------
# Tests: Full Analyze - Misalignment
# ---------------------------------------------------------------------------


class TestMisalignmentDetection:
    """Tests for misalignment detection logic in analyze()."""

    def test_misalignment_confirmed_in_2_frames(self, analyzer, frame_sequence):
        """Req 3.2: Misaligned in >=2 frames => is_hazard=True."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 1
        assert misalignment_events[0].is_hazard is True
        assert misalignment_events[0].metadata.frames_detected >= 2

    def test_misalignment_single_frame_not_hazard(self, analyzer, frame_sequence):
        """Req 3.4: Misaligned in <2 frames => is_hazard=False."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 1
        assert misalignment_events[0].is_hazard is False

    def test_misalignment_below_threshold_no_event(self, analyzer, frame_sequence):
        """Req 3.3: Below confidence threshold => log only (no event)."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.3, bbox)
        detections_per_frame = [[det], [det], [det]]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 0

    def test_misalignment_suppressed_by_stacked_higher_conf(
        self, analyzer, frame_sequence
    ):
        """Req 3.5: Overlapping stacked with higher confidence wins."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        misaligned = _make_detection("Container - Misaligned", 0.6, bbox)
        stacked = _make_detection("Container - Stacked", 0.9, bbox)
        detections_per_frame = [
            [misaligned, stacked],
            [misaligned, stacked],
            [],
        ]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        # Stacked wins (higher conf), misalignment is suppressed
        assert len(misalignment_events) == 0

    def test_misalignment_wins_over_lower_conf_stacked(
        self, analyzer, frame_sequence
    ):
        """Req 3.5: Misaligned wins when it has higher confidence."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        misaligned = _make_detection("Container - Misaligned", 0.9, bbox)
        stacked = _make_detection("Container - Stacked", 0.6, bbox)
        detections_per_frame = [
            [misaligned, stacked],
            [misaligned, stacked],
            [],
        ]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 1
        assert misalignment_events[0].is_hazard is True


# ---------------------------------------------------------------------------
# Tests: Full Analyze - Door Open
# ---------------------------------------------------------------------------


class TestDoorOpenDetection:
    """Tests for door open detection with suppression."""

    def test_door_open_confirmed_in_2_frames(self, analyzer, frame_sequence):
        """Req 4.2: Open in >=2 frames => is_hazard=True."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        det = _make_detection("Container - Open", 0.7, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        door_events = [
            e for e in events if e.hazard_type == "container_door_open"
        ]
        assert len(door_events) == 1
        assert door_events[0].is_hazard is True

    def test_door_open_single_frame_not_hazard(self, analyzer, frame_sequence):
        """Req 4.3: Open in <2 frames => is_hazard=False."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        det = _make_detection("Container - Open", 0.7, bbox)
        detections_per_frame = [[det], [], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        door_events = [
            e for e in events if e.hazard_type == "container_door_open"
        ]
        assert len(door_events) == 1
        assert door_events[0].is_hazard is False

    def test_door_open_suppressed_by_picked_overlap(
        self, analyzer, frame_sequence
    ):
        """Req 4.4: Overlapping with Picked suppresses door open."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _make_detection("Container - Open", 0.8, bbox)
        picked = _make_detection("Container - Picked", 0.7, bbox)
        detections_per_frame = [
            [door_open, picked],
            [door_open, picked],
            [],
        ]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        door_events = [
            e for e in events if e.hazard_type == "container_door_open"
        ]
        assert len(door_events) == 0

    def test_door_open_suppressed_by_crane_overlap(
        self, analyzer, frame_sequence
    ):
        """Req 4.4: Overlapping with Crane suppresses door open."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.3, height=0.2)
        door_open = _make_detection("Container - Open", 0.8, bbox)
        crane = _make_detection("Crane", 0.9, bbox)
        detections_per_frame = [
            [door_open, crane],
            [door_open, crane],
            [],
        ]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        door_events = [
            e for e in events if e.hazard_type == "container_door_open"
        ]
        assert len(door_events) == 0


# ---------------------------------------------------------------------------
# Tests: Full Analyze - Flipped
# ---------------------------------------------------------------------------


class TestFlippedDetection:
    """Tests for flipped container detection."""

    def test_flipped_confirmed_in_2_frames(self, analyzer, frame_sequence):
        """Req 5.2: Flipped aspect ratio in >=2 frames => is_hazard=True."""
        # height/width = 0.5/0.2 = 2.5 > 1.5 threshold
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.5)
        det = _make_detection("Container - Stacked", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        flipped_events = [
            e for e in events if e.hazard_type == "container_flipped"
        ]
        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is True

    def test_flipped_single_frame_not_hazard(self, analyzer, frame_sequence):
        """Req 5.7: Flipped in <2 frames => is_hazard=False."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.5)
        det = _make_detection("Container - Stacked", 0.8, bbox)
        detections_per_frame = [[det], [], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        flipped_events = [
            e for e in events if e.hazard_type == "container_flipped"
        ]
        assert len(flipped_events) == 1
        assert flipped_events[0].is_hazard is False

    def test_normal_aspect_ratio_no_flipped_event(self, analyzer, frame_sequence):
        """Normal container ratio should not produce flipped event."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Stacked", 0.8, bbox)
        detections_per_frame = [[det], [det], [det]]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        flipped_events = [
            e for e in events if e.hazard_type == "container_flipped"
        ]
        assert len(flipped_events) == 0


# ---------------------------------------------------------------------------
# Tests: Full Analyze - Dangling
# ---------------------------------------------------------------------------


class TestDanglingDetection:
    """Tests for dangling container detection."""

    def test_dangling_no_crane_confirmed(self, analyzer, frame_sequence):
        """Req 5.4: Picked with no Crane in >=2 frames => is_hazard=True."""
        bbox = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.2)
        det = _make_detection("Container - Picked", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        dangling_events = [
            e for e in events if e.hazard_type == "container_dangling"
        ]
        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is True

    def test_dangling_single_frame_not_hazard(self, analyzer, frame_sequence):
        """Req 5.7: Dangling in <2 frames => is_hazard=False."""
        bbox = BBox(x_center=0.5, y_center=0.2, width=0.3, height=0.2)
        det = _make_detection("Container - Picked", 0.8, bbox)
        detections_per_frame = [[det], [], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        dangling_events = [
            e for e in events if e.hazard_type == "container_dangling"
        ]
        assert len(dangling_events) == 1
        assert dangling_events[0].is_hazard is False

    def test_picked_with_adequate_crane_not_dangling(
        self, analyzer, frame_sequence
    ):
        """Req 5.6: Safe crane overlap => no dangling event."""
        bbox = BBox(x_center=0.5, y_center=0.3, width=0.3, height=0.2)
        picked = _make_detection("Container - Picked", 0.8, bbox)
        # Crane fully covers picked
        crane = _make_detection("Crane", 0.9,
                                BBox(x_center=0.5, y_center=0.3, width=0.6, height=0.6))
        detections_per_frame = [
            [picked, crane],
            [picked, crane],
            [],
        ]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        dangling_events = [
            e for e in events if e.hazard_type == "container_dangling"
        ]
        assert len(dangling_events) == 0


# ---------------------------------------------------------------------------
# Tests: Event Structure
# ---------------------------------------------------------------------------


class TestEventStructure:
    """Tests for emitted HazardEvent structure."""

    def test_event_has_required_fields(self, analyzer, frame_sequence):
        """Req 8.1: Events contain all mandatory fields."""
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        assert len(events) >= 1
        event = [e for e in events if e.hazard_type == "container_misalignment"][0]

        assert event.event_id is not None and len(event.event_id) > 0
        assert event.hazard_type == "container_misalignment"
        assert event.camera_id == "cam_01"
        assert event.timestamp is not None
        assert isinstance(event.is_hazard, bool)
        assert 0.0 <= event.confidence <= 1.0
        assert event.bbox is not None
        assert event.metadata is not None
        assert event.metadata.frames_detected >= 2



# ---------------------------------------------------------------------------
# Tests: Flow Consistency Scoring (Property 8)
# ---------------------------------------------------------------------------


class TestFlowConsistencyScoring:
    """
    Property 8: Flow consistency scoring.

    For any container bounding box in a Frame_Sequence with >=2 frames,
    the Container_Analyzer SHALL compute a flow_consistency_score from
    optical flow magnitude variance. When this score exceeds the configured
    motion_threshold, the container SHALL be flagged as a motion-based
    misalignment indicator with the score included in diagnostic metadata.

    **Validates: Requirements 3.5, 3.6**
    """

    def test_flow_score_above_threshold_flagged_in_metadata(
        self, default_config, frame_sequence
    ):
        """Flow variance above motion_threshold includes score in metadata."""
        mock_flow = MagicMock(spec=OpticalFlowAnalyzer)
        mock_flow.compute_flow.return_value = MagicMock(
            flow_vectors=np.zeros((100, 100, 2), dtype=np.float32),
            confidence=np.ones((100, 100), dtype=np.float32) * 0.5,
            mean_magnitude=0.1,
            mean_direction=0.0,
            frame_shape=(100, 100),
        )
        # Score 0.9 > motion_threshold 0.7
        mock_flow.get_flow_consistency_score.return_value = 0.9
        analyzer = ContainerAnalyzer(flow_analyzer=mock_flow, config=default_config)

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 1
        event = misalignment_events[0]
        # Flow score should be included in diagnostic metadata
        assert event.metadata.flow_consistency_score is not None
        assert event.metadata.flow_consistency_score == pytest.approx(0.9)

    def test_flow_score_below_threshold_still_recorded(
        self, default_config, frame_sequence
    ):
        """Flow score below threshold is still stored but doesn't flag motion."""
        mock_flow = MagicMock(spec=OpticalFlowAnalyzer)
        mock_flow.compute_flow.return_value = MagicMock(
            flow_vectors=np.zeros((100, 100, 2), dtype=np.float32),
            confidence=np.ones((100, 100), dtype=np.float32) * 0.5,
            mean_magnitude=0.1,
            mean_direction=0.0,
            frame_shape=(100, 100),
        )
        # Score 0.3 < motion_threshold 0.7
        mock_flow.get_flow_consistency_score.return_value = 0.3
        analyzer = ContainerAnalyzer(flow_analyzer=mock_flow, config=default_config)

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [det], []]

        events = analyzer.analyze(detections_per_frame, frame_sequence)
        misalignment_events = [
            e for e in events if e.hazard_type == "container_misalignment"
        ]
        assert len(misalignment_events) == 1
        event = misalignment_events[0]
        # Flow score should still be recorded even if below threshold
        assert event.metadata.flow_consistency_score is not None
        assert event.metadata.flow_consistency_score == pytest.approx(0.3)

    def test_flow_computation_invoked_for_multi_frame_sequence(
        self, default_config, frame_sequence
    ):
        """Flow analyzer is called for sequences with >=2 frames."""
        mock_flow = MagicMock(spec=OpticalFlowAnalyzer)
        mock_flow.compute_flow.return_value = MagicMock(
            flow_vectors=np.zeros((100, 100, 2), dtype=np.float32),
            confidence=np.ones((100, 100), dtype=np.float32) * 0.5,
            mean_magnitude=0.1,
            mean_direction=0.0,
            frame_shape=(100, 100),
        )
        mock_flow.get_flow_consistency_score.return_value = 0.8
        analyzer = ContainerAnalyzer(flow_analyzer=mock_flow, config=default_config)

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        det = _make_detection("Container - Misaligned", 0.8, bbox)
        detections_per_frame = [[det], [det], [det]]

        analyzer.analyze(detections_per_frame, frame_sequence)

        # With 3 frames, compute_flow should be called for 2 consecutive pairs
        assert mock_flow.compute_flow.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Visual Output Generation
# ---------------------------------------------------------------------------


class TestVisualOutputGeneration:
    """
    Visual output tests for container analyzer diagnostics.

    Generates:
    - tests/output/container_iou_distribution.png
    - tests/output/container_detection_frames.png
    """

    def test_generate_iou_distribution_plot(self, analyzer, output_dir):
        """
        Generate visual output: IoU distribution plot for test cases.
        Saves to tests/output/container_iou_distribution.png.
        """
        from tests.visual_helpers import plot_iou_distribution

        # Compute IoU values for a variety of bbox pairs
        test_pairs = [
            # Identical boxes
            (BBox(0.5, 0.5, 0.4, 0.4), BBox(0.5, 0.5, 0.4, 0.4)),
            # Slight offset
            (BBox(0.4, 0.5, 0.4, 0.4), BBox(0.5, 0.5, 0.4, 0.4)),
            # Moderate overlap
            (BBox(0.3, 0.5, 0.3, 0.3), BBox(0.5, 0.5, 0.3, 0.3)),
            # Large offset
            (BBox(0.2, 0.5, 0.3, 0.3), BBox(0.7, 0.5, 0.3, 0.3)),
            # Non-overlapping
            (BBox(0.1, 0.5, 0.1, 0.1), BBox(0.9, 0.5, 0.1, 0.1)),
            # Different sizes
            (BBox(0.5, 0.5, 0.2, 0.2), BBox(0.5, 0.5, 0.6, 0.6)),
            # Partial vertical overlap
            (BBox(0.5, 0.3, 0.3, 0.2), BBox(0.5, 0.5, 0.3, 0.2)),
            # Corner overlap
            (BBox(0.3, 0.3, 0.2, 0.2), BBox(0.4, 0.4, 0.2, 0.2)),
            # Tall vs wide
            (BBox(0.5, 0.5, 0.1, 0.5), BBox(0.5, 0.5, 0.5, 0.1)),
            # Near-identical
            (BBox(0.5, 0.5, 0.3, 0.3), BBox(0.51, 0.51, 0.3, 0.3)),
            # Misaligned vs stacked typical scenario
            (BBox(0.6, 0.4, 0.25, 0.15), BBox(0.6, 0.35, 0.24, 0.14)),
            # Container at edge
            (BBox(0.05, 0.5, 0.1, 0.2), BBox(0.1, 0.5, 0.1, 0.2)),
            # Small overlap
            (BBox(0.4, 0.5, 0.2, 0.2), BBox(0.55, 0.5, 0.2, 0.2)),
            # Complete containment (small inside large)
            (BBox(0.5, 0.5, 0.1, 0.1), BBox(0.5, 0.5, 0.5, 0.5)),
            # Touching edges
            (BBox(0.3, 0.5, 0.2, 0.2), BBox(0.5, 0.5, 0.2, 0.2)),
        ]

        ious = [analyzer._compute_iou(a, b) for a, b in test_pairs]

        output_path = output_dir / "container_iou_distribution.png"
        plot_iou_distribution(
            ious=ious,
            output_path=output_path,
            title="Container Detection IoU Distribution (Test Cases)",
            threshold=0.5,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_annotated_detection_frames(self, output_dir):
        """
        Generate visual output: annotated sample frames showing bounding boxes
        for container detections. Saves to tests/output/container_detection_frames.png.
        """
        from tests.visual_helpers import plot_annotated_frame

        # Create a synthetic frame (gray with noise simulating a yard scene)
        rng = np.random.default_rng(42)
        frame = rng.integers(80, 180, size=(640, 640, 3), dtype=np.uint8)

        # Simulated container detections in a typical yard scene
        detections = [
            {
                "bbox": {"x_center": 0.3, "y_center": 0.4, "width": 0.25, "height": 0.15},
                "class_label": "Container - Misaligned",
                "confidence": 0.82,
            },
            {
                "bbox": {"x_center": 0.7, "y_center": 0.35, "width": 0.24, "height": 0.14},
                "class_label": "Container - Stacked",
                "confidence": 0.91,
            },
            {
                "bbox": {"x_center": 0.5, "y_center": 0.6, "width": 0.2, "height": 0.18},
                "class_label": "Container - Open",
                "confidence": 0.74,
            },
            {
                "bbox": {"x_center": 0.5, "y_center": 0.2, "width": 0.2, "height": 0.12},
                "class_label": "Container - Picked",
                "confidence": 0.85,
            },
            {
                "bbox": {"x_center": 0.5, "y_center": 0.12, "width": 0.3, "height": 0.2},
                "class_label": "Crane",
                "confidence": 0.95,
            },
            {
                "bbox": {"x_center": 0.15, "y_center": 0.7, "width": 0.15, "height": 0.45},
                "class_label": "Container - Stacked",
                "confidence": 0.88,
            },
        ]

        output_path = output_dir / "container_detection_frames.png"
        plot_annotated_frame(
            frame=frame,
            detections=detections,
            output_path=output_path,
            title="Container Detection Annotated Frame (Test Sample)",
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
