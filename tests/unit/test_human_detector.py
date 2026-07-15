"""
Unit tests for the Human Detector module.

Tests cover:
- Property 2: Human zone violation detection
- Property 3: Temporal classification of zone violations
- Property 4: PPE violation regardless of zone
- Property 5: Confidence threshold filtering
- Detections outside all zones treated as no-people zone
- Visual outputs: annotated frames with zone overlays, confidence distribution

**Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9**
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.human_detector import HumanDetector
from hazard_detection.models import (
    BBox,
    Detection,
    HazardEvent,
    HumanDetectorConfig,
    ZonePolygon,
)
from hazard_detection.zone_map import ZoneMap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zone_map_with_zones():
    """ZoneMap with no-people (left), operation (middle), dropoff (right)."""
    zm = ZoneMap()
    # Manually inject zones for cam_01
    zm._zones["cam_01"] = [
        ZonePolygon(
            vertices=[(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
            zone_type="no_people",
            camera_id="cam_01",
        ),
        ZonePolygon(
            vertices=[(0.4, 0.0), (0.7, 0.0), (0.7, 1.0), (0.4, 1.0)],
            zone_type="operation",
            camera_id="cam_01",
        ),
        ZonePolygon(
            vertices=[(0.7, 0.0), (1.0, 0.0), (1.0, 1.0), (0.7, 1.0)],
            zone_type="dropoff",
            camera_id="cam_01",
        ),
    ]
    return zm


@pytest.fixture
def zone_map_no_config():
    """ZoneMap with no zones configured — entire FOV defaults to no-people."""
    return ZoneMap()


@pytest.fixture
def default_config():
    """HumanDetectorConfig with default threshold 0.5."""
    return HumanDetectorConfig(confidence_threshold=0.5)


@pytest.fixture
def detector(zone_map_with_zones, default_config):
    """HumanDetector with standard zone map and config."""
    return HumanDetector(zone_map=zone_map_with_zones, config=default_config)


@pytest.fixture
def detector_no_zones(zone_map_no_config, default_config):
    """HumanDetector with no zone config (entire FOV = no-people)."""
    return HumanDetector(zone_map=zone_map_no_config, config=default_config)


def _make_detection(
    x_center: float = 0.5,
    y_center: float = 0.5,
    confidence: float = 0.8,
    class_label: str = "Human",
) -> Detection:
    """Helper to create a Detection with given parameters."""
    return Detection(
        bbox=BBox(x_center=x_center, y_center=y_center, width=0.08, height=0.2),
        class_label=class_label,
        confidence=confidence,
    )


# ===========================================================================
# Property 2: Human zone violation detection
# **Validates: Requirements 2.2, 2.4, 2.9**
# ===========================================================================


class TestZoneViolationDetection:
    """Property 2: person in no-people zone emits zone_violation,
    person in operation/dropoff does not."""

    def test_person_in_no_people_zone_emits_zone_violation(self, detector):
        """Person at (0.2, 0.5) is in no-people zone → zone_violation emitted."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        # 2 consecutive frames for confirmation
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) >= 1
        assert zone_events[0].is_hazard is True

    def test_person_in_operation_zone_no_zone_violation(self, detector):
        """Person at (0.55, 0.5) is in operation zone → no zone_violation."""
        det = _make_detection(x_center=0.55, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 0

    def test_person_in_dropoff_zone_no_zone_violation(self, detector):
        """Person at (0.85, 0.5) is in dropoff zone → no zone_violation."""
        det = _make_detection(x_center=0.85, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 0

    def test_detection_outside_all_zones_treated_as_no_people(
        self, zone_map_with_zones, default_config
    ):
        """Requirement 2.9: Detection outside all defined zones → no-people zone.

        We create a zone map with a gap (zones only cover x in [0.0, 0.4] and
        [0.4, 1.0] but y only covers [0.0, 0.8]) so a point at y=0.95 is outside.
        """
        # Create a zone map with partial coverage
        zm = ZoneMap()
        zm._zones["cam_02"] = [
            ZonePolygon(
                vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                zone_type="operation",
                camera_id="cam_02",
            ),
        ]
        det_outside = HumanDetector(zone_map=zm, config=default_config)

        # Point at (0.5, 0.8) is outside the operation zone (y>0.5)
        det = _make_detection(x_center=0.5, y_center=0.8, confidence=0.9)
        detections_per_frame = [[det], [det]]
        events = det_outside.analyze(detections_per_frame, "cam_02")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) >= 1
        assert zone_events[0].is_hazard is True

    def test_no_zones_configured_entire_fov_is_no_people(self, detector_no_zones):
        """No zone config → entire FOV is no-people zone → zone_violation."""
        det = _make_detection(x_center=0.7, y_center=0.3, confidence=0.75)
        detections_per_frame = [[det], [det]]
        events = detector_no_zones.analyze(detections_per_frame, "cam_99")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) >= 1
        assert zone_events[0].is_hazard is True


# ===========================================================================
# Property 3: Temporal classification of zone violations
# **Validates: Requirements 2.6, 2.7**
# ===========================================================================


class TestTemporalClassification:
    """Property 3: 1 frame = transient (is_hazard=False),
    >=2 consecutive = confirmed (is_hazard=True)."""

    def test_single_frame_is_transient(self, detector):
        """Detection in only 1 frame → is_hazard=False (transient)."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        # Only 1 frame with detection
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1
        assert zone_events[0].is_hazard is False
        assert zone_events[0].metadata.frames_detected == 1

    def test_two_consecutive_frames_is_confirmed(self, detector):
        """Detection in 2 consecutive frames → is_hazard=True (confirmed)."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1
        assert zone_events[0].is_hazard is True
        assert zone_events[0].metadata.frames_detected == 2

    def test_three_consecutive_frames_is_confirmed(self, detector):
        """Detection in 3 consecutive frames → is_hazard=True."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1
        assert zone_events[0].is_hazard is True
        assert zone_events[0].metadata.frames_detected == 3

    def test_non_consecutive_frames_are_transient(self, detector):
        """Detection in frames 0 and 2 (gap at frame 1) → two transient events."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        # Frame 0: detection, Frame 1: empty, Frame 2: detection
        detections_per_frame = [[det], [], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        # Each isolated frame produces a transient event
        assert len(zone_events) == 2
        assert all(e.is_hazard is False for e in zone_events)
        assert all(e.metadata.frames_detected == 1 for e in zone_events)

    def test_mixed_consecutive_and_gap(self, detector):
        """Frames 0-1 consecutive, gap at 2, frame 3 alone → 1 confirmed + 1 transient."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det], [], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 2
        # First event (frames 0-1): confirmed
        confirmed = [e for e in zone_events if e.is_hazard is True]
        transient = [e for e in zone_events if e.is_hazard is False]
        assert len(confirmed) == 1
        assert len(transient) == 1
        assert confirmed[0].metadata.frames_detected == 2
        assert transient[0].metadata.frames_detected == 1


# ===========================================================================
# Property 4: PPE violation regardless of zone
# **Validates: Requirements 2.8**
# ===========================================================================


class TestPPEViolation:
    """Property 4: 'Human - No Safety Clothes' always emits ppe_violation
    with is_hazard=True regardless of zone."""

    def test_ppe_violation_in_no_people_zone(self, detector):
        """PPE violation in no-people zone → ppe_violation emitted."""
        det = _make_detection(
            x_center=0.2, y_center=0.5, confidence=0.7,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        assert len(ppe_events) == 1
        assert ppe_events[0].is_hazard is True

    def test_ppe_violation_in_operation_zone(self, detector):
        """PPE violation in operation zone → ppe_violation emitted."""
        det = _make_detection(
            x_center=0.55, y_center=0.5, confidence=0.65,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        assert len(ppe_events) == 1
        assert ppe_events[0].is_hazard is True

    def test_ppe_violation_in_dropoff_zone(self, detector):
        """PPE violation in dropoff zone → ppe_violation emitted."""
        det = _make_detection(
            x_center=0.85, y_center=0.5, confidence=0.6,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        assert len(ppe_events) == 1
        assert ppe_events[0].is_hazard is True

    def test_ppe_violation_single_frame_is_hazard(self, detector):
        """PPE violations do NOT require temporal confirmation.
        Even a single frame is is_hazard=True."""
        det = _make_detection(
            x_center=0.2, y_center=0.5, confidence=0.8,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        assert len(ppe_events) == 1
        assert ppe_events[0].is_hazard is True

    def test_ppe_violation_below_threshold_not_emitted(self, detector):
        """PPE detection below confidence threshold → no ppe_violation event."""
        det = _make_detection(
            x_center=0.2, y_center=0.5, confidence=0.3,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        assert len(ppe_events) == 0


# ===========================================================================
# Property 5: Confidence threshold filtering
# **Validates: Requirements 2.3, 2.5**
# ===========================================================================


class TestConfidenceThresholdFiltering:
    """Property 5: below-threshold detections logged but no event emitted."""

    def test_below_threshold_no_zone_violation(self, detector):
        """Confidence 0.3 < 0.5 threshold → no zone_violation emitted."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.3)
        detections_per_frame = [[det], [det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 0

    def test_at_threshold_emits_event(self, detector):
        """Confidence exactly at 0.5 threshold → event emitted."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.5)
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1
        assert zone_events[0].is_hazard is True

    def test_above_threshold_emits_event(self, detector):
        """Confidence 0.9 > 0.5 threshold → event emitted."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.9)
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1

    def test_custom_threshold_filters_correctly(self, zone_map_with_zones):
        """Custom threshold 0.7 → detections at 0.6 are not emitted."""
        config = HumanDetectorConfig(confidence_threshold=0.7)
        det_high_thresh = HumanDetector(zone_map=zone_map_with_zones, config=config)

        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.6)
        detections_per_frame = [[det], [det], [det]]
        events = det_high_thresh.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 0

    def test_mixed_confidence_only_above_threshold_counted(self, detector):
        """Mix of above and below threshold → only above-threshold contribute."""
        det_high = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        det_low = _make_detection(x_center=0.2, y_center=0.5, confidence=0.3)
        # Frame 0: above threshold, Frame 1: below threshold, Frame 2: above
        detections_per_frame = [[det_high], [det_low], [det_high]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        # Consecutive streak broken by below-threshold frame
        # → two transient events (each 1 frame)
        assert all(e.is_hazard is False for e in zone_events)


# ===========================================================================
# Additional edge cases
# ===========================================================================


class TestEdgeCases:
    """Additional edge cases for human detection."""

    def test_empty_frame_sequence(self, detector):
        """No frames at all → no events."""
        events = detector.analyze([], "cam_01")
        assert len(events) == 0

    def test_frames_with_no_human_detections(self, detector):
        """Frames with non-human detections → no events."""
        det = Detection(
            bbox=BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.15),
            class_label="Container - Stacked",
            confidence=0.9,
        )
        detections_per_frame = [[det], [det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")
        assert len(events) == 0

    def test_both_ppe_and_zone_violation_emitted(self, detector):
        """'Human - No Safety Clothes' in no-people zone → both events."""
        det = _make_detection(
            x_center=0.2, y_center=0.5, confidence=0.8,
            class_label="Human - No Safety Clothes",
        )
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        ppe_events = [e for e in events if e.hazard_type == "ppe_violation"]
        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(ppe_events) >= 1
        assert len(zone_events) >= 1

    def test_hazard_event_has_correct_confidence(self, detector):
        """Emitted event contains the detection's confidence score."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.77)
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        assert len(zone_events) == 1
        assert zone_events[0].confidence == 0.77

    def test_hazard_event_validates_successfully(self, detector):
        """All emitted events pass HazardEvent.validate()."""
        det = _make_detection(x_center=0.2, y_center=0.5, confidence=0.8)
        detections_per_frame = [[det], [det]]
        events = detector.analyze(detections_per_frame, "cam_01")

        for event in events:
            assert event.validate() is True


# ===========================================================================
# Visual output generation
# ===========================================================================


class TestVisualOutputs:
    """Generate visual diagnostic outputs for human detection tests."""

    def test_generate_annotated_frame_with_zones(self, detector, output_dir):
        """Generate annotated sample frames showing bounding boxes and zone overlays.

        Saves to tests/output/human_detection_zones.png
        """
        from tests.visual_helpers import plot_annotated_frame

        # Create a sample frame (dark background with some structure)
        rng = np.random.default_rng(42)
        frame = rng.integers(30, 80, size=(640, 640, 3), dtype=np.uint8)

        # Define zones for visualization
        zones = [
            {
                "vertices": [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
                "zone_type": "no_people",
            },
            {
                "vertices": [(0.4, 0.0), (0.7, 0.0), (0.7, 1.0), (0.4, 1.0)],
                "zone_type": "operation",
            },
            {
                "vertices": [(0.7, 0.0), (1.0, 0.0), (1.0, 1.0), (0.7, 1.0)],
                "zone_type": "dropoff",
            },
        ]

        # Sample detections across different zones
        detections = [
            {
                "bbox": {"x_center": 0.2, "y_center": 0.5, "width": 0.08, "height": 0.2},
                "class_label": "Human",
                "confidence": 0.85,
            },
            {
                "bbox": {"x_center": 0.15, "y_center": 0.3, "width": 0.07, "height": 0.22},
                "class_label": "Human - No Safety Clothes",
                "confidence": 0.72,
            },
            {
                "bbox": {"x_center": 0.55, "y_center": 0.6, "width": 0.08, "height": 0.2},
                "class_label": "Human",
                "confidence": 0.91,
            },
            {
                "bbox": {"x_center": 0.85, "y_center": 0.4, "width": 0.07, "height": 0.18},
                "class_label": "Human",
                "confidence": 0.65,
            },
        ]

        output_path = output_dir / "human_detection_zones.png"
        plot_annotated_frame(
            frame=frame,
            detections=detections,
            zones=zones,
            output_path=output_path,
            title="Human Detection with Zone Overlays",
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_confidence_distribution(self, detector, output_dir):
        """Generate confidence distribution plot for human detections.

        Saves to tests/output/human_confidence_distribution.png
        """
        from tests.visual_helpers import plot_confidence_histogram

        # Simulate a range of confidence scores for human detections
        rng = np.random.default_rng(123)
        # Bimodal distribution: some low-confidence, some high-confidence
        low_conf = rng.uniform(0.1, 0.45, size=40)
        high_conf = rng.uniform(0.55, 0.98, size=60)
        all_confidences = np.concatenate([low_conf, high_conf]).tolist()

        output_path = output_dir / "human_confidence_distribution.png"
        plot_confidence_histogram(
            confidences=all_confidences,
            output_path=output_path,
            title="Human Detection Confidence Distribution",
            threshold=0.5,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
