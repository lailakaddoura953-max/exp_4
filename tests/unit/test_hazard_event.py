"""
Unit tests for the HazardEvent data model.

Tests cover:
- Property 13: Unique event identifiers (UUID generation uniqueness)
- Property 14: Hazard event structural completeness (mandatory field validation)
- Validation rejection for invalid fields
- Binary hazard classification (is_hazard=True/False)
- Visual output: event field coverage report as JSON

Validates: Requirements 8.1, 8.5, 8.6
"""

import uuid
from pathlib import Path

import pytest

from hazard_detection.models import (
    BBox,
    DiagnosticMetadata,
    HazardEvent,
    VALID_HAZARD_TYPES,
)
from tests.visual_helpers import save_json_report


# =============================================================================
# Property 13: Unique event identifiers
# =============================================================================


class TestUniqueEventIdentifiers:
    """
    Property 13: For any set of Hazard_Events produced during the system's
    operational lifetime, every event_id SHALL be unique.

    **Validates: Requirements 8.5**
    """

    def test_generate_10000_unique_event_ids(self):
        """Generate 10000 event IDs and assert all are unique."""
        ids = [HazardEvent.generate_event_id() for _ in range(10000)]
        assert len(set(ids)) == 10000, (
            f"Expected 10000 unique IDs, got {len(set(ids))}"
        )

    def test_event_id_is_valid_uuid4(self):
        """Generated event IDs are valid UUID4 strings."""
        event_id = HazardEvent.generate_event_id()
        parsed = uuid.UUID(event_id, version=4)
        assert str(parsed) == event_id

    def test_factory_events_have_unique_ids(self, hazard_event_factory):
        """Events created via the factory each have distinct event_ids."""
        events = [hazard_event_factory() for _ in range(100)]
        ids = [e.event_id for e in events]
        assert len(set(ids)) == 100


# =============================================================================
# Property 14: Hazard event structural completeness
# =============================================================================


class TestHazardEventStructuralCompleteness:
    """
    Property 14: Any Hazard_Event emitted SHALL contain all mandatory fields.
    Events missing a mandatory field SHALL be rejected.

    **Validates: Requirements 8.1, 8.6**
    """

    def test_valid_event_passes_validation(self, hazard_event_factory):
        """A fully populated HazardEvent passes validate()."""
        event = hazard_event_factory()
        assert event.validate() is True

    def test_valid_event_all_hazard_types(self, hazard_event_factory):
        """Valid events can be created for each recognized hazard type."""
        for hazard_type in VALID_HAZARD_TYPES:
            event = hazard_event_factory(hazard_type=hazard_type)
            assert event.validate() is True

    def test_valid_event_is_hazard_false(self, hazard_event_factory):
        """An event with is_hazard=False passes validation (binary classification)."""
        event = hazard_event_factory(is_hazard=False)
        assert event.validate() is True

    def test_valid_event_boundary_confidence(self, hazard_event_factory):
        """Events at confidence boundaries (0.0 and 1.0) pass validation."""
        event_low = hazard_event_factory(confidence=0.0)
        event_high = hazard_event_factory(confidence=1.0)
        assert event_low.validate() is True
        assert event_high.validate() is True

    # ----- Rejection: empty event_id -----

    def test_reject_empty_event_id(self, hazard_event_factory):
        """Validate rejects events with empty event_id."""
        event = hazard_event_factory()
        event.event_id = ""
        with pytest.raises(ValueError, match="event_id"):
            event.validate()

    def test_reject_whitespace_event_id(self, hazard_event_factory):
        """Validate rejects events with whitespace-only event_id."""
        event = hazard_event_factory()
        event.event_id = "   "
        with pytest.raises(ValueError, match="event_id"):
            event.validate()

    # ----- Rejection: invalid hazard_type -----

    def test_reject_invalid_hazard_type(self, hazard_event_factory):
        """Validate rejects events with unrecognized hazard_type."""
        event = hazard_event_factory(hazard_type="unknown_hazard")
        with pytest.raises(ValueError, match="hazard_type"):
            event.validate()

    def test_reject_empty_hazard_type(self, hazard_event_factory):
        """Validate rejects events with empty hazard_type."""
        event = hazard_event_factory(hazard_type="")
        with pytest.raises(ValueError, match="hazard_type"):
            event.validate()

    # ----- Rejection: empty camera_id -----

    def test_reject_empty_camera_id(self, hazard_event_factory):
        """Validate rejects events with empty camera_id."""
        event = hazard_event_factory(camera_id="")
        with pytest.raises(ValueError, match="camera_id"):
            event.validate()

    def test_reject_whitespace_camera_id(self, hazard_event_factory):
        """Validate rejects events with whitespace-only camera_id."""
        event = hazard_event_factory(camera_id="  \t  ")
        with pytest.raises(ValueError, match="camera_id"):
            event.validate()

    # ----- Rejection: invalid timestamp -----

    def test_reject_invalid_timestamp_format(self, hazard_event_factory):
        """Validate rejects events with non-ISO 8601 timestamp."""
        event = hazard_event_factory(timestamp="not-a-timestamp")
        with pytest.raises(ValueError, match="timestamp"):
            event.validate()

    def test_reject_empty_timestamp(self, hazard_event_factory):
        """Validate rejects events with empty timestamp."""
        event = hazard_event_factory()
        event.timestamp = ""
        with pytest.raises(ValueError, match="timestamp"):
            event.validate()

    # ----- Rejection: non-bool is_hazard -----

    def test_reject_non_bool_is_hazard(self, hazard_event_factory):
        """Validate rejects events where is_hazard is not a bool."""
        event = hazard_event_factory()
        event.is_hazard = 1  # int, not bool
        with pytest.raises(ValueError, match="is_hazard"):
            event.validate()

    def test_reject_string_is_hazard(self, hazard_event_factory):
        """Validate rejects events where is_hazard is a string."""
        event = hazard_event_factory()
        event.is_hazard = "true"
        with pytest.raises(ValueError, match="is_hazard"):
            event.validate()

    # ----- Rejection: out-of-range confidence -----

    def test_reject_confidence_above_one(self, hazard_event_factory):
        """Validate rejects events with confidence > 1.0."""
        event = hazard_event_factory()
        event.confidence = 1.5
        with pytest.raises(ValueError, match="confidence"):
            event.validate()

    def test_reject_confidence_below_zero(self, hazard_event_factory):
        """Validate rejects events with confidence < 0.0."""
        event = hazard_event_factory()
        event.confidence = -0.1
        with pytest.raises(ValueError, match="confidence"):
            event.validate()

    def test_reject_non_numeric_confidence(self, hazard_event_factory):
        """Validate rejects events with non-numeric confidence."""
        event = hazard_event_factory()
        event.confidence = "high"
        with pytest.raises(ValueError, match="confidence"):
            event.validate()

    # ----- Rejection: invalid bbox -----

    def test_reject_non_bbox_object(self, hazard_event_factory):
        """Validate rejects events where bbox is not a BBox instance."""
        event = hazard_event_factory()
        event.bbox = {"x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.2}
        with pytest.raises(ValueError, match="bbox"):
            event.validate()

    def test_reject_none_bbox(self, hazard_event_factory):
        """Validate rejects events where bbox is None."""
        event = hazard_event_factory()
        event.bbox = None
        with pytest.raises(ValueError, match="bbox"):
            event.validate()

    # ----- Rejection: invalid metadata -----

    def test_reject_non_metadata_object(self, hazard_event_factory):
        """Validate rejects events where metadata is not a DiagnosticMetadata instance."""
        event = hazard_event_factory()
        event.metadata = {"frame_index": 0, "detection_class": "Human"}
        with pytest.raises(ValueError, match="metadata"):
            event.validate()

    def test_reject_none_metadata(self, hazard_event_factory):
        """Validate rejects events where metadata is None."""
        event = hazard_event_factory()
        event.metadata = None
        with pytest.raises(ValueError, match="metadata"):
            event.validate()


# =============================================================================
# Binary hazard classification tests
# =============================================================================


class TestBinaryHazardClassification:
    """
    Binary classification: is_hazard=True means confirmed hazard (dispatch alert),
    is_hazard=False means not a hazard (logged only).
    """

    def test_is_hazard_true_is_valid(self, hazard_event_factory):
        """is_hazard=True creates a valid event for alert dispatch."""
        event = hazard_event_factory(is_hazard=True)
        assert event.validate() is True
        assert event.is_hazard is True

    def test_is_hazard_false_is_valid(self, hazard_event_factory):
        """is_hazard=False creates a valid event for logging only."""
        event = hazard_event_factory(is_hazard=False)
        assert event.validate() is True
        assert event.is_hazard is False


# =============================================================================
# Visual output: event field coverage report
# =============================================================================


class TestHazardEventCoverageReport:
    """Generate a coverage report summarizing which fields and validation paths are tested."""

    def test_generate_coverage_report(self, output_dir, hazard_event_factory):
        """Save event field coverage report as JSON to tests/output/."""
        # Build the coverage report
        coverage = {
            "test_file": "tests/unit/test_hazard_event.py",
            "validates_requirements": ["8.1", "8.5", "8.6"],
            "properties_tested": [
                {
                    "id": "Property 13",
                    "name": "Unique event identifiers",
                    "description": "All generated event_ids are unique across 10000 samples",
                    "test_count": 3,
                },
                {
                    "id": "Property 14",
                    "name": "Hazard event structural completeness",
                    "description": "Mandatory fields validated; invalid events rejected",
                    "test_count": 18,
                },
            ],
            "mandatory_fields_tested": {
                "event_id": {
                    "valid": True,
                    "rejection_cases": ["empty string", "whitespace only"],
                },
                "hazard_type": {
                    "valid": True,
                    "rejection_cases": ["unrecognized type", "empty string"],
                },
                "camera_id": {
                    "valid": True,
                    "rejection_cases": ["empty string", "whitespace only"],
                },
                "timestamp": {
                    "valid": True,
                    "rejection_cases": ["invalid format", "empty string"],
                },
                "is_hazard": {
                    "valid": True,
                    "rejection_cases": ["integer value", "string value"],
                },
                "confidence": {
                    "valid": True,
                    "rejection_cases": ["above 1.0", "below 0.0", "non-numeric"],
                },
                "bbox": {
                    "valid": True,
                    "rejection_cases": ["dict instead of BBox", "None"],
                },
                "metadata": {
                    "valid": True,
                    "rejection_cases": ["dict instead of DiagnosticMetadata", "None"],
                },
            },
            "binary_classification_tested": {
                "is_hazard_true": "confirmed hazard, alert dispatched",
                "is_hazard_false": "not a hazard, logged only",
            },
            "valid_hazard_types": sorted(VALID_HAZARD_TYPES),
        }

        report_path = output_dir / "hazard_event_coverage.json"
        save_json_report(coverage, report_path)

        assert report_path.exists()
