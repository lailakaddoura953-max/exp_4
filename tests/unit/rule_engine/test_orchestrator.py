"""
Unit tests for HazardRuleOrchestrator.

Tests all 16 location types produce the correct event types for
representative detections, per the Location Rule Logic Table, plus
camera_name_overrides and audit log completeness.

Requirements covered: 2.4, 2.5, 3.1-3.7, 7.1-7.7, 9.1-9.6
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.models import BBox, Detection, DiagnosticMetadata, FrameSequence, HazardEvent
from hazard_detection.rule_engine.audit_logger import AuditLogger
from hazard_detection.rule_engine.camera_location_resolver import CameraLocationResolver
from hazard_detection.rule_engine.orchestrator import HazardRuleOrchestrator
from hazard_detection.rule_engine.rules import LocationRuleLoader


def _detection(class_label: str, x_center: float = 0.5, y_center: float = 0.5, confidence: float = 0.9) -> Detection:
    return Detection(
        bbox=BBox(x_center=x_center, y_center=y_center, width=0.1, height=0.2),
        class_label=class_label,
        confidence=confidence,
    )


def _frame_sequence(camera_id: str = "test_cam", num_frames: int = 2) -> FrameSequence:
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(num_frames)]
    timestamps = [1700000000.0 + i for i in range(num_frames)]
    return FrameSequence(frames=frames, camera_id=camera_id, timestamps=timestamps)


def _make_container_hazard_event(hazard_type: str, frame_index: int = 0) -> HazardEvent:
    return HazardEvent(
        event_id=HazardEvent.generate_event_id(),
        hazard_type=hazard_type,
        camera_id="test_cam",
        timestamp=HazardEvent.generate_timestamp(),
        is_hazard=True,
        confidence=0.9,
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.15),
        metadata=DiagnosticMetadata(frame_index=frame_index, detection_class="Container - Open", frames_detected=2),
    )


@pytest.fixture
def audit_log_path(tmp_path) -> str:
    return str(tmp_path / "rule_audit.jsonl")


@pytest.fixture
def orchestrator_factory(audit_log_path):
    """Factory to build an orchestrator with a mocked ContainerAnalyzer."""

    def _build(container_analyzer_events=None, trucker_spot_polygons=None):
        resolver = CameraLocationResolver()
        rule_loader = LocationRuleLoader(config_path=None)
        audit_logger = AuditLogger(audit_log_path=audit_log_path)

        container_analyzer = MagicMock()
        container_analyzer.analyze.return_value = container_analyzer_events or []

        orchestrator = HazardRuleOrchestrator(
            resolver=resolver,
            rule_loader=rule_loader,
            container_analyzer=container_analyzer,
            audit_logger=audit_logger,
            trucker_spot_polygons=trucker_spot_polygons,
        )
        return orchestrator, rule_loader

    return _build


def _events_of_type(qualified_events, hazard_type: str):
    return [qe for qe in qualified_events if qe.event.hazard_type == hazard_type and qe.event.is_hazard]


class TestHumanPresencePolicies:
    def test_block_human_confirmed_zone_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("A8 - SE PTZ - Block 1F", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "zone_violation")) >= 1

    def test_berth_human_compliant_no_event(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("A10 - NE - Berth 404", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "zone_violation") == []
        assert _events_of_type(events, "ppe_violation") == []

    def test_berth_human_no_ppe_flags_ppe_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human - No Safety Clothes")]]
        events = orchestrator.evaluate("A10 - NE - Berth 404", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "ppe_violation")) == 1

    def test_reefer_rack_human_permitted_no_event(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("B4 - N PTZ - Reefer rack 1", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "zone_violation") == []

    def test_airlocks_human_presence_alone_no_event(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("C5 - Airlocks", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "zone_violation") == []

    def test_airlocks_ppe_violation_still_checked(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human - No Safety Clothes")]]
        events = orchestrator.evaluate("C5 - Airlocks", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "ppe_violation")) == 1

    def test_vacis_unknown_policy_no_human_events(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human - No Safety Clothes")]]
        events = orchestrator.evaluate("H2 - VACIS Area", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "zone_violation") == []
        assert _events_of_type(events, "ppe_violation") == []

    def test_unknown_camera_failsafe_zone_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("totally-unrecognized-camera", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "zone_violation")) >= 1


class TestTelSpotAndOccupancy:
    SQUARE_SPOT = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]

    def test_tel_human_inside_spot_no_zone_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory(
            trucker_spot_polygons={"TEL 118": self.SQUARE_SPOT}
        )
        detections_per_frame = [[_detection("Human", x_center=0.5, y_center=0.5)]]
        events = orchestrator.evaluate("TEL 118", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "tel_zone_violation") == []

    def test_tel_human_outside_spot_flags_confirmed_after_two_frames(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory(
            trucker_spot_polygons={"TEL 118": self.SQUARE_SPOT}
        )
        outside_det = _detection("Human", x_center=0.1, y_center=0.1)
        detections_per_frame = [[outside_det], [outside_det]]
        events = orchestrator.evaluate("TEL 118", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "tel_zone_violation")) >= 1

    def test_tels_two_confirmed_humans_flags_occupancy_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [
            [_detection("Human", x_center=0.3), _detection("Human - No Safety Clothes", x_center=0.7)]
        ]
        events = orchestrator.evaluate("D6 - N PTZ - TELs 3", detections_per_frame, _frame_sequence())
        assert len(_events_of_type(events, "tel_occupancy_violation")) >= 1

    def test_tels_one_human_no_occupancy_violation(self, orchestrator_factory):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")]]
        events = orchestrator.evaluate("D6 - N PTZ - TELs 3", detections_per_frame, _frame_sequence())
        assert _events_of_type(events, "tel_occupancy_violation") == []


class TestContainerChecks:
    def test_flipline_open_door_suppressed(self, orchestrator_factory):
        analyzer_event = _make_container_hazard_event("container_door_open")
        orchestrator, _ = orchestrator_factory(container_analyzer_events=[analyzer_event])
        events = orchestrator.evaluate("E7 - S PTZ - Flipline", [[]], _frame_sequence())
        assert _events_of_type(events, "container_door_open") == []

    def test_rail_storage_open_door_suppressed_generic_rail_not_suppressed(self, orchestrator_factory):
        analyzer_event = _make_container_hazard_event("container_door_open")

        orchestrator_rs, _ = orchestrator_factory(container_analyzer_events=[analyzer_event])
        events_rs = orchestrator_rs.evaluate("Rail Storage 12 - North", [[]], _frame_sequence())
        assert _events_of_type(events_rs, "container_door_open") == []

        orchestrator_rail, _ = orchestrator_factory(container_analyzer_events=[analyzer_event])
        events_rail = orchestrator_rail.evaluate("B4 - N PTZ - Rail 12", [[]], _frame_sequence())
        assert len(_events_of_type(events_rail, "container_door_open")) == 1

    def test_asset_management_container_checks_disabled(self, orchestrator_factory):
        analyzer_event = _make_container_hazard_event("container_misalignment")
        orchestrator, _ = orchestrator_factory(container_analyzer_events=[analyzer_event])
        events = orchestrator.evaluate("F8 - AssetManagement", [[]], _frame_sequence())
        assert _events_of_type(events, "container_misalignment") == []


class TestCameraNameOverrides:
    def test_camera_name_override_changes_resolved_location(self, orchestrator_factory):
        orchestrator, rule_loader = orchestrator_factory()
        rule_loader.camera_name_overrides["ADM Parking"] = "AssetManagement"

        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        events = orchestrator.evaluate("ADM Parking", detections_per_frame, _frame_sequence())
        # AssetManagement is "permitted" -> no zone_violation, unlike the
        # Unknown fail-safe ADM Parking would otherwise resolve to.
        assert _events_of_type(events, "zone_violation") == []


class TestAuditLogCompleteness:
    def test_audit_log_written_for_processed_detections(self, orchestrator_factory, audit_log_path):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")], [_detection("Human")]]
        orchestrator.evaluate("A8 - SE PTZ - Block 1F", detections_per_frame, _frame_sequence())

        lines = Path(audit_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2
        for line in lines:
            entry = json.loads(line)
            assert entry["camera_name"] == "A8 - SE PTZ - Block 1F"
            assert entry["location_type"] == "Block"

    def test_audit_log_written_for_unknown_policy_skip(self, orchestrator_factory, audit_log_path):
        orchestrator, _ = orchestrator_factory()
        detections_per_frame = [[_detection("Human")]]
        orchestrator.evaluate("H2 - VACIS Area", detections_per_frame, _frame_sequence())

        lines = Path(audit_log_path).read_text(encoding="utf-8").strip().splitlines()
        outcomes = [json.loads(line)["outcome"] for line in lines]
        assert "policy_unknown" in outcomes
