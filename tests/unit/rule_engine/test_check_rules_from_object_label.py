"""
Unit tests for check_rules_from_object_label.py's five check functions:
check_human, check_container, check_vehicle, check_tel_spot,
check_tel_occupancy.

Requirements covered: 3.1-3.4, 4.1-4.4, 5.1-5.6, 6.1-6.7
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.models import BBox, Detection, DiagnosticMetadata, HazardEvent
from hazard_detection.rule_engine.rules import DEFAULT_RULES, LocationRuleSet, PPERequirement
from hazard_detection.rule_engine.check_rules_from_object_label import (
    check_container,
    check_human,
    check_tel_occupancy,
    check_tel_spot,
    check_vehicle,
)


def _detection(class_label: str, x_center: float = 0.5, y_center: float = 0.5) -> Detection:
    return Detection(
        bbox=BBox(x_center=x_center, y_center=y_center, width=0.1, height=0.2),
        class_label=class_label,
        confidence=0.9,
    )


def _hazard_event(hazard_type: str) -> HazardEvent:
    return HazardEvent(
        event_id=str(uuid.uuid4()),
        hazard_type=hazard_type,
        camera_id="cam_01",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        is_hazard=True,
        confidence=0.9,
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.2),
        metadata=DiagnosticMetadata(frame_index=0, detection_class="Container - Open", frames_detected=2),
    )


# ---------------------------------------------------------------------------
# check_human() — Requirement 3.1, 3.2, 3.3, 3.4
# ---------------------------------------------------------------------------

class TestCheckHuman:
    def test_unknown_policy_always_none_for_human(self):
        rule_set = DEFAULT_RULES["VACIS"]
        assert check_human(_detection("Human"), rule_set) is None

    def test_unknown_policy_always_none_for_no_ppe(self):
        rule_set = DEFAULT_RULES["VACIS"]
        assert check_human(_detection("Human - No Safety Clothes"), rule_set) is None

    def test_prohibited_policy_flags_human(self):
        rule_set = DEFAULT_RULES["Block"]
        result = check_human(_detection("Human"), rule_set)
        assert result == ("zone_violation", "human_presence_prohibited")

    def test_prohibited_policy_flags_no_ppe_too(self):
        rule_set = DEFAULT_RULES["Block"]
        result = check_human(_detection("Human - No Safety Clothes"), rule_set)
        assert result == ("zone_violation", "human_presence_prohibited")

    def test_permitted_policy_no_ppe_with_vest_required_flags_ppe_violation(self):
        rule_set = DEFAULT_RULES["Berth"]  # vest_required=True
        result = check_human(_detection("Human - No Safety Clothes"), rule_set)
        assert result == ("ppe_violation", "ppe_vest_missing")

    def test_permitted_policy_human_compliant_no_event(self):
        rule_set = DEFAULT_RULES["Berth"]
        assert check_human(_detection("Human"), rule_set) is None

    def test_conditional_policy_no_ppe_flags_ppe_violation(self):
        rule_set = DEFAULT_RULES["Airlocks"]  # conditional, vest_required=True
        result = check_human(_detection("Human - No Safety Clothes"), rule_set)
        assert result == ("ppe_violation", "ppe_vest_missing")

    def test_conditional_policy_human_presence_alone_no_event(self):
        rule_set = DEFAULT_RULES["Airlocks"]
        assert check_human(_detection("Human"), rule_set) is None

    def test_non_human_class_returns_none(self):
        rule_set = DEFAULT_RULES["Block"]
        assert check_human(_detection("Vehicle"), rule_set) is None

    def test_permitted_policy_no_vest_requirement_no_ppe_violation(self):
        # Construct a rule set with permitted policy but vest not required,
        # to confirm the ppe_violation is gated on vest_required.
        rule_set = LocationRuleSet(
            location_type="TestLoc",
            human_presence_policy="permitted",
            ppe_requirement=PPERequirement(vest_required=False),
        )
        assert check_human(_detection("Human - No Safety Clothes"), rule_set) is None


# ---------------------------------------------------------------------------
# check_container() — Requirement 4.1, 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------

class TestCheckContainer:
    def test_enabled_and_not_suppressed_is_kept(self):
        rule_set = DEFAULT_RULES["Berth"]  # open_doors enabled, not suppressed
        event = _hazard_event("container_door_open")
        result = check_container(_detection("Container - Open"), rule_set, event)
        assert result == ("container_door_open", "container_open_doors_check")

    def test_rail_storage_suppresses_open_doors(self):
        rule_set = DEFAULT_RULES["Rail Storage"]
        event = _hazard_event("container_door_open")
        result = check_container(_detection("Container - Open"), rule_set, event)
        assert result is None

    def test_flipline_suppresses_open_doors(self):
        rule_set = DEFAULT_RULES["Flipline"]
        event = _hazard_event("container_door_open")
        result = check_container(_detection("Container - Open"), rule_set, event)
        assert result is None

    def test_rail_storage_still_flags_misalignment(self):
        rule_set = DEFAULT_RULES["Rail Storage"]
        event = _hazard_event("container_misalignment")
        result = check_container(_detection("Container - Misaligned"), rule_set, event)
        assert result == ("container_misalignment", "container_misalignment_check")

    def test_not_enabled_check_is_dropped(self):
        rule_set = DEFAULT_RULES["AssetManagement"]  # no container checks enabled
        event = _hazard_event("container_misalignment")
        result = check_container(_detection("Container - Misaligned"), rule_set, event)
        assert result is None

    def test_none_analyzer_result_returns_none(self):
        rule_set = DEFAULT_RULES["Berth"]
        assert check_container(_detection("Container - Open"), rule_set, None) is None

    def test_non_container_hazard_type_returns_none(self):
        rule_set = DEFAULT_RULES["Berth"]
        event = _hazard_event("zone_violation")
        assert check_container(_detection("Human"), rule_set, event) is None


# ---------------------------------------------------------------------------
# check_vehicle() — Requirement 5.1, 5.2, 5.3, 5.5, 5.6
# ---------------------------------------------------------------------------

class TestCheckVehicle:
    def test_at_or_above_threshold_flags_event(self):
        rule_set = DEFAULT_RULES["Berth"]  # vehicle_checks_enabled=True
        result = check_vehicle(_detection("Vehicle", y_center=0.9), rule_set, unsafe_y_threshold=0.8)
        assert result == ("vehicle_proximity", "vehicle_proximity_check")

    def test_below_threshold_returns_none(self):
        rule_set = DEFAULT_RULES["Berth"]
        result = check_vehicle(_detection("Vehicle", y_center=0.3), rule_set, unsafe_y_threshold=0.8)
        assert result is None

    def test_disabled_rule_set_always_none_regardless_of_position(self):
        rule_set = DEFAULT_RULES["Reefer Rack"]  # vehicle_checks_enabled=False
        result = check_vehicle(_detection("Vehicle", y_center=0.99), rule_set, unsafe_y_threshold=0.1)
        assert result is None

    def test_non_vehicle_class_returns_none(self):
        rule_set = DEFAULT_RULES["Berth"]
        result = check_vehicle(_detection("Human", y_center=0.9), rule_set, unsafe_y_threshold=0.1)
        assert result is None

    def test_exactly_at_threshold_is_flagged(self):
        rule_set = DEFAULT_RULES["Berth"]
        result = check_vehicle(_detection("Truck - With Container", y_center=0.8), rule_set, unsafe_y_threshold=0.8)
        assert result == ("vehicle_proximity", "vehicle_proximity_check")


# ---------------------------------------------------------------------------
# check_tel_spot() — Requirement 6.1, 6.2, 6.3, 6.4, 6.5
# ---------------------------------------------------------------------------

class TestCheckTelSpot:
    SQUARE_SPOT = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]

    def test_inside_polygon_returns_none(self):
        det = _detection("Human", x_center=0.5, y_center=0.5)
        assert check_tel_spot(det, self.SQUARE_SPOT) is None

    def test_outside_polygon_flags_event(self):
        det = _detection("Human", x_center=0.1, y_center=0.1)
        result = check_tel_spot(det, self.SQUARE_SPOT)
        assert result == ("tel_zone_violation", "tel_trucker_spot")

    def test_none_polygon_is_failsafe_flagged(self):
        det = _detection("Human", x_center=0.5, y_center=0.5)
        result = check_tel_spot(det, None)
        assert result == ("tel_zone_violation", "tel_trucker_spot")


# ---------------------------------------------------------------------------
# check_tel_occupancy() — Requirement 6.6, 6.7
# ---------------------------------------------------------------------------

class TestCheckTelOccupancy:
    def test_at_limit_returns_none(self):
        rule_set = DEFAULT_RULES["TELs"]  # occupancy_limit=1
        detections = [_detection("Human")]
        assert check_tel_occupancy(detections, rule_set) is None

    def test_over_limit_flags_event(self):
        rule_set = DEFAULT_RULES["TELs"]
        detections = [_detection("Human"), _detection("Human - No Safety Clothes")]
        result = check_tel_occupancy(detections, rule_set)
        assert result == ("tel_occupancy_violation", "tel_occupancy_limit")

    def test_no_occupancy_limit_configured_returns_none(self):
        rule_set = DEFAULT_RULES["Berth"]  # occupancy_limit=None
        detections = [_detection("Human"), _detection("Human"), _detection("Human")]
        assert check_tel_occupancy(detections, rule_set) is None

    def test_maintenance_exception_has_no_effect(self):
        # Confirm behavior is IDENTICAL whether or not
        # occupancy_maintenance_exception is True — there is no detection
        # signal for "maintenance is occurring" today (Requirement 6.7).
        rule_set_with_exception = DEFAULT_RULES["TELs"]
        assert rule_set_with_exception.occupancy_maintenance_exception is True

        detections = [_detection("Human"), _detection("Human - No Safety Clothes")]
        result_with_exception_flag_true = check_tel_occupancy(detections, rule_set_with_exception)

        # Build an identical rule set but with the exception flag off.
        rule_set_without_exception = LocationRuleSet(
            location_type="TELs",
            human_presence_policy="permitted",
            occupancy_limit=1,
            occupancy_maintenance_exception=False,
        )
        result_without_exception_flag = check_tel_occupancy(detections, rule_set_without_exception)

        assert result_with_exception_flag_true == result_without_exception_flag
        assert result_with_exception_flag_true == ("tel_occupancy_violation", "tel_occupancy_limit")

    def test_non_human_detections_not_counted(self):
        rule_set = DEFAULT_RULES["TELs"]
        detections = [_detection("Vehicle"), _detection("Truck - With Container")]
        assert check_tel_occupancy(detections, rule_set) is None
