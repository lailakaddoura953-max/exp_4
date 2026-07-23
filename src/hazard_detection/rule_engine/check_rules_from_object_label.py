"""
Per-Detection Rule Checks for the Camera-Location-Aware Hazard Rules engine.

This file answers "given this rule set and this detection, is it a
hazard?" and contains no camera-name parsing or YAML loading of its own —
per the module-structure split requested for this project (see
requirements.md Requirement 9.2 and design.md's Overview).

None of the functions here apply confidence-threshold or temporal
(>=2 consecutive frames) filtering themselves — that responsibility stays
with the caller (the orchestrator, delegating to the existing
HumanDetector/ContainerAnalyzer confirmation logic), matching how the rest
of the pipeline already works.

Requirements covered: 3.1-3.4, 4.1-4.4, 5.1-5.6, 6.1-6.7
"""

from typing import List, Optional, Tuple

from hazard_detection.diagnostics import get_logger
from hazard_detection.models import BBox, Detection, HazardEvent
from hazard_detection.rule_engine.rules import LocationRuleSet
from hazard_detection.zone_map import ZoneMap

logger = get_logger("rule_engine.check_rules_from_object_label")

# Human detection class labels (matches human_detector.py's HUMAN_CLASSES).
HUMAN_CLASS = "Human"
PPE_VIOLATION_CLASS = "Human - No Safety Clothes"
HUMAN_CLASSES = {HUMAN_CLASS, PPE_VIOLATION_CLASS}

# Vehicle detection class labels that can trigger vehicle_proximity checks.
VEHICLE_CLASSES = {"Truck - No Container", "Truck - With Container", "Vehicle"}


# ============================================================================
# check_human() — Requirement 3.1, 3.2, 3.3, 3.4
# ============================================================================


def check_human(
    detection: Detection, rule_set: LocationRuleSet
) -> Optional[Tuple[str, str]]:
    """
    Check one human-class detection against a rule set's presence policy
    and PPE requirement.

    Behavior by human_presence_policy:
      - "unknown": always returns None. The caller is responsible for
        logging this as a skipped/policy_unknown outcome (Requirement 3.4
        / design.md's "unknown policy produces zero human events").
      - "prohibited": ANY human-class detection (Human or
        Human - No Safety Clothes) -> ("zone_violation",
        "human_presence_prohibited"), regardless of PPE state
        (Requirement 3.1).
      - "permitted" / "conditional":
          - detection.class_label == "Human - No Safety Clothes" AND
            rule_set.ppe_requirement.vest_required -> ("ppe_violation",
            "ppe_vest_missing") (Requirement 3.2).
          - detection.class_label == "Human" -> None. Vest/coveralls PPE
            is implied present by this class; shoes/helmet are NOT
            verifiable from either human class alone with today's
            Detection_Class set, so no pass/fail signal is fabricated for
            them here (Requirement 3.3, 3.4).

    Does NOT apply confidence/temporal filtering — that stays the
    caller's responsibility, matching existing HumanDetector behavior.

    Args:
        detection: A Detection with class_label in HUMAN_CLASSES.
        rule_set: The resolved LocationRuleSet for the camera.

    Returns:
        (hazard_type, matched_rule) tuple if this detection is a hazard
        under the given rule set, otherwise None.
    """
    if detection.class_label not in HUMAN_CLASSES:
        return None

    policy = rule_set.human_presence_policy

    if policy == "unknown":
        return None

    if policy == "prohibited":
        return ("zone_violation", "human_presence_prohibited")

    if policy in ("permitted", "conditional"):
        if detection.class_label == PPE_VIOLATION_CLASS and rule_set.ppe_requirement.vest_required:
            return ("ppe_violation", "ppe_vest_missing")
        # class_label == "Human": vest/coveralls PPE implied present.
        # shoes_required/helmet_required are unverifiable from this class
        # alone (Requirement 3.4) — no event fabricated for them.
        return None

    # Defensive fallback for an unrecognised policy value (should not
    # occur given rules.py's validation, but fail safe rather than crash).
    logger.warning(
        f"check_human(): unrecognised human_presence_policy "
        f"'{policy}' for location_type '{rule_set.location_type}'; "
        f"treating as prohibited (fail-safe)."
    )
    return ("zone_violation", "human_presence_prohibited")


# ============================================================================
# check_container() — Requirement 4.1, 4.2, 4.3, 4.4
# ============================================================================

# HazardEvent.hazard_type -> the container_checks_enabled/suppressed check
# name it corresponds to (see models.py's VALID_HAZARD_TYPES and rules.py's
# VALID_CONTAINER_CHECKS).
_CONTAINER_HAZARD_TYPE_TO_CHECK_NAME = {
    "container_misalignment": "misalignment",
    "container_door_open": "open_doors",
    "container_flipped": "flipped",
    "container_dangling": "dangling",
}


def check_container(
    detection: Detection,
    rule_set: LocationRuleSet,
    analyzer_result: Optional[HazardEvent],
) -> Optional[Tuple[str, str]]:
    """
    Given a HazardEvent already produced by the existing ContainerAnalyzer
    for this detection, decide whether to keep or drop it based on
    rule_set.container_checks_enabled / container_checks_suppressed.

    Does NOT re-implement any visual container analysis — this is purely
    a gate on already-computed results (Requirement 4.1, 4.2).

    Args:
        detection: The Detection the analyzer_result was derived from
            (accepted for interface symmetry with the other check_*
            functions and for future use, e.g. logging).
        rule_set: The resolved LocationRuleSet for the camera.
        analyzer_result: The HazardEvent produced by ContainerAnalyzer for
            this detection, or None if the analyzer produced no event.

    Returns:
        (hazard_type, matched_rule) tuple if the check is enabled and not
        suppressed for this location, otherwise None.
    """
    if analyzer_result is None:
        return None

    check_name = _CONTAINER_HAZARD_TYPE_TO_CHECK_NAME.get(analyzer_result.hazard_type)
    if check_name is None:
        # Not a container-check hazard type at all; nothing for this
        # function to gate.
        return None

    if check_name in rule_set.container_checks_suppressed:
        return None
    if check_name not in rule_set.container_checks_enabled:
        return None

    return (analyzer_result.hazard_type, f"container_{check_name}_check")


# ============================================================================
# check_vehicle() — Requirement 5.1, 5.2, 5.3, 5.5, 5.6
# ============================================================================


def check_vehicle(
    detection: Detection,
    rule_set: LocationRuleSet,
    unsafe_y_threshold: float,
) -> Optional[Tuple[str, str]]:
    """
    Check a vehicle-class detection for proximity hazard.

    unsafe_y_threshold is a TUNABLE PLACEHOLDER, not a validated safety
    distance — HSSE could not provide a specific quantifiable threshold
    ("pretty tight, not sure how to provide a meaningful data point"; see
    requirements.md Requirement 5.2). It is documented here as such, not
    presented as a validated value.

    Args:
        detection: A Detection with class_label in VEHICLE_CLASSES.
        rule_set: The resolved LocationRuleSet for the camera.
        unsafe_y_threshold: Normalized y-coordinate threshold; a
            detection whose bbox center is at or below this value (i.e.
            closer to the bottom/near edge of frame) is treated as
            unsafely close.

    Returns:
        ("vehicle_proximity", "vehicle_proximity_check") if
        rule_set.vehicle_checks_enabled and the detection's bbox center
        is at/beyond the threshold; None otherwise (including when
        vehicle_checks_enabled is False, regardless of position —
        Requirement 5.5).
    """
    if detection.class_label not in VEHICLE_CLASSES:
        return None

    if not rule_set.vehicle_checks_enabled:
        return None

    if detection.bbox.y_center >= unsafe_y_threshold:
        return ("vehicle_proximity", "vehicle_proximity_check")

    return None


# ============================================================================
# check_tel_spot() — Requirement 6.1, 6.2, 6.3, 6.4, 6.5
# ============================================================================


def check_tel_spot(
    detection: Detection,
    trucker_spot_polygon: Optional[List[Tuple[float, float]]],
) -> Optional[Tuple[str, str]]:
    """
    Check whether a human detection at a TEL/TELs camera falls outside
    the designated trucker spot polygon.

    Reuses ZoneMap's ray-casting point-in-polygon logic rather than
    reimplementing it (design.md).

    Args:
        detection: A human-class Detection at a TEL/TELs camera.
        trucker_spot_polygon: List of (x, y) normalized vertices defining
            the trucker spot, or None if no polygon is configured for
            this camera.

    Returns:
        ("tel_zone_violation", "tel_trucker_spot") if the detection's
        bbox center is outside the polygon, OR if trucker_spot_polygon is
        None (fail-safe, Requirement 6.4). Returns None if inside the
        polygon.
    """
    if trucker_spot_polygon is None:
        logger.warning(
            "check_tel_spot(): no trucker spot polygon configured; "
            "treating entire frame as outside the permitted zone (fail-safe)."
        )
        return ("tel_zone_violation", "tel_trucker_spot")

    x, y = detection.bbox.center
    if ZoneMap._point_in_polygon(x, y, trucker_spot_polygon):
        return None

    return ("tel_zone_violation", "tel_trucker_spot")


# ============================================================================
# check_tel_occupancy() — Requirement 6.6, 6.7
# ============================================================================


def check_tel_occupancy(
    detections_in_frame: List[Detection],
    rule_set: LocationRuleSet,
) -> Optional[Tuple[str, str]]:
    """
    Count confirmed human-class detections in one frame and check against
    the rule set's occupancy_limit.

    Always evaluates as if occupancy_maintenance_exception does NOT
    apply — there is no detection signal today for "maintenance activity
    is occurring" (requirements.md Requirement 6.7). This is a documented
    limitation, not an oversight; rule_set.notes carries the corresponding
    operator-facing documentation.

    Args:
        detections_in_frame: Detections already confirmed (confidence +
            temporal filtering applied by the caller) for a single frame.
        rule_set: The resolved LocationRuleSet for the camera.

    Returns:
        ("tel_occupancy_violation", "tel_occupancy_limit") if the number
        of human-class detections exceeds rule_set.occupancy_limit;
        None if under the limit or no limit is configured.
    """
    if rule_set.occupancy_limit is None:
        return None

    human_count = sum(
        1 for d in detections_in_frame if d.class_label in HUMAN_CLASSES
    )

    if human_count > rule_set.occupancy_limit:
        return ("tel_occupancy_violation", "tel_occupancy_limit")

    return None
