"""
Unit tests for classify_all (task 2.3).

Covers one representative example per rule branch to verify the priority-
ordered rule chain is wired correctly.
"""

import pytest

from hazard_detection.models import BBox, Detection
from dashboard.models import HazardResult, InferenceEngineConfig
from dashboard.rules import classify_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> InferenceEngineConfig:
    return InferenceEngineConfig(
        checkpoint_path="checkpoints/yolov12_best.pt",
        confidence_threshold=0.5,
        flipped_aspect_ratio_threshold=1.5,
        iou_threshold=0.5,
    )


def make_det(label: str, x=0.5, y=0.5, w=0.4, h=0.2, conf=0.9) -> Detection:
    """Helper: build a Detection with sensible defaults."""
    return Detection(bbox=BBox(x, y, w, h), class_label=label, confidence=conf)


def make_flipped_det(label: str, conf=0.9) -> Detection:
    """A container bbox whose height/width > 1.5 (flipped)."""
    # width=0.2, height=0.4 → ratio = 2.0 > 1.5
    return Detection(bbox=BBox(0.5, 0.5, 0.2, 0.4), class_label=label, confidence=conf)


# ---------------------------------------------------------------------------
# Pass 1 — context lookups (smoke tests: empty frame)
# ---------------------------------------------------------------------------

def test_empty_detections_returns_empty_list(cfg):
    results = classify_all([], cfg, "cam_stub_01")
    assert results == []


def test_returns_one_result_per_detection(cfg):
    dets = [
        make_det("Container - Stacked"),
        make_det("Human"),
        make_det("Crane"),
    ]
    results = classify_all(dets, cfg, "cam")
    assert len(results) == len(dets)


def test_camera_id_propagated(cfg):
    dets = [make_det("Crane")]
    results = classify_all(dets, cfg, "cam_test_42")
    assert all(r.camera_id == "cam_test_42" for r in results)


# ---------------------------------------------------------------------------
# Rule 0 — Flipped container
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "Container - Stacked",
    "Container - Reefer",
    "Container - Separate",
    "Container - Open",
    "Container - Picked",
    "Container - Misaligned",
    "Container - Water Drop",
])
def test_rule0_flipped_fires_for_all_container_classes(cfg, label):
    det = make_flipped_det(label)
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "flipped_container"


def test_rule0_flipped_overrides_misaligned(cfg):
    """Flipped container-misaligned reports flipped_container, not misaligned_container."""
    det = make_flipped_det("Container - Misaligned")
    results = classify_all([det], cfg, "cam")
    assert results[0].hazard_reason == "flipped_container"


def test_rule0_does_not_fire_when_width_zero(cfg):
    """Width=0 bbox must not trigger flipped rule (Req 7.4)."""
    # BBox validates width ∈ [0,1] and 0 is valid
    det = Detection(bbox=BBox(0.5, 0.5, 0.0, 0.4), class_label="Container - Stacked", confidence=0.9)
    results = classify_all([det], cfg, "cam")
    # Should fall through to Rule 7 (non-hazard)
    assert results[0].is_hazard is False


def test_rule0_does_not_fire_when_ratio_below_threshold(cfg):
    """A wide container (ratio < 1.5) must not be flagged as flipped."""
    # width=0.4, height=0.2 → ratio=0.5
    det = Detection(bbox=BBox(0.5, 0.5, 0.4, 0.2), class_label="Container - Stacked", confidence=0.9)
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is False


# ---------------------------------------------------------------------------
# Rule 1 — Unconditional hazards
# ---------------------------------------------------------------------------

def test_rule1_misaligned_container(cfg):
    det = make_det("Container - Misaligned")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "misaligned_container"


def test_rule1_water_drop_container(cfg):
    det = make_det("Container - Water Drop")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "water_drop_container"


# ---------------------------------------------------------------------------
# Rule 2 — Container - Open loading suppression
# ---------------------------------------------------------------------------

def test_rule2_open_container_no_overlap_is_hazard(cfg):
    """No crane/picked nearby → hazard."""
    open_det = make_det("Container - Open", x=0.1, y=0.1, w=0.1, h=0.1)
    crane_det = make_det("Crane", x=0.9, y=0.9, w=0.1, h=0.1)
    results = classify_all([open_det, crane_det], cfg, "cam")
    open_result = next(r for r in results if r.class_label == "Container - Open")
    assert open_result.is_hazard is True
    assert open_result.hazard_reason == "open_container_unsecured"


def test_rule2_open_container_overlapping_crane_suppressed(cfg):
    """Crane overlapping same bbox → suppressed (is_hazard=False)."""
    bbox = BBox(0.5, 0.5, 0.3, 0.3)
    open_det = Detection(bbox=bbox, class_label="Container - Open", confidence=0.9)
    crane_det = Detection(bbox=bbox, class_label="Crane", confidence=0.9)
    results = classify_all([open_det, crane_det], cfg, "cam")
    open_result = next(r for r in results if r.class_label == "Container - Open")
    assert open_result.is_hazard is False
    assert open_result.hazard_reason == ""


def test_rule2_open_container_overlapping_picked_suppressed(cfg):
    """Container-Picked overlapping open container → suppressed."""
    bbox = BBox(0.5, 0.5, 0.3, 0.3)
    open_det = Detection(bbox=bbox, class_label="Container - Open", confidence=0.9)
    picked_det = Detection(bbox=bbox, class_label="Container - Picked", confidence=0.9)
    results = classify_all([open_det, picked_det], cfg, "cam")
    open_result = next(r for r in results if r.class_label == "Container - Open")
    assert open_result.is_hazard is False


# ---------------------------------------------------------------------------
# Rule 3 — Container - Picked
# ---------------------------------------------------------------------------

def test_rule3_picked_no_crane(cfg):
    det = make_det("Container - Picked")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "picked_no_crane"


def test_rule3_picked_with_crane_no_person_is_safe(cfg):
    picked = make_det("Container - Picked", x=0.5, y=0.5, w=0.2, h=0.2)
    crane = make_det("Crane", x=0.5, y=0.5, w=0.3, h=0.3)
    results = classify_all([picked, crane], cfg, "cam")
    picked_result = next(r for r in results if r.class_label == "Container - Picked")
    assert picked_result.is_hazard is False
    assert picked_result.hazard_reason == ""


def test_rule3_picked_with_crane_and_person_below_hazard(cfg):
    """Person whose y_center >= crane y_center triggers picked_person_below_crane."""
    # crane y_center=0.3; person y_center=0.8 → below crane
    crane = Detection(bbox=BBox(0.5, 0.3, 0.3, 0.3), class_label="Crane", confidence=0.9)
    picked = make_det("Container - Picked")
    human = Detection(bbox=BBox(0.5, 0.8, 0.1, 0.1), class_label="Human", confidence=0.9)
    results = classify_all([picked, crane, human], cfg, "cam")
    picked_result = next(r for r in results if r.class_label == "Container - Picked")
    assert picked_result.is_hazard is True
    assert picked_result.hazard_reason == "picked_person_below_crane"


# ---------------------------------------------------------------------------
# Rule 4 — PPE violation
# ---------------------------------------------------------------------------

def test_rule4_ppe_violation(cfg):
    det = make_det("Human - No Safety Clothes")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "ppe_violation"


# ---------------------------------------------------------------------------
# Rule 5 — Human stub and crane override
# ---------------------------------------------------------------------------

def test_rule5_human_stub_no_crane(cfg):
    det = make_det("Human")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is True
    assert results[0].hazard_reason == "human_detected_stub"


def test_rule5_human_below_crane_override(cfg):
    """Human y_center >= crane y_center → human_below_crane."""
    crane = Detection(bbox=BBox(0.5, 0.3, 0.3, 0.3), class_label="Crane", confidence=0.9)
    human = Detection(bbox=BBox(0.5, 0.8, 0.1, 0.1), class_label="Human", confidence=0.9)
    results = classify_all([crane, human], cfg, "cam")
    human_result = next(r for r in results if r.class_label == "Human")
    assert human_result.is_hazard is True
    assert human_result.hazard_reason == "human_below_crane"


def test_rule5_human_above_crane_is_stub(cfg):
    """Human y_center < crane y_center → no crane override, falls to stub."""
    crane = Detection(bbox=BBox(0.5, 0.8, 0.3, 0.3), class_label="Crane", confidence=0.9)
    human = Detection(bbox=BBox(0.5, 0.2, 0.1, 0.1), class_label="Human", confidence=0.9)
    results = classify_all([crane, human], cfg, "cam")
    human_result = next(r for r in results if r.class_label == "Human")
    assert human_result.hazard_reason == "human_detected_stub"


# ---------------------------------------------------------------------------
# Rule 6 — Zone classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "Yard - No People",
    "Yard - Operation Zone",
    "Yard - Dropoff zone",
])
def test_rule6_zone_classes_never_hazard(cfg, label):
    det = make_det(label)
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is False
    assert results[0].hazard_reason == ""


# ---------------------------------------------------------------------------
# Rule 7 — Non-hazard classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [
    "Container - Separate",
    "Container - Stacked",
    "Container - Reefer",
    "Vehicle",
    "Truck - No Container",
    "Truck - With Container",
    "Boat - With Cargo",
    "Crane",
])
def test_rule7_non_hazard_classes(cfg, label):
    # Normal (non-flipped) bbox: width=0.4, height=0.2 → ratio=0.5
    det = Detection(bbox=BBox(0.5, 0.5, 0.4, 0.2), class_label=label, confidence=0.9)
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is False
    assert results[0].hazard_reason == ""


# ---------------------------------------------------------------------------
# Rule 8 — Unknown class defensive fallback
# ---------------------------------------------------------------------------

def test_rule8_unknown_class_fallback(cfg):
    det = make_det("SomeUnknownClass_XYZ")
    results = classify_all([det], cfg, "cam")
    assert results[0].is_hazard is False
    assert results[0].hazard_reason == ""


# ---------------------------------------------------------------------------
# Requirement 16.3 — is_hazard=False always means hazard_reason=""
# ---------------------------------------------------------------------------

def test_non_hazard_results_have_empty_reason(cfg):
    dets = [
        make_det("Container - Separate"),
        make_det("Vehicle"),
        make_det("Yard - No People"),
    ]
    results = classify_all(dets, cfg, "cam")
    for r in results:
        if not r.is_hazard:
            assert r.hazard_reason == "", f"Expected '' for {r.class_label}, got {r.hazard_reason!r}"
