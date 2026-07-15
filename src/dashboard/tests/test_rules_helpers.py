"""
Unit tests for the pure helper functions in dashboard.rules:
  - compute_iou
  - is_flipped
  - person_below_crane

These tests verify the three functions introduced in task 2.1.
"""

import pytest
from hazard_detection.models import BBox
from dashboard.rules import compute_iou, is_flipped, person_below_crane


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

class TestComputeIou:
    """Tests for compute_iou(box_a, box_b)."""

    def test_identical_boxes_returns_near_one(self):
        """Identical boxes should produce an IoU at or very close to 1.0."""
        b = BBox(0.5, 0.5, 0.4, 0.4)
        result = compute_iou(b, b)
        assert abs(result - 1.0) < 1e-9

    def test_non_overlapping_boxes_returns_zero(self):
        """Boxes far apart should produce IoU = 0.0."""
        b1 = BBox(0.1, 0.1, 0.1, 0.1)
        b2 = BBox(0.9, 0.9, 0.1, 0.1)
        assert compute_iou(b1, b2) == 0.0

    def test_zero_area_box_returns_zero(self):
        """Zero-area bbox: union is zero → return 0.0, not a div-by-zero error."""
        bz = BBox(0.5, 0.5, 0.0, 0.0)
        assert compute_iou(bz, bz) == 0.0

    def test_result_in_unit_interval(self):
        """Partially overlapping boxes must produce IoU in [0.0, 1.0]."""
        val = compute_iou(BBox(0.5, 0.5, 0.3, 0.3), BBox(0.55, 0.55, 0.3, 0.3))
        assert 0.0 <= val <= 1.0

    def test_partial_overlap_value(self):
        """Verify a known partial-overlap result against manual calculation."""
        # Two 0.2×0.2 boxes whose centres are 0.1 apart — manual IoU ≈ 1/7
        a = BBox(0.4, 0.5, 0.2, 0.2)
        b = BBox(0.5, 0.5, 0.2, 0.2)
        # Intersection: x-overlap = 0.1, y-overlap = 0.2 → area = 0.02
        # Union: 0.04 + 0.04 - 0.02 = 0.06 → IoU = 0.02/0.06 ≈ 0.3333
        result = compute_iou(a, b)
        assert abs(result - (1.0 / 3.0)) < 1e-9

    def test_symmetry(self):
        """IoU is symmetric: compute_iou(a, b) == compute_iou(b, a)."""
        a = BBox(0.3, 0.3, 0.2, 0.3)
        b = BBox(0.4, 0.4, 0.3, 0.2)
        assert compute_iou(a, b) == compute_iou(b, a)

    def test_result_never_exceeds_one(self):
        """Due to clamping, result must never exceed 1.0."""
        b = BBox(0.5, 0.5, 0.5, 0.5)
        assert compute_iou(b, b) <= 1.0

    def test_result_never_below_zero(self):
        """Result must never be negative."""
        b1 = BBox(0.2, 0.2, 0.1, 0.1)
        b2 = BBox(0.8, 0.8, 0.1, 0.1)
        assert compute_iou(b1, b2) >= 0.0


# ---------------------------------------------------------------------------
# is_flipped
# ---------------------------------------------------------------------------

class TestIsFlipped:
    """Tests for is_flipped(bbox, threshold)."""

    def test_width_zero_returns_false(self):
        """Degenerate bbox with width == 0 must return False (Req 7.4)."""
        b = BBox(0.5, 0.5, 0.0, 0.0)
        assert is_flipped(b, 1.5) is False

    def test_below_threshold_returns_false(self):
        """h/w = 0.5, threshold = 1.5 → not flipped."""
        b = BBox(0.5, 0.5, 0.4, 0.2)   # h/w = 0.5
        assert is_flipped(b, 1.5) is False

    def test_above_threshold_returns_true(self):
        """h/w = 2.5, threshold = 1.5 → flipped."""
        b = BBox(0.5, 0.5, 0.2, 0.5)   # h/w = 2.5
        assert is_flipped(b, 1.5) is True

    def test_exactly_at_threshold_returns_false(self):
        """h/w == threshold: the check is strict (>) so this is False."""
        # 0.3 / 0.2 == 1.5 exactly
        b = BBox(0.5, 0.5, 0.2, 0.3)
        assert is_flipped(b, 1.5) is False

    def test_custom_threshold(self):
        """is_flipped respects an arbitrary threshold value."""
        b = BBox(0.5, 0.5, 0.2, 0.5)   # h/w = 2.5
        assert is_flipped(b, 2.0) is True
        assert is_flipped(b, 3.0) is False

    def test_landscape_container_is_not_flipped(self):
        """Wide (landscape) container should never trigger the flipped flag."""
        b = BBox(0.5, 0.5, 0.5, 0.2)   # h/w = 0.4
        assert is_flipped(b, 1.5) is False

    def test_square_bbox_with_high_threshold(self):
        """h/w = 1.0, threshold = 1.5 → not flipped."""
        b = BBox(0.5, 0.5, 0.3, 0.3)
        assert is_flipped(b, 1.5) is False


# ---------------------------------------------------------------------------
# person_below_crane
# ---------------------------------------------------------------------------

class TestPersonBelowCrane:
    """Tests for person_below_crane(person_bbox, crane_bbox)."""

    def test_person_at_same_y_center_is_below(self):
        """Equal y_center → True (person is at the crane midpoint, i.e. in danger zone)."""
        person = BBox(0.5, 0.6, 0.1, 0.1)
        crane  = BBox(0.5, 0.6, 0.3, 0.5)
        assert person_below_crane(person, crane) is True

    def test_person_below_crane_in_image(self):
        """Larger y_center means lower in the image → True."""
        person = BBox(0.5, 0.8, 0.1, 0.1)
        crane  = BBox(0.5, 0.6, 0.3, 0.5)
        assert person_below_crane(person, crane) is True

    def test_person_above_crane_in_image(self):
        """Smaller y_center means higher in the image → False."""
        person = BBox(0.5, 0.3, 0.1, 0.1)
        crane  = BBox(0.5, 0.6, 0.3, 0.5)
        assert person_below_crane(person, crane) is False

    def test_person_just_below_by_tiny_margin(self):
        """Very small margin above the crane midpoint → True."""
        crane  = BBox(0.5, 0.5, 0.3, 0.6)
        person = BBox(0.5, 0.5 + 1e-10, 0.1, 0.1)
        assert person_below_crane(person, crane) is True

    def test_person_just_above_by_tiny_margin(self):
        """Very small margin below the crane midpoint → False."""
        crane  = BBox(0.5, 0.5, 0.3, 0.6)
        person = BBox(0.5, 0.5 - 1e-10, 0.1, 0.1)
        assert person_below_crane(person, crane) is False

    def test_boundary_condition_crane_at_top(self):
        """Crane at y_center = 0.0: any person (y >= 0) is always at or below."""
        crane  = BBox(0.5, 0.0, 0.3, 0.1)
        person = BBox(0.5, 0.0, 0.1, 0.1)
        assert person_below_crane(person, crane) is True
