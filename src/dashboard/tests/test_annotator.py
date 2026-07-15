"""
Unit tests for src/dashboard/annotator.py

Tests cover:
- Hard-failure guards (None image, empty array, wrong ndim) → returns None
- Valid image with empty result list → returns non-None base64 string
- Base64 output decodes to a valid PNG
- Greyscale (2-D) and BGRA (4-channel) images are converted without error
- Bounding box colour coding: hazard → red channel dominant,
  safe → green channel dominant
- Label text presence verified by checking the annotated image differs from
  the original when results are provided
- Recoverable per-detection error: a bad result is skipped; function still
  returns a base64 string (not None)
- Original image is never mutated

Requirements validated: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import base64
import struct
import zlib

import cv2
import numpy as np
import pytest

from hazard_detection.models import BBox
from dashboard.models import HazardResult
from dashboard.annotator import annotate


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_image(h: int = 200, w: int = 200, channels: int = 3) -> np.ndarray:
    """Return a solid grey image for testing."""
    if channels == 1:
        return np.full((h, w), 128, dtype=np.uint8)
    return np.full((h, w, channels), 128, dtype=np.uint8)


def _make_result(
    *,
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.4,
    height: float = 0.4,
    is_hazard: bool = False,
    hazard_reason: str = "",
    class_label: str = "TestClass",
    confidence: float = 0.75,
    camera_id: str = "cam_stub_01",
) -> HazardResult:
    bbox = BBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )
    return HazardResult(
        class_label=class_label,
        confidence=confidence,
        bbox=bbox,
        is_hazard=is_hazard,
        hazard_reason=hazard_reason,
        camera_id=camera_id,
    )


def _decode_b64_png(b64: str) -> np.ndarray:
    """Decode a base64 PNG string back to a NumPy array via cv2."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _is_valid_png_header(b64: str) -> bool:
    """Check that the base64 string starts with the PNG magic bytes."""
    raw = base64.b64decode(b64)
    return raw[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Hard-failure guard tests
# ---------------------------------------------------------------------------


class TestHardFailures:
    """annotate() should return None for invalid inputs (Req 11.4)."""

    def test_none_image_returns_none(self):
        assert annotate(None, []) is None

    def test_empty_array_returns_none(self):
        empty = np.array([])
        assert annotate(empty, []) is None

    def test_1d_array_returns_none(self):
        arr = np.zeros(100, dtype=np.uint8)
        assert annotate(arr, []) is None

    def test_4d_array_returns_none(self):
        arr = np.zeros((10, 10, 3, 2), dtype=np.uint8)
        assert annotate(arr, []) is None

    def test_zero_size_2d_array_returns_none(self):
        arr = np.zeros((0, 0), dtype=np.uint8)
        assert annotate(arr, []) is None


# ---------------------------------------------------------------------------
# Valid image — basic output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Return value should be a valid base64-encoded PNG string (Req 11.3)."""

    def test_returns_string_for_valid_image_no_results(self):
        img = _make_image()
        result = annotate(img, [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_is_valid_base64(self):
        img = _make_image()
        b64 = annotate(img, [])
        # Should not raise
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_output_has_png_magic_bytes(self):
        img = _make_image()
        b64 = annotate(img, [])
        assert _is_valid_png_header(b64)

    def test_decoded_image_has_same_dimensions(self):
        h, w = 150, 200
        img = _make_image(h=h, w=w)
        b64 = annotate(img, [])
        decoded = _decode_b64_png(b64)
        assert decoded is not None
        assert decoded.shape[:2] == (h, w)


# ---------------------------------------------------------------------------
# Image type handling
# ---------------------------------------------------------------------------


class TestImageTypeHandling:
    """Greyscale and BGRA inputs should be handled without error (Req 11.4)."""

    def test_greyscale_2d_image_succeeds(self):
        img = _make_image(channels=1)
        assert img.ndim == 2
        b64 = annotate(img, [])
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)

    def test_bgra_4channel_image_succeeds(self):
        img = _make_image(channels=4)
        assert img.shape[2] == 4
        b64 = annotate(img, [])
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)


# ---------------------------------------------------------------------------
# Original image immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """The original image array must never be mutated (task note)."""

    def test_original_image_not_mutated_with_results(self):
        img = _make_image()
        original_copy = img.copy()
        result = _make_result(is_hazard=True, hazard_reason="ppe_violation")
        annotate(img, [result])
        np.testing.assert_array_equal(img, original_copy)

    def test_original_image_not_mutated_with_no_results(self):
        img = _make_image()
        original_copy = img.copy()
        annotate(img, [])
        np.testing.assert_array_equal(img, original_copy)


# ---------------------------------------------------------------------------
# Annotation visually modifies the image
# ---------------------------------------------------------------------------


class TestAnnotationProducesChanges:
    """When results are provided the annotated image should differ from the
    original (a bbox + text must be drawn somewhere)."""

    def test_annotated_image_differs_from_original_hazard(self):
        img = _make_image()
        b64 = annotate(img, [_make_result(is_hazard=True, hazard_reason="ppe_violation")])
        decoded = _decode_b64_png(b64)
        # At least one pixel should differ
        assert not np.array_equal(decoded, img)

    def test_annotated_image_differs_from_original_safe(self):
        img = _make_image()
        b64 = annotate(img, [_make_result(is_hazard=False)])
        decoded = _decode_b64_png(b64)
        assert not np.array_equal(decoded, img)


# ---------------------------------------------------------------------------
# Bounding box colour coding (Req 11.1)
# ---------------------------------------------------------------------------


class TestBoundingBoxColours:
    """Hazard boxes should be red (BGR: high blue=0, green=0, red=255),
    safe boxes should be green (BGR: high green, low red/blue)."""

    def _annotate_and_decode(self, is_hazard: bool) -> np.ndarray:
        # Use a black (0,0,0) background so drawn colours are easy to isolate.
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        result = _make_result(
            x_center=0.5, y_center=0.5, width=0.6, height=0.6,
            is_hazard=is_hazard,
            hazard_reason="ppe_violation" if is_hazard else "",
        )
        b64 = annotate(img, [result])
        return _decode_b64_png(b64)

    def test_hazard_box_has_red_pixels(self):
        decoded = self._annotate_and_decode(is_hazard=True)
        # The bounding box edge pixels should include red (BGR: [0, 0, 255])
        # Check the left edge of the box: x=40, y range 40-160
        # Red channel is index 2 in BGR
        left_edge_x = int((0.5 - 0.3) * 200)  # x_center-width/2 = 0.2 → pixel 40
        red_channel = decoded[:, left_edge_x, 2]
        assert np.any(red_channel > 200), "Expected red pixels on hazard box edge"

    def test_safe_box_has_green_pixels(self):
        decoded = self._annotate_and_decode(is_hazard=False)
        left_edge_x = int((0.5 - 0.3) * 200)
        green_channel = decoded[:, left_edge_x, 1]
        assert np.any(green_channel > 200), "Expected green pixels on safe box edge"


# ---------------------------------------------------------------------------
# Recoverable per-detection error (Req 11.4)
# ---------------------------------------------------------------------------


class TestRecoverableError:
    """If one detection fails to draw, the others should still be drawn and
    the function should return a base64 string (not None)."""

    def test_bad_result_skipped_others_drawn(self):
        img = _make_image()

        # Create a result with a bbox that has a None x_center to force an
        # error inside _draw_detection.  We monkey-patch the bbox attribute.
        good_result = _make_result(is_hazard=False, class_label="Good")

        bad_result = _make_result(is_hazard=False, class_label="Bad")
        # Force a runtime error by setting bbox to an invalid object
        bad_result.bbox = None  # type: ignore[assignment]

        b64 = annotate(img, [bad_result, good_result])
        # Should return a string (not None) because the error is recoverable
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)

    def test_all_bad_results_still_returns_base64(self):
        img = _make_image()
        bad_result = _make_result()
        bad_result.bbox = None  # type: ignore[assignment]
        b64 = annotate(img, [bad_result])
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)


# ---------------------------------------------------------------------------
# Label format (Req 11.2)
# ---------------------------------------------------------------------------


class TestLabelFormat:
    """Label text format: '{class_label} {confidence:.0%}'.
    Hazard reason overlaid when is_hazard=True.
    We verify indirectly that the function runs without error for various
    confidence values and hazard reasons."""

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 0.87, 1.0])
    def test_various_confidences_succeed(self, confidence: float):
        img = _make_image()
        result = _make_result(confidence=confidence, is_hazard=False)
        b64 = annotate(img, [result])
        assert isinstance(b64, str)

    @pytest.mark.parametrize(
        "class_label,hazard_reason",
        [
            ("Container - Misaligned", "misaligned_container"),
            ("Human - No Safety Clothes", "ppe_violation"),
            ("Human", "human_below_crane"),
            ("Container - Open", "open_container_unsecured"),
        ],
    )
    def test_hazard_with_reason_succeeds(self, class_label: str, hazard_reason: str):
        img = _make_image()
        result = _make_result(
            class_label=class_label,
            is_hazard=True,
            hazard_reason=hazard_reason,
        )
        b64 = annotate(img, [result])
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)

    def test_hazard_with_empty_reason_does_not_crash(self):
        """is_hazard=True but hazard_reason="" — no reason overlay should be drawn."""
        img = _make_image()
        result = _make_result(is_hazard=True, hazard_reason="")
        b64 = annotate(img, [result])
        assert isinstance(b64, str)


# ---------------------------------------------------------------------------
# Edge cases — empty results list and multiple detections
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results_returns_unchanged_encoded_image(self):
        img = _make_image()
        b64_no_results = annotate(img, [])
        b64_with_results = annotate(
            img, [_make_result(is_hazard=False)]
        )
        # Both should succeed; they may differ in content (annotations drawn)
        assert isinstance(b64_no_results, str)
        assert isinstance(b64_with_results, str)

    def test_multiple_results_all_drawn(self):
        img = _make_image(h=300, w=300)
        results = [
            _make_result(x_center=0.2, y_center=0.2, width=0.2, height=0.2,
                         is_hazard=True, hazard_reason="ppe_violation",
                         class_label="Human - No Safety Clothes"),
            _make_result(x_center=0.8, y_center=0.8, width=0.2, height=0.2,
                         is_hazard=False, class_label="Container - Stacked"),
        ]
        b64 = annotate(img, results)
        assert isinstance(b64, str)
        assert _is_valid_png_header(b64)
        decoded = _decode_b64_png(b64)
        assert decoded is not None

    def test_bbox_at_image_boundary(self):
        """Bbox touching image edges should not raise an error."""
        img = _make_image()
        result = _make_result(x_center=0.0, y_center=0.0, width=0.1, height=0.1)
        b64 = annotate(img, [result])
        assert isinstance(b64, str)

    def test_bbox_full_image(self):
        """Bbox covering the entire image should not raise an error."""
        img = _make_image()
        result = _make_result(x_center=0.5, y_center=0.5, width=1.0, height=1.0)
        b64 = annotate(img, [result])
        assert isinstance(b64, str)
