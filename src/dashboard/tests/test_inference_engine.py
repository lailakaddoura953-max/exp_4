"""
Unit tests for InferenceEngine (tasks 3.1 and 3.2).

Tests cover:
- Construction with valid config
- ValueError propagation from InferenceEngineConfig on bad config
- run() always returns a list
- run() returns [] on detector failure (exception swallowing)
- camera_id is attached to every HazardResult
- Results contain one entry per detection at or above threshold

These tests mock YOLODetector.detect() so they run without a GPU or
a real YOLO checkpoint on disk.  The engine itself is tested against the
contract described in requirements 1.1–1.6, 16.2, 16.5, 17.3.
"""

from __future__ import annotations

import importlib
import sys
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dashboard.models import InferenceEngineConfig, HazardResult
from hazard_detection.models import BBox, Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    checkpoint_path: str = "checkpoints/yolov12_best.pt",
    confidence_threshold: float = 0.5,
    device: str = "cpu",
) -> InferenceEngineConfig:
    return InferenceEngineConfig(
        checkpoint_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def _make_detection(
    confidence: float = 0.8,
    class_label: str = "Container - Misaligned",
) -> Detection:
    """Return a minimal Detection for use in mocked detector output."""
    return Detection(
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1),
        class_label=class_label,
        confidence=confidence,
    )


def _dummy_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a small black BGR image suitable for test input."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixture: build an InferenceEngine with a patched YOLODetector
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine_and_mock(tmp_path):
    """
    Yield (engine, mock_detector) where the engine's internal _detector
    has its detect() method replaced by a MagicMock.

    A dummy checkpoint file is created so YOLODetector's FileNotFoundError
    guard does not fire.
    """
    # Create a real (empty) checkpoint file so the path-existence check passes
    checkpoint = tmp_path / "yolov12_best.pt"
    checkpoint.write_bytes(b"")

    config = _make_config(checkpoint_path=str(checkpoint))

    # Patch YOLODetector so we never load a real model
    mock_detector = MagicMock()
    # Default: return one empty frame-detection list
    mock_detector.detect.return_value = [[]]
    mock_detector._device = "cpu"

    with patch("dashboard.inference_engine.YOLODetector", return_value=mock_detector):
        # Also patch ultralytics YOLO import inside yolo_detector if needed
        from dashboard.inference_engine import InferenceEngine
        engine = InferenceEngine(config)

    # Restore the mock onto the already-constructed engine
    engine._detector = mock_detector
    yield engine, mock_detector


# ---------------------------------------------------------------------------
# Task 3.1 — __init__ tests
# ---------------------------------------------------------------------------


class TestInferenceEngineInit:
    """Tests for InferenceEngine.__init__ (task 3.1)."""

    def test_valid_config_constructs_without_error(self, tmp_path):
        """Engine constructs successfully given a valid config."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")

        config = _make_config(checkpoint_path=str(checkpoint))
        mock_detector = MagicMock()
        mock_detector._device = "cpu"

        with patch("dashboard.inference_engine.YOLODetector", return_value=mock_detector):
            from dashboard.inference_engine import InferenceEngine
            engine = InferenceEngine(config)

        assert engine is not None

    def test_empty_checkpoint_path_raises_value_error(self):
        """Empty checkpoint_path raises ValueError at construction time."""
        with pytest.raises(ValueError, match="checkpoint_path"):
            _make_config(checkpoint_path="")

    def test_whitespace_checkpoint_path_raises_value_error(self):
        """Whitespace-only checkpoint_path raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_path"):
            _make_config(checkpoint_path="   ")

    def test_confidence_threshold_too_high_raises_value_error(self):
        """confidence_threshold > 1.0 raises ValueError at construction time."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            _make_config(confidence_threshold=1.1)

    def test_confidence_threshold_negative_raises_value_error(self):
        """confidence_threshold < 0.0 raises ValueError at construction time."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            _make_config(confidence_threshold=-0.01)

    def test_boundary_confidence_threshold_zero_is_valid(self, tmp_path):
        """confidence_threshold=0.0 is a valid boundary value."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        config = _make_config(checkpoint_path=str(checkpoint), confidence_threshold=0.0)
        mock_detector = MagicMock()
        mock_detector._device = "cpu"

        with patch("dashboard.inference_engine.YOLODetector", return_value=mock_detector):
            from dashboard.inference_engine import InferenceEngine
            engine = InferenceEngine(config)
        assert engine._config.confidence_threshold == 0.0

    def test_boundary_confidence_threshold_one_is_valid(self, tmp_path):
        """confidence_threshold=1.0 is a valid boundary value."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        config = _make_config(checkpoint_path=str(checkpoint), confidence_threshold=1.0)
        mock_detector = MagicMock()
        mock_detector._device = "cpu"

        with patch("dashboard.inference_engine.YOLODetector", return_value=mock_detector):
            from dashboard.inference_engine import InferenceEngine
            engine = InferenceEngine(config)
        assert engine._config.confidence_threshold == 1.0

    def test_config_stored_on_engine(self, tmp_path):
        """The config object passed to __init__ is accessible as _config."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        config = _make_config(checkpoint_path=str(checkpoint), confidence_threshold=0.7)
        mock_detector = MagicMock()
        mock_detector._device = "cpu"

        with patch("dashboard.inference_engine.YOLODetector", return_value=mock_detector):
            from dashboard.inference_engine import InferenceEngine
            engine = InferenceEngine(config)

        assert engine._config is config


# ---------------------------------------------------------------------------
# Task 3.2 — run() tests
# ---------------------------------------------------------------------------


class TestInferenceEngineRun:
    """Tests for InferenceEngine.run() (task 3.2)."""

    def test_run_returns_list(self, engine_and_mock):
        """run() always returns a list — even when the detector returns nothing."""
        engine, mock_detector = engine_and_mock
        mock_detector.detect.return_value = [[]]

        result = engine.run(_dummy_image(), "cam_01")

        assert isinstance(result, list)

    def test_run_returns_empty_list_when_no_detections(self, engine_and_mock):
        """No detections → empty list (Requirement 1.2)."""
        engine, mock_detector = engine_and_mock
        mock_detector.detect.return_value = [[]]

        result = engine.run(_dummy_image(), "cam_01")

        assert result == []

    def test_run_returns_empty_list_on_detector_exception(self, engine_and_mock):
        """Detector exception → [] returned, not raised (Requirement 1.4)."""
        engine, mock_detector = engine_and_mock
        mock_detector.detect.side_effect = RuntimeError("GPU OOM")

        result = engine.run(_dummy_image(), "cam_01")

        assert result == []

    def test_run_returns_empty_list_on_arbitrary_exception(self, engine_and_mock):
        """Any exception type is caught and [] is returned."""
        engine, mock_detector = engine_and_mock
        mock_detector.detect.side_effect = ValueError("unexpected")

        result = engine.run(_dummy_image(), "cam_01")

        assert result == []

    def test_camera_id_attached_to_all_results(self, engine_and_mock):
        """Every HazardResult carries the supplied camera_id (Requirement 1.3)."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.9, class_label="Container - Misaligned")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_test_42")

        assert len(results) > 0
        for r in results:
            assert r.camera_id == "cam_test_42"

    def test_detections_below_threshold_are_filtered(self, engine_and_mock):
        """Detections below confidence_threshold produce no results (Req 1.1)."""
        engine, mock_detector = engine_and_mock
        # engine threshold is 0.5; provide detections at 0.4 (below) and 0.8 (above)
        det_below = _make_detection(confidence=0.4, class_label="Container - Misaligned")
        det_above = _make_detection(confidence=0.8, class_label="Container - Misaligned")
        mock_detector.detect.return_value = [[det_below, det_above]]

        results = engine.run(_dummy_image(), "cam_01")

        # Only the det_above should produce a result
        assert len(results) == 1
        assert results[0].confidence <= 1.0

    def test_one_result_per_filtered_detection(self, engine_and_mock):
        """One HazardResult per detection at or above threshold (Req 16.5)."""
        engine, mock_detector = engine_and_mock
        dets = [
            _make_detection(confidence=0.6, class_label="Container - Misaligned"),
            _make_detection(confidence=0.7, class_label="Container - Water Drop"),
            _make_detection(confidence=0.9, class_label="Human - No Safety Clothes"),
        ]
        mock_detector.detect.return_value = [dets]

        results = engine.run(_dummy_image(), "cam_01")

        assert len(results) == 3

    def test_result_confidence_clamped_to_valid_range(self, engine_and_mock):
        """All result confidences are in [0.0, 1.0] (Requirement 16.2)."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.85, class_label="Human - No Safety Clothes")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_01")

        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_hazard_results_have_correct_types(self, engine_and_mock):
        """Each element of the return value is a HazardResult instance."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.9, class_label="Container - Misaligned")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_01")

        assert len(results) == 1
        assert isinstance(results[0], HazardResult)

    def test_misaligned_container_flagged_as_hazard(self, engine_and_mock):
        """Container - Misaligned above threshold → is_hazard=True (Req 2.1)."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.9, class_label="Container - Misaligned")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_01")

        assert results[0].is_hazard is True
        assert results[0].hazard_reason == "misaligned_container"

    def test_ppe_violation_always_flagged(self, engine_and_mock):
        """Human - No Safety Clothes → ppe_violation (Req 5.1)."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.75, class_label="Human - No Safety Clothes")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_ppe")

        assert results[0].is_hazard is True
        assert results[0].hazard_reason == "ppe_violation"
        assert results[0].camera_id == "cam_ppe"

    def test_non_hazard_class_is_not_hazard(self, engine_and_mock):
        """Container - Stacked (non-hazard class) → is_hazard=False (Req 8.1)."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.8, class_label="Container - Stacked")
        mock_detector.detect.return_value = [[det]]

        results = engine.run(_dummy_image(), "cam_01")

        assert results[0].is_hazard is False
        assert results[0].hazard_reason == ""

    def test_run_with_different_camera_ids(self, engine_and_mock):
        """camera_id is propagated per call, not cached between calls."""
        engine, mock_detector = engine_and_mock
        det = _make_detection(confidence=0.9, class_label="Container - Misaligned")
        mock_detector.detect.return_value = [[det]]

        results_a = engine.run(_dummy_image(), "cam_A")
        results_b = engine.run(_dummy_image(), "cam_B")

        assert results_a[0].camera_id == "cam_A"
        assert results_b[0].camera_id == "cam_B"
