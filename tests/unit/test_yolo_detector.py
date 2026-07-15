"""
Unit tests for the YOLODetector wrapper.

Tests cover:
- Property 23: YOLO detection output structure (valid bbox, class_label, confidence)
- Property 24: YOLO preprocessing consistency (resolution, normalization)
- Confidence threshold filtering
- CPU fallback when CUDA unavailable
- FileNotFoundError for invalid checkpoint path
- Visual output generation (class distribution, confidence histogram, feature distributions)

Validates: Requirements 13.2, 13.3, 13.6, 13.7
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.models import (
    BBox,
    Detection,
    FrameSequence,
    YOLOConfig,
)
from hazard_detection.yolo_detector import YOLODetector, ROBOFLOW_CLASSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_config(tmp_path):
    """YOLOConfig with a valid (existing) checkpoint path."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("fake model weights")
    return YOLOConfig(
        checkpoint_path=str(checkpoint),
        device="cpu",
        input_resolution=640,
        confidence_threshold=0.5,
    )


@pytest.fixture
def mock_frame_sequence():
    """6-frame sequence with random data."""
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(6)]
    timestamps = [1700000000.0 + i * 0.5 for i in range(6)]
    return FrameSequence(frames=frames, camera_id="cam_test", timestamps=timestamps)


def _make_mock_result(detections_data):
    """
    Create a mock Ultralytics result object.

    detections_data: list of (class_idx, confidence, x_center, y_center, width, height)
    """
    result = MagicMock()
    if not detections_data:
        result.boxes = None
        return result

    n = len(detections_data)
    boxes = MagicMock()
    boxes.__len__ = lambda self: n

    confs = [d[1] for d in detections_data]
    cls_ids = [d[0] for d in detections_data]
    xywhn_data = [[d[2], d[3], d[4], d[5]] for d in detections_data]

    boxes.conf = confs
    boxes.cls = cls_ids
    boxes.xywhn = [np.array(x) for x in xywhn_data]

    result.boxes = boxes
    result.names = {i: name for i, name in enumerate(ROBOFLOW_CLASSES)}
    return result


# ---------------------------------------------------------------------------
# Tests: Checkpoint Validation (Requirement 13.4)
# ---------------------------------------------------------------------------


class TestCheckpointValidation:
    """Test checkpoint path validation."""

    def test_raises_file_not_found_for_invalid_path(self):
        """FileNotFoundError raised when checkpoint path doesn't exist."""
        config = YOLOConfig(
            checkpoint_path="/nonexistent/path/model.pt",
            device="cpu",
            input_resolution=640,
            confidence_threshold=0.5,
        )
        with pytest.raises(FileNotFoundError, match="YOLO checkpoint not found"):
            YOLODetector(config)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_loads_model_with_valid_path(self, mock_yolo_class, valid_config):
        """Model loads successfully when checkpoint exists."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)
        mock_yolo_class.assert_called_once_with(valid_config.checkpoint_path)


# ---------------------------------------------------------------------------
# Tests: CUDA Fallback (Requirement 13.5)
# ---------------------------------------------------------------------------


class TestCUDAFallback:
    """Test CUDA fallback behavior."""

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_falls_back_to_cpu_when_cuda_unavailable(self, mock_yolo_class, tmp_path):
        """Falls back to CPU if CUDA requested but unavailable."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake")
        config = YOLOConfig(
            checkpoint_path=str(checkpoint),
            device="cuda",
            input_resolution=640,
            confidence_threshold=0.5,
        )
        mock_yolo_class.return_value = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            detector = YOLODetector(config)
        assert detector._device == "cpu"

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_uses_cuda_when_available(self, mock_yolo_class, tmp_path):
        """Uses CUDA device when it is available."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake")
        config = YOLOConfig(
            checkpoint_path=str(checkpoint),
            device="cuda",
            input_resolution=640,
            confidence_threshold=0.5,
        )
        mock_yolo_class.return_value = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            detector = YOLODetector(config)
        assert detector._device == "cuda"

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_cpu_device_used_directly(self, mock_yolo_class, valid_config):
        """CPU device is used as-is without checking CUDA."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)
        assert detector._device == "cpu"


# ---------------------------------------------------------------------------
# Tests: Preprocessing — Property 24: YOLO preprocessing consistency
# Validates: Requirement 13.7
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """
    Property 24: YOLO preprocessing consistency.

    For any input frame of arbitrary dimensions, the YOLO_Detector preprocessing
    SHALL produce an output tensor with spatial dimensions matching the configured
    square resolution (320-750) and pixel values normalized with mean [0.485, 0.456, 0.406]
    and std [0.229, 0.224, 0.225].

    **Validates: Requirements 13.7**
    """

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_resizes_to_configured_resolution(self, mock_yolo_class, valid_config):
        """Frame is resized to square resolution from config."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)

        frame = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        result = detector.preprocess(frame)

        assert result.shape == (640, 640, 3)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_applies_normalization(self, mock_yolo_class, valid_config):
        """Frame values are normalized with ImageNet stats."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)

        # All-zero frame: normalized should be -mean/std
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.preprocess(frame)

        expected_channel_0 = -0.485 / 0.229
        assert abs(result[0, 0, 0] - expected_channel_0) < 1e-5

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_with_custom_resolution(self, mock_yolo_class, tmp_path):
        """Preprocessing respects different resolution settings."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("fake")
        config = YOLOConfig(
            checkpoint_path=str(checkpoint),
            device="cpu",
            input_resolution=320,
            confidence_threshold=0.5,
        )
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(config)

        frame = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        result = detector.preprocess(frame)

        assert result.shape == (320, 320, 3)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_with_optical_flow(self, mock_yolo_class, valid_config):
        """Optical flow magnitude appended as 4th channel."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)

        frame = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        flow = np.random.rand(480, 640).astype(np.float32) * 10.0

        result = detector.preprocess(frame, flow_magnitude=flow)

        assert result.shape == (640, 640, 4)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_flow_normalized_to_01(self, mock_yolo_class, valid_config):
        """Optical flow channel is normalized to [0, 1] range."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)

        frame = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        flow = np.ones((100, 100), dtype=np.float32) * 5.0

        result = detector.preprocess(frame, flow_magnitude=flow)

        assert abs(result[:, :, 3].max() - 1.0) < 1e-5

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_preprocess_zero_flow_stays_zero(self, mock_yolo_class, valid_config):
        """Zero optical flow stays zero after normalization."""
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(valid_config)

        frame = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        flow = np.zeros((100, 100), dtype=np.float32)

        result = detector.preprocess(frame, flow_magnitude=flow)

        assert result[:, :, 3].max() == 0.0


# ---------------------------------------------------------------------------
# Tests: Detection and Inference — Property 23: YOLO detection output structure
# Validates: Requirements 13.2, 13.3, 13.6
# ---------------------------------------------------------------------------


class TestDetection:
    """
    Property 23: YOLO detection output structure.

    For any frame processed by the YOLO_Detector, every detection in the output
    SHALL contain a normalized bounding box [x_center, y_center, width, height],
    a class_label from the 17-class set, and a confidence score in [0.0, 1.0].
    No detection with confidence below the configured threshold SHALL appear
    in the output.

    **Validates: Requirements 13.2, 13.3, 13.6**
    """

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_returns_list_per_frame(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """detect() returns one list of detections per frame."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([])]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        assert len(results) == mock_frame_sequence.frame_count
        for frame_dets in results:
            assert isinstance(frame_dets, list)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_parses_detections_correctly(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Detections are parsed into Detection objects with correct fields."""
        mock_result = _make_mock_result([
            (9, 0.85, 0.5, 0.5, 0.1, 0.3),   # Human, conf 0.85
            (0, 0.72, 0.3, 0.4, 0.2, 0.15),   # Boat - With Cargo, conf 0.72
        ])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        for frame_dets in results:
            assert len(frame_dets) == 2
            assert frame_dets[0].class_label == "Human"
            assert abs(frame_dets[0].confidence - 0.85) < 1e-5
            assert frame_dets[1].class_label == "Boat - With Cargo"

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_filters_below_threshold(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Detections below confidence threshold are filtered out."""
        mock_result = _make_mock_result([
            (9, 0.85, 0.5, 0.5, 0.1, 0.3),   # Above threshold (0.5)
            (10, 0.3, 0.2, 0.2, 0.1, 0.2),    # Below threshold
        ])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        for frame_dets in results:
            assert len(frame_dets) == 1
            assert frame_dets[0].class_label == "Human"

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_with_no_detections(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Empty results return empty detection lists."""
        mock_result = _make_mock_result([])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        for frame_dets in results:
            assert frame_dets == []

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_with_optical_flow(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """detect() accepts optical flow magnitude maps."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([])]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        flow_maps = [
            np.random.rand(480, 640).astype(np.float32)
            for _ in range(mock_frame_sequence.frame_count)
        ]

        results = detector.detect(mock_frame_sequence, flow_magnitudes=flow_maps)
        assert len(results) == mock_frame_sequence.frame_count

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_bbox_normalized(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Bounding box coordinates are normalized to [0, 1]."""
        mock_result = _make_mock_result([
            (8, 0.9, 0.75, 0.25, 0.3, 0.4),  # Crane
        ])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        det = results[0][0]
        assert 0.0 <= det.bbox.x_center <= 1.0
        assert 0.0 <= det.bbox.y_center <= 1.0
        assert 0.0 <= det.bbox.width <= 1.0
        assert 0.0 <= det.bbox.height <= 1.0

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detect_all_17_classes(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """All 17 Roboflow classes can be correctly mapped."""
        all_detections = [
            (i, 0.8, 0.5, 0.5, 0.1, 0.1) for i in range(17)
        ]
        mock_result = _make_mock_result(all_detections)
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        labels = {det.class_label for det in results[0]}
        assert labels == set(ROBOFLOW_CLASSES)

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detection_confidence_in_valid_range(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """All returned detections have confidence in [0.0, 1.0]."""
        mock_result = _make_mock_result([
            (9, 0.85, 0.5, 0.5, 0.1, 0.3),
            (1, 0.62, 0.3, 0.7, 0.2, 0.1),
            (8, 0.99, 0.8, 0.2, 0.15, 0.25),
        ])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        for frame_dets in results:
            for det in frame_dets:
                assert 0.0 <= det.confidence <= 1.0

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_detection_class_label_in_17_class_set(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """All returned detections have class_label from the 17-class set."""
        mock_result = _make_mock_result([
            (0, 0.7, 0.5, 0.5, 0.1, 0.1),
            (9, 0.8, 0.3, 0.3, 0.2, 0.3),
            (16, 0.6, 0.7, 0.7, 0.15, 0.15),
        ])
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        results = detector.detect(mock_frame_sequence)

        for frame_dets in results:
            for det in frame_dets:
                assert det.class_label in ROBOFLOW_CLASSES


# ---------------------------------------------------------------------------
# Tests: Inference Calls
# ---------------------------------------------------------------------------


class TestInferenceCalls:
    """Test that inference is called correctly with expected parameters."""

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_predict_called_per_frame(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Model.predict is called once per frame in the sequence."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([])]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        detector.detect(mock_frame_sequence)

        assert mock_model.predict.call_count == mock_frame_sequence.frame_count

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_predict_called_with_correct_params(self, mock_yolo_class, valid_config, mock_frame_sequence):
        """Model.predict receives correct resolution, confidence, and device."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [_make_mock_result([])]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(valid_config)
        detector.detect(mock_frame_sequence)

        call_kwargs = mock_model.predict.call_args_list[0][1]
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["conf"] == 0.5
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["verbose"] is False


# ---------------------------------------------------------------------------
# Tests: Visual Output Generation
# ---------------------------------------------------------------------------


class TestVisualOutputGeneration:
    """
    Generate visual diagnostic outputs for YOLO detection analysis.

    Produces:
    - tests/output/yolo_class_distribution.png
    - tests/output/yolo_confidence_histogram.png
    - tests/output/yolo_feature_distributions.png
    """

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_generate_class_distribution_chart(self, mock_yolo_class, valid_config, mock_frame_sequence, output_dir):
        """Generate class distribution chart from simulated detections."""
        # Create diverse detections across all 17 classes
        rng = np.random.default_rng(123)
        class_counts = {}
        for i, class_name in enumerate(ROBOFLOW_CLASSES):
            count = int(rng.integers(5, 50))
            class_counts[class_name] = count

        # Import visual helpers and generate the chart
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from visual_helpers import plot_class_distribution

        output_path = output_dir / "yolo_class_distribution.png"
        plot_class_distribution(
            class_counts=class_counts,
            output_path=output_path,
            title="YOLO Detection Class Distribution",
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_generate_confidence_histogram(self, mock_yolo_class, valid_config, mock_frame_sequence, output_dir):
        """Generate confidence score histogram from simulated detections."""
        # Simulate confidence scores from a real detection scenario
        rng = np.random.default_rng(456)
        # Mix of high confidence (true positives) and lower confidence (uncertain)
        high_conf = rng.uniform(0.7, 0.99, size=80).tolist()
        med_conf = rng.uniform(0.4, 0.7, size=40).tolist()
        low_conf = rng.uniform(0.1, 0.4, size=20).tolist()
        all_confidences = high_conf + med_conf + low_conf

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from visual_helpers import plot_confidence_histogram

        output_path = output_dir / "yolo_confidence_histogram.png"
        plot_confidence_histogram(
            confidences=all_confidences,
            output_path=output_path,
            title="YOLO Confidence Score Distribution",
            threshold=0.5,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @patch("hazard_detection.yolo_detector.YOLO")
    def test_generate_feature_distributions(self, mock_yolo_class, valid_config, mock_frame_sequence, output_dir):
        """Generate feature distribution plots (bbox dimensions, aspect ratios)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(789)

        # Simulate detection features
        n_detections = 200
        widths = rng.uniform(0.02, 0.5, size=n_detections)
        heights = rng.uniform(0.02, 0.6, size=n_detections)
        aspect_ratios = heights / widths
        x_centers = rng.uniform(0.05, 0.95, size=n_detections)
        y_centers = rng.uniform(0.05, 0.95, size=n_detections)
        areas = widths * heights

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("YOLO Detection Feature Distributions", fontsize=16, fontweight="bold")

        # Width distribution
        axes[0, 0].hist(widths, bins=25, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[0, 0].set_title("BBox Width")
        axes[0, 0].set_xlabel("Normalized Width")
        axes[0, 0].set_ylabel("Frequency")

        # Height distribution
        axes[0, 1].hist(heights, bins=25, color="#2ecc71", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[0, 1].set_title("BBox Height")
        axes[0, 1].set_xlabel("Normalized Height")

        # Aspect ratio distribution
        axes[0, 2].hist(aspect_ratios, bins=25, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[0, 2].set_title("Aspect Ratio (H/W)")
        axes[0, 2].set_xlabel("Ratio")

        # X center distribution
        axes[1, 0].hist(x_centers, bins=25, color="#f39c12", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[1, 0].set_title("X Center Position")
        axes[1, 0].set_xlabel("Normalized X")
        axes[1, 0].set_ylabel("Frequency")

        # Y center distribution
        axes[1, 1].hist(y_centers, bins=25, color="#9b59b6", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[1, 1].set_title("Y Center Position")
        axes[1, 1].set_xlabel("Normalized Y")

        # Area distribution
        axes[1, 2].hist(areas, bins=25, color="#1abc9c", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[1, 2].set_title("BBox Area")
        axes[1, 2].set_xlabel("Normalized Area")

        plt.tight_layout()
        output_path = output_dir / "yolo_feature_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert output_path.exists()
        assert output_path.stat().st_size > 0
