"""
Unit tests for the CNN fallback annotation pipeline
(scripts/annotation/cnn_auto_annotate.py).

Tests cover:
- Chrome-cropping / coordinate-offset math (crop_chrome, offset_boxes_to_original)
- Rectangle-polygon label encoding and round-trip (box_to_yolo_polygon, write_label_file)
- process_image() routing (accept / review / reject / no_detections / unreadable)
- load_model() error paths (missing checkpoint, CUDA->CPU fallback)
- --input_dir default resolution and the image_data_normal -> roboflow data fallback

No GPU, real checkpoint, or network access is required — ultralytics.YOLO and
cv2 I/O are mocked/stubbed throughout.

Validates (see .kiro/specs/cnn-fallback-annotation-pipeline/requirements.md):
    Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 4.1, 4.5, 4.6, 5.1, 5.2, 5.3
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import cnn_auto_annotate.py directly by path (scripts/annotation has no
# __init__.py, so it isn't an importable package on sys.path by default).
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parent.parent.parent / "scripts" / "annotation" / "cnn_auto_annotate.py"
_spec = importlib.util.spec_from_file_location("cnn_auto_annotate", _MODULE_PATH)
cnn_auto_annotate = importlib.util.module_from_spec(_spec)
sys.modules["cnn_auto_annotate"] = cnn_auto_annotate
_spec.loader.exec_module(cnn_auto_annotate)


# ---------------------------------------------------------------------------
# crop_chrome
# ---------------------------------------------------------------------------


class TestCropChrome:
    def test_zero_margins_is_noop_crop(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped, offset = cnn_auto_annotate.crop_chrome(image, 0, 0, 0, 0)
        assert cropped.shape == (100, 200, 3)
        assert offset == (0, 0)

    def test_margins_crop_expected_region(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped, offset = cnn_auto_annotate.crop_chrome(
            image, margin_top=10, margin_bottom=20, margin_left=30, margin_right=40
        )
        # x: [30, 200-40=160) -> width 130 ; y: [10, 100-20=80) -> height 70
        assert cropped.shape == (70, 130, 3)
        assert offset == (30, 10)

    def test_full_frame_margins_clamped_to_minimum_size(self):
        """Margins that would fully consume the frame are clamped so the
        crop region never becomes empty or inverted."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped, offset = cnn_auto_annotate.crop_chrome(
            image, margin_top=200, margin_bottom=200, margin_left=200, margin_right=200
        )
        h, w = cropped.shape[:2]
        assert h >= 1
        assert w >= 1


# ---------------------------------------------------------------------------
# offset_boxes_to_original
# ---------------------------------------------------------------------------


class TestOffsetBoxesToOriginal:
    def test_shifts_boxes_by_offset(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        result = cnn_auto_annotate.offset_boxes_to_original(
            boxes, offset_xy=(30, 10), orig_w=200, orig_h=100
        )
        np.testing.assert_array_equal(result, np.array([[30.0, 10.0, 40.0, 20.0]]))

    def test_clips_to_original_image_bounds(self):
        # Box that would exceed original bounds after applying the offset
        boxes = np.array([[190.0, 90.0, 250.0, 150.0]])
        result = cnn_auto_annotate.offset_boxes_to_original(
            boxes, offset_xy=(30, 10), orig_w=200, orig_h=100
        )
        # x2 = 250+30=280 clipped to 200 ; y2 = 150+10=160 clipped to 100
        assert result[0, 2] == 200
        assert result[0, 3] == 100

    def test_negative_offset_clips_to_zero(self):
        boxes = np.array([[-50.0, -50.0, 5.0, 5.0]])
        result = cnn_auto_annotate.offset_boxes_to_original(
            boxes, offset_xy=(0, 0), orig_w=200, orig_h=100
        )
        assert result[0, 0] == 0
        assert result[0, 1] == 0

    def test_multiple_boxes_shifted_independently(self):
        boxes = np.array([
            [0.0, 0.0, 5.0, 5.0],
            [10.0, 10.0, 20.0, 20.0],
        ])
        result = cnn_auto_annotate.offset_boxes_to_original(
            boxes, offset_xy=(5, 5), orig_w=100, orig_h=100
        )
        np.testing.assert_array_equal(
            result, np.array([[5.0, 5.0, 10.0, 10.0], [15.0, 15.0, 25.0, 25.0]])
        )


# ---------------------------------------------------------------------------
# box_to_yolo_polygon / label round-trip
# ---------------------------------------------------------------------------


class TestBoxToYoloPolygon:
    def test_produces_four_corners_in_expected_order(self):
        box = np.array([10.0, 20.0, 110.0, 220.0])
        polygon = cnn_auto_annotate.box_to_yolo_polygon(box, img_w=200, img_h=400)

        assert len(polygon) == 4
        # top-left, top-right, bottom-right, bottom-left
        expected = [
            (10.0 / 200, 20.0 / 400),
            (110.0 / 200, 20.0 / 400),
            (110.0 / 200, 220.0 / 400),
            (10.0 / 200, 220.0 / 400),
        ]
        for (ax, ay), (ex, ey) in zip(polygon, expected):
            assert ax == pytest.approx(ex)
            assert ay == pytest.approx(ey)

    def test_normalised_coordinates_within_unit_range(self):
        box = np.array([0.0, 0.0, 640.0, 480.0])
        polygon = cnn_auto_annotate.box_to_yolo_polygon(box, img_w=640, img_h=480)
        for x, y in polygon:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0


class TestLabelRoundTrip:
    def test_write_and_reparse_preserves_class_and_coords(self, tmp_path):
        annotations = [
            (2, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]),
            (9, [(0.5, 0.5), (0.6, 0.5), (0.6, 0.6), (0.5, 0.6)]),
        ]
        label_path = tmp_path / "test.txt"
        cnn_auto_annotate.write_label_file(label_path, annotations)

        lines = label_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        reparsed = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            reparsed.append((class_id, points))

        for (orig_cid, orig_pts), (new_cid, new_pts) in zip(annotations, reparsed):
            assert orig_cid == new_cid
            for (ox, oy), (nx, ny) in zip(orig_pts, new_pts):
                assert ox == pytest.approx(nx, abs=1e-6)
                assert oy == pytest.approx(ny, abs=1e-6)


# ---------------------------------------------------------------------------
# process_image routing
# ---------------------------------------------------------------------------


def _make_mock_results(boxes_xyxy, cls_ids, confs):
    """Build a stubbed Ultralytics Results list as returned by model.predict()."""
    result = MagicMock()
    if len(boxes_xyxy) == 0:
        result.boxes = None
        return [result]

    boxes = MagicMock()
    boxes.__len__ = lambda self: len(boxes_xyxy)
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(boxes_xyxy, dtype=float)
    boxes.cls = list(cls_ids)
    boxes.conf = list(confs)
    result.boxes = boxes
    return [result]


def _default_args(**overrides):
    base = dict(
        chrome_top=0, chrome_bottom=0, chrome_left=0, chrome_right=0,
        imgsz=640, confidence=0.35, review_threshold=0.55, classes=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestProcessImageRouting:
    @patch("cnn_auto_annotate.cv2.imread")
    def test_unreadable_image_returns_unreadable_status(self, mock_imread):
        mock_imread.return_value = None
        model = MagicMock()
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert result["status"] == "unreadable"
        assert result["accepted"] == []
        assert result["review"] == []

    @patch("cnn_auto_annotate.cv2.imread")
    def test_no_detections_returns_no_detections_status(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        model.predict.return_value = _make_mock_results([], [], [])
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert result["status"] == "no_detections"

    @patch("cnn_auto_annotate.cv2.imread")
    def test_confidence_above_review_threshold_routes_to_accepted(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        # class 8 (Crane) is not safety-critical; review_threshold=0.55
        model.predict.return_value = _make_mock_results(
            [[10.0, 10.0, 50.0, 50.0]], [8], [0.9]
        )
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert result["status"] == "ok"
        assert len(result["accepted"]) == 1
        assert len(result["review"]) == 0
        assert result["accepted"][0]["class_id"] == 8

    @patch("cnn_auto_annotate.cv2.imread")
    def test_confidence_between_thresholds_routes_to_review(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        # confidence 0.4 is above --confidence (0.35 used only by the model
        # upstream) but below review_threshold (0.55), non-safety class
        model.predict.return_value = _make_mock_results(
            [[10.0, 10.0, 50.0, 50.0]], [8], [0.4]
        )
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert len(result["review"]) == 1
        assert len(result["accepted"]) == 0

    @patch("cnn_auto_annotate.cv2.imread")
    def test_safety_critical_class_uses_lowered_effective_threshold(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        # class 9 (Human) is safety-critical: effective threshold =
        # min(0.55, 0.55*0.85) = 0.4675. A confidence of 0.5 clears that
        # lowered bar and should be accepted, even though it's still below
        # the plain review_threshold of 0.55.
        model.predict.return_value = _make_mock_results(
            [[10.0, 10.0, 50.0, 50.0]], [9], [0.5]
        )
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert len(result["accepted"]) == 1
        assert result["accepted"][0]["class_id"] == 9

    @patch("cnn_auto_annotate.cv2.imread")
    def test_unknown_class_id_is_rejected(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        model.predict.return_value = _make_mock_results(
            [[10.0, 10.0, 50.0, 50.0]], [99], [0.9]
        )
        result = cnn_auto_annotate.process_image(
            Path("fake.png"), model, "cpu", _default_args()
        )
        assert len(result["rejected"]) == 1
        assert result["rejected"][0]["reason"] == "unknown_class"

    @patch("cnn_auto_annotate.cv2.imread")
    def test_classes_filter_excludes_non_matching_detections(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        model = MagicMock()
        model.predict.return_value = _make_mock_results(
            [[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]],
            [8, 9],
            [0.9, 0.9],
        )
        args = _default_args(classes=[9])
        result = cnn_auto_annotate.process_image(Path("fake.png"), model, "cpu", args)
        total_kept = len(result["accepted"]) + len(result["review"])
        assert total_kept == 1


# ---------------------------------------------------------------------------
# load_model error paths
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError, match="YOLO checkpoint not found"):
            cnn_auto_annotate.load_model(str(missing), "cpu")

    def test_resolve_device_falls_back_to_cpu_without_cuda(self):
        with patch.object(cnn_auto_annotate, "_cuda_available", return_value=False):
            assert cnn_auto_annotate._resolve_device("cuda") == "cpu"

    def test_resolve_device_keeps_cuda_when_available(self):
        with patch.object(cnn_auto_annotate, "_cuda_available", return_value=True):
            assert cnn_auto_annotate._resolve_device("cuda") == "cuda"

    def test_resolve_device_passes_through_cpu(self):
        assert cnn_auto_annotate._resolve_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# --input_dir default resolution (image_data_normal -> roboflow data fallback)
# ---------------------------------------------------------------------------


class TestInputDirResolution:
    def test_prefers_image_data_normal_when_present(self):
        with patch.object(Path, "exists", return_value=True):
            assert cnn_auto_annotate.resolve_default_input_dir() == cnn_auto_annotate.PREFERRED_INPUT_DIR

    def test_falls_back_to_roboflow_data_when_absent(self):
        with patch.object(Path, "exists", return_value=False):
            assert cnn_auto_annotate.resolve_default_input_dir() == cnn_auto_annotate.FALLBACK_INPUT_DIR
