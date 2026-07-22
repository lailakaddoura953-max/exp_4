"""
Unit tests for the pre-training hazard sanity check
(scripts/pretrain_hazard_sanity_check.py).

Tests cover:
- iou_xyxy / polygon_to_bbox geometry helpers
- ClassRecall.recall, including the zero-ground-truth vacuous-pass case
- parse_label_line
- build_yolo_split train/val partitioning, data.yaml contents, and the
  "extremely small dataset" guard
- main() precondition failures (--skip_generation with a missing synthetic
  dir; --skip_training without/with a missing --checkpoint)
- resolve_default_normal_dir and the image_data_normal -> roboflow data
  fallback warning behavior

No GPU, real checkpoint, or network access is required. Training
(train_sanity_check_model) and evaluation (evaluate_hazard_recall) are
mocked wherever main() is exercised end-to-end.

Validates (see .kiro/specs/cnn-fallback-annotation-pipeline/requirements.md):
    Requirements 3.2, 3.7, 3.8, 4.5, 4.6, 4.7, 5.4
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import pretrain_hazard_sanity_check.py directly by path. It lives in
# scripts/ (not scripts/annotation/) and itself does
# `sys.path.insert(0, str(Path(__file__).resolve().parent))` followed by
# `import generate_hazard_augmentations as hazard_gen`, so loading it this
# way (with its real __file__ pointing at scripts/) lets that internal
# import resolve normally.
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parent.parent.parent / "scripts" / "pretrain_hazard_sanity_check.py"
_spec = importlib.util.spec_from_file_location("pretrain_hazard_sanity_check", _MODULE_PATH)
sanity_check = importlib.util.module_from_spec(_spec)
sys.modules["pretrain_hazard_sanity_check"] = sanity_check
_spec.loader.exec_module(sanity_check)


# ---------------------------------------------------------------------------
# iou_xyxy
# ---------------------------------------------------------------------------


class TestIouXyxy:
    def test_identical_boxes_have_iou_one(self):
        box = (0.1, 0.1, 0.5, 0.5)
        assert sanity_check.iou_xyxy(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        box_a = (0.0, 0.0, 0.2, 0.2)
        box_b = (0.5, 0.5, 0.7, 0.7)
        assert sanity_check.iou_xyxy(box_a, box_b) == 0.0

    def test_partial_overlap_matches_known_value(self):
        # box_a: [0,0]-[1,1] area=1 ; box_b: [0.5,0.5]-[1.5,1.5] area=1
        # intersection: [0.5,0.5]-[1,1] area=0.25 ; union = 1+1-0.25 = 1.75
        box_a = (0.0, 0.0, 1.0, 1.0)
        box_b = (0.5, 0.5, 1.5, 1.5)
        expected = 0.25 / 1.75
        assert sanity_check.iou_xyxy(box_a, box_b) == pytest.approx(expected)

    def test_touching_edges_have_iou_zero(self):
        box_a = (0.0, 0.0, 0.5, 0.5)
        box_b = (0.5, 0.0, 1.0, 0.5)
        assert sanity_check.iou_xyxy(box_a, box_b) == 0.0


# ---------------------------------------------------------------------------
# polygon_to_bbox
# ---------------------------------------------------------------------------


class TestPolygonToBbox:
    def test_rectangle_polygon_returns_its_corners(self):
        points = [(0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (0.1, 0.6)]
        assert sanity_check.polygon_to_bbox(points) == pytest.approx((0.1, 0.2, 0.4, 0.6))

    def test_irregular_polygon_returns_enclosing_bbox(self):
        points = [(0.2, 0.5), (0.5, 0.1), (0.8, 0.5), (0.5, 0.9)]  # a diamond
        result = sanity_check.polygon_to_bbox(points)
        assert result == pytest.approx((0.2, 0.1, 0.8, 0.9))


# ---------------------------------------------------------------------------
# ClassRecall.recall
# ---------------------------------------------------------------------------


class TestClassRecall:
    def test_normal_case_computes_detected_over_ground_truth(self):
        result = sanity_check.ClassRecall(class_id=9, class_name="Human", ground_truth_count=4, detected_count=3)
        assert result.recall == pytest.approx(0.75)

    def test_zero_ground_truth_is_vacuous_pass(self):
        result = sanity_check.ClassRecall(class_id=2, class_name="Container - Open", ground_truth_count=0, detected_count=0)
        assert result.recall == 1.0

    def test_zero_detected_nonzero_ground_truth_is_zero(self):
        result = sanity_check.ClassRecall(class_id=10, class_name="Human - No Safety Clothes", ground_truth_count=5, detected_count=0)
        assert result.recall == 0.0

    def test_perfect_recall(self):
        result = sanity_check.ClassRecall(class_id=9, class_name="Human", ground_truth_count=5, detected_count=5)
        assert result.recall == 1.0


# ---------------------------------------------------------------------------
# HazardGroupRecall / evaluate_multi_hazard_breakdown
# ---------------------------------------------------------------------------


class TestHazardGroupRecall:
    def test_normal_case_computes_detected_over_ground_truth(self):
        group = sanity_check.HazardGroupRecall("multi_hazard", image_count=3, ground_truth_count=6, detected_count=3)
        assert group.recall == pytest.approx(0.5)

    def test_zero_ground_truth_is_vacuous_pass(self):
        group = sanity_check.HazardGroupRecall("multi_hazard", image_count=0, ground_truth_count=0, detected_count=0)
        assert group.recall == 1.0


class _FakeBoxes:
    """Minimal stand-in for ultralytics' Boxes object, exposing just the
    .xyxyn / .cls / len() surface evaluate_multi_hazard_breakdown reads.
    __len__ must be defined on the class (not assigned per-instance) for
    Python's len() builtin to find it -- that's the bug this class fixes
    relative to an earlier attempt with types.SimpleNamespace."""
    def __init__(self, boxes_list, cls_list):
        import numpy as np
        self._xyxyn = np.array(boxes_list)
        self._cls = np.array(cls_list)

    def __len__(self):
        return len(self._xyxyn)

    @property
    def xyxyn(self):
        import types
        return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: self._xyxyn))

    @property
    def cls(self):
        import types
        return types.SimpleNamespace(
            cpu=lambda: types.SimpleNamespace(
                numpy=lambda: types.SimpleNamespace(astype=lambda t: self._cls.astype(t))
            )
        )


class TestEvaluateMultiHazardBreakdown:
    def _mock_predict_matching_gt(self, monkeypatch):
        """Patch ultralytics.YOLO so predictions exactly match whatever
        ground-truth boxes are read from each label file, giving perfect
        recall -- lets tests isolate the single/multi grouping logic itself
        rather than detection accuracy. Uses real numpy arrays (already a
        project dependency) so .tolist()/.astype(int)/iteration all behave
        exactly like the real ultralytics Boxes API without a hand-rolled
        fake ndarray."""
        import types
        import numpy as np

        def _fake_predict(self, source, imgsz, conf, device, verbose):
            label_path = Path(source).with_suffix(".txt")
            boxes_list = []
            cls_list = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = sanity_check.parse_label_line(line)
                if parsed is None:
                    continue
                cid, points = parsed
                boxes_list.append(sanity_check.polygon_to_bbox(points))
                cls_list.append(cid)

            result = types.SimpleNamespace()
            if boxes_list:
                result.boxes = _FakeBoxes(boxes_list, cls_list)
            else:
                result.boxes = None
            return [result]

        class FakeYOLO:
            def __init__(self, path):
                pass
            predict = _fake_predict

        # evaluate_multi_hazard_breakdown does `from ultralytics import YOLO`
        # inside the function body, so patch it at the ultralytics module level.
        import ultralytics
        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

    def test_groups_images_by_distinct_hazard_class_count(self, tmp_path, monkeypatch):
        self._mock_predict_matching_gt(monkeypatch)

        # One single-hazard image (class 9 only), one multi-hazard image
        # (classes 2 and 9 both present).
        single_img = tmp_path / "single.png"
        single_img.write_bytes(b"fake")
        single_lbl = tmp_path / "single.txt"
        single_lbl.write_text("9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")

        multi_img = tmp_path / "multi.png"
        multi_img.write_bytes(b"fake")
        multi_lbl = tmp_path / "multi.txt"
        multi_lbl.write_text(
            "9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
            "2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n",
            encoding="utf-8",
        )

        val_pairs = [(single_img, single_lbl), (multi_img, multi_lbl)]
        groups = sanity_check.evaluate_multi_hazard_breakdown(
            checkpoint_path=Path("fake.pt"), val_pairs=val_pairs,
            hazard_classes={2: "Container - Open", 9: "Human", 10: "Human - No Safety Clothes"},
            conf_threshold=0.35, iou_threshold=0.4, device="cpu", imgsz=640,
        )

        assert groups["single_hazard"].image_count == 1
        assert groups["multi_hazard"].image_count == 1
        assert groups["single_hazard"].ground_truth_count == 1
        assert groups["multi_hazard"].ground_truth_count == 2

    def test_perfect_predictions_yield_full_recall_in_both_groups(self, tmp_path, monkeypatch):
        self._mock_predict_matching_gt(monkeypatch)

        multi_img = tmp_path / "multi.png"
        multi_img.write_bytes(b"fake")
        multi_lbl = tmp_path / "multi.txt"
        multi_lbl.write_text(
            "9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
            "2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n",
            encoding="utf-8",
        )

        groups = sanity_check.evaluate_multi_hazard_breakdown(
            checkpoint_path=Path("fake.pt"), val_pairs=[(multi_img, multi_lbl)],
            hazard_classes={2: "Container - Open", 9: "Human", 10: "Human - No Safety Clothes"},
            conf_threshold=0.35, iou_threshold=0.4, device="cpu", imgsz=640,
        )
        assert groups["multi_hazard"].recall == pytest.approx(1.0)

    def test_no_ground_truth_annotations_are_skipped(self, tmp_path, monkeypatch):
        self._mock_predict_matching_gt(monkeypatch)

        img = tmp_path / "empty.png"
        img.write_bytes(b"fake")
        lbl = tmp_path / "empty.txt"
        lbl.write_text("", encoding="utf-8")

        groups = sanity_check.evaluate_multi_hazard_breakdown(
            checkpoint_path=Path("fake.pt"), val_pairs=[(img, lbl)],
            hazard_classes={2: "Container - Open", 9: "Human", 10: "Human - No Safety Clothes"},
            conf_threshold=0.35, iou_threshold=0.4, device="cpu", imgsz=640,
        )
        assert groups["single_hazard"].image_count == 0
        assert groups["multi_hazard"].image_count == 0


# ---------------------------------------------------------------------------
# parse_label_line
# ---------------------------------------------------------------------------


class TestParseLabelLine:
    def test_valid_line_parses_class_and_points(self):
        line = "9 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4"
        result = sanity_check.parse_label_line(line)
        assert result is not None
        class_id, points = result
        assert class_id == 9
        assert points == [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]

    def test_too_few_tokens_returns_none(self):
        assert sanity_check.parse_label_line("9 0.1 0.2") is None

    def test_odd_coordinate_count_returns_none(self):
        assert sanity_check.parse_label_line("9 0.1 0.2 0.3 0.2 0.3") is None

    def test_non_numeric_tokens_return_none(self):
        assert sanity_check.parse_label_line("9 a b c d e f") is None


# ---------------------------------------------------------------------------
# build_yolo_split
# ---------------------------------------------------------------------------


def _make_fake_pair(directory: Path, name: str, class_id: int = 9) -> None:
    """Create a minimal fake image + YOLO polygon label file pair on disk.
    build_yolo_split only copies these files -- their content doesn't need
    to be a real decodable image."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.png").write_bytes(b"not a real png, just needs to exist")
    (directory / f"{name}.txt").write_text(
        f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
    )


class TestBuildYoloSplit:
    def test_partitions_respect_val_fraction(self, tmp_path):
        synthetic_dir = tmp_path / "synthetic"
        for i in range(10):
            _make_fake_pair(synthetic_dir, f"img_{i:03d}")

        split_dir = tmp_path / "split"
        data_yaml, val_pairs = sanity_check.build_yolo_split(
            synthetic_dir=synthetic_dir, split_dir=split_dir,
            val_fraction=0.2, seed=42, class_names=["a", "b"],
        )

        assert len(val_pairs) == 2  # 20% of 10
        train_images = list((split_dir / "train" / "images").glob("*.png"))
        assert len(train_images) == 8

    def test_data_yaml_has_correct_nc_and_names(self, tmp_path):
        synthetic_dir = tmp_path / "synthetic"
        for i in range(5):
            _make_fake_pair(synthetic_dir, f"img_{i:03d}")

        split_dir = tmp_path / "split"
        data_yaml, _ = sanity_check.build_yolo_split(
            synthetic_dir=synthetic_dir, split_dir=split_dir,
            val_fraction=0.2, seed=1, class_names=["Human", "Container - Open"],
        )

        text = data_yaml.read_text(encoding="utf-8")
        assert "nc: 2" in text
        assert "Human" in text
        assert "Container - Open" in text
        assert "train: train/images" in text
        assert "val: val/images" in text

    def test_extremely_small_dataset_guarantees_one_training_example(self, tmp_path):
        synthetic_dir = tmp_path / "synthetic"
        # Only 1 pair total; val_fraction would take n_val = max(1, int(1*0.2)) = 1,
        # leaving train_pairs empty without the guard.
        _make_fake_pair(synthetic_dir, "only_one")

        split_dir = tmp_path / "split"
        data_yaml, val_pairs = sanity_check.build_yolo_split(
            synthetic_dir=synthetic_dir, split_dir=split_dir,
            val_fraction=0.2, seed=1, class_names=["a"],
        )

        train_images = list((split_dir / "train" / "images").glob("*.png"))
        assert len(train_images) == 1
        assert len(val_pairs) == 0

    def test_no_pairs_found_exits_nonzero(self, tmp_path):
        empty_synthetic = tmp_path / "empty_synthetic"
        empty_synthetic.mkdir()
        split_dir = tmp_path / "split"

        with pytest.raises(SystemExit) as exc_info:
            sanity_check.build_yolo_split(
                synthetic_dir=empty_synthetic, split_dir=split_dir,
                val_fraction=0.2, seed=1, class_names=["a"],
            )
        assert exc_info.value.code != 0

    def test_val_pairs_reference_split_dir_not_synthetic_dir(self, tmp_path):
        synthetic_dir = tmp_path / "synthetic"
        for i in range(10):
            _make_fake_pair(synthetic_dir, f"img_{i:03d}")

        split_dir = tmp_path / "split"
        _, val_pairs = sanity_check.build_yolo_split(
            synthetic_dir=synthetic_dir, split_dir=split_dir,
            val_fraction=0.2, seed=42, class_names=["a"],
        )
        for img_path, label_path in val_pairs:
            assert str(split_dir) in str(img_path)
            assert str(split_dir) in str(label_path)


# ---------------------------------------------------------------------------
# resolve_default_normal_dir / image_data_normal -> roboflow data fallback
# ---------------------------------------------------------------------------


class TestResolveDefaultNormalDir:
    def test_prefers_image_data_normal_when_present(self):
        with patch.object(Path, "exists", return_value=True):
            assert str(sanity_check.resolve_default_normal_dir()) == sanity_check.PREFERRED_NORMAL_DIR

    def test_falls_back_to_roboflow_data_when_absent(self):
        with patch.object(Path, "exists", return_value=False):
            assert str(sanity_check.resolve_default_normal_dir()) == sanity_check.FALLBACK_NORMAL_DIR


# ---------------------------------------------------------------------------
# main() precondition failures
# ---------------------------------------------------------------------------


class TestMainPreconditionFailures:
    def test_skip_generation_with_missing_synthetic_dir_exits_nonzero(self, tmp_path, monkeypatch):
        missing_dir = tmp_path / "does_not_exist"
        monkeypatch.setattr(sys, "argv", [
            "pretrain_hazard_sanity_check.py",
            "--skip_generation",
            "--synthetic_dir", str(missing_dir),
        ])
        with pytest.raises(SystemExit) as exc_info:
            sanity_check.main()
        assert exc_info.value.code != 0

    def test_skip_training_without_checkpoint_exits_nonzero(self, tmp_path, monkeypatch):
        # Populate a valid synthetic_dir so Step 1 (skipped) and Step 2
        # (build_yolo_split) succeed, so the failure under test is really
        # the --skip_training precondition and not an earlier one.
        synthetic_dir = tmp_path / "synthetic"
        for i in range(5):
            _make_fake_pair(synthetic_dir, f"img_{i:03d}")
        split_dir = tmp_path / "split"

        monkeypatch.setattr(sys, "argv", [
            "pretrain_hazard_sanity_check.py",
            "--skip_generation", "--synthetic_dir", str(synthetic_dir),
            "--split_dir", str(split_dir),
            "--skip_training",  # no --checkpoint given
        ])
        with pytest.raises(SystemExit) as exc_info:
            sanity_check.main()
        assert exc_info.value.code != 0

    def test_skip_training_with_missing_checkpoint_exits_nonzero(self, tmp_path, monkeypatch):
        synthetic_dir = tmp_path / "synthetic"
        for i in range(5):
            _make_fake_pair(synthetic_dir, f"img_{i:03d}")
        split_dir = tmp_path / "split"
        missing_checkpoint = tmp_path / "no_such_checkpoint.pt"

        monkeypatch.setattr(sys, "argv", [
            "pretrain_hazard_sanity_check.py",
            "--skip_generation", "--synthetic_dir", str(synthetic_dir),
            "--split_dir", str(split_dir),
            "--skip_training", "--checkpoint", str(missing_checkpoint),
        ])
        with pytest.raises(SystemExit) as exc_info:
            sanity_check.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# main() -- roboflow-data fallback warning (mocked training/evaluation so no
# real model work happens)
# ---------------------------------------------------------------------------


class TestMainRoboflowFallbackWarning:
    def _run_main_with_mocks(self, tmp_path, monkeypatch, normal_dir_arg, capsys):
        synthetic_dir = tmp_path / "synthetic"
        split_dir = tmp_path / "split"
        checkpoint_path = tmp_path / "fake_checkpoint.pt"
        checkpoint_path.write_bytes(b"fake")

        def _fake_generate(roboflow_dir, normal_dir, synthetic_dir, injections_per_image, max_images, seed):
            for i in range(5):
                _make_fake_pair(synthetic_dir, f"img_{i:03d}")

        monkeypatch.setattr(sanity_check, "generate_synthetic_hazards", _fake_generate)
        monkeypatch.setattr(
            sanity_check, "evaluate_hazard_recall",
            lambda **kwargs: {
                2: sanity_check.ClassRecall(2, "Container - Open", 1, 1),
                9: sanity_check.ClassRecall(9, "Human", 1, 1),
                10: sanity_check.ClassRecall(10, "Human - No Safety Clothes", 1, 1),
            },
        )
        monkeypatch.setattr(
            sanity_check, "evaluate_multi_hazard_breakdown",
            lambda **kwargs: {
                "single_hazard": sanity_check.HazardGroupRecall("single_hazard", 4, 3, 3),
                "multi_hazard": sanity_check.HazardGroupRecall("multi_hazard", 1, 2, 1),
            },
        )

        argv = [
            "pretrain_hazard_sanity_check.py",
            "--normal_dir", normal_dir_arg,
            "--synthetic_dir", str(synthetic_dir),
            "--split_dir", str(split_dir),
            "--skip_training", "--checkpoint", str(checkpoint_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            sanity_check.main()
        return capsys.readouterr().out

    def test_fallback_to_roboflow_data_prints_caveat_warning(self, tmp_path, monkeypatch, capsys):
        # normal_dir explicitly set to the fallback path itself -- this
        # exercises the "already using roboflow data" warning branch
        # directly regardless of whether image_data_normal exists on the
        # machine running the test.
        output = self._run_main_with_mocks(
            tmp_path, monkeypatch, sanity_check.FALLBACK_NORMAL_DIR, capsys
        )
        assert "mechanics" in output.lower() or "not representative" in output.lower()

    def test_explicit_normal_dir_that_exists_does_not_warn(self, tmp_path, monkeypatch, capsys):
        real_dir = tmp_path / "my_real_normal_dir"
        real_dir.mkdir()
        output = self._run_main_with_mocks(tmp_path, monkeypatch, str(real_dir), capsys)
        assert "not representative" not in output.lower()
