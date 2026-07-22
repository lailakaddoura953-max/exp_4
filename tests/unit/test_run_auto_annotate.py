"""
Unit tests for the auto-annotation pipeline selector
(scripts/annotation/run_auto_annotate.py).

Tests cover:
- build_segmentation_args / build_cnn_args field completeness against each
  backend's own parse_args() field names
- Default --output_dir resolution per pipeline choice
- --pipeline choices restriction (segmentation/cnn only)
- The image_data_normal -> roboflow data default --input_dir fallback

No GPU, real checkpoint, or network access is required. The `auto_annotate`
and `cnn_auto_annotate` backend modules are pre-seeded into sys.modules as
mocks before main() runs, so main()'s `import auto_annotate as backend` /
`import cnn_auto_annotate as backend` statements bind those mocks instead of
loading and executing the real modules.

Validates (see .kiro/specs/cnn-fallback-annotation-pipeline/requirements.md):
    Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.5, 5.5
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import run_auto_annotate.py directly by path (scripts/annotation has no
# __init__.py, so it isn't an importable package on sys.path by default).
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).parent.parent.parent / "scripts" / "annotation" / "run_auto_annotate.py"
_spec = importlib.util.spec_from_file_location("run_auto_annotate", _MODULE_PATH)
run_auto_annotate = importlib.util.module_from_spec(_spec)
sys.modules["run_auto_annotate"] = run_auto_annotate
_spec.loader.exec_module(run_auto_annotate)


# Field names each backend's own parse_args() Namespace is expected to carry,
# cross-checked by hand against auto_annotate.parse_args() and
# cnn_auto_annotate.parse_args() (scripts/annotation/*.py).
SEGMENTATION_EXPECTED_FIELDS = {
    "input_dir", "output_dir", "confidence", "review_threshold",
    "chrome_top", "chrome_bottom", "chrome_left", "chrome_right",
    "simplify", "classes", "limit", "dry_run", "verify",
}
CNN_EXPECTED_FIELDS = {
    "input_dir", "output_dir", "checkpoint", "confidence", "review_threshold",
    "imgsz", "device", "chrome_top", "chrome_bottom", "chrome_left",
    "chrome_right", "classes", "limit", "dry_run", "verify",
}


def _shared_namespace(**overrides) -> argparse.Namespace:
    """Build a Namespace with every field the dispatcher's own argparse setup
    would produce, so build_segmentation_args/build_cnn_args have everything
    they read available on it."""
    base = dict(
        pipeline="segmentation",
        input_dir="roboflow data",
        output_dir="image_data_annotated",
        confidence=0.35,
        review_threshold=0.55,
        chrome_top=60, chrome_bottom=30, chrome_left=220, chrome_right=10,
        classes=None, limit=None, dry_run=False, verify=False,
        simplify=2.0,
        checkpoint="runs/train/hazard_yolo/weights/best.pt",
        imgsz=640, device="cuda",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# build_segmentation_args / build_cnn_args field completeness
# ---------------------------------------------------------------------------


class TestArgMapping:
    def test_build_segmentation_args_has_all_expected_fields(self):
        result = run_auto_annotate.build_segmentation_args(_shared_namespace())
        assert isinstance(result, SimpleNamespace)
        assert set(vars(result).keys()) == SEGMENTATION_EXPECTED_FIELDS

    def test_build_cnn_args_has_all_expected_fields(self):
        result = run_auto_annotate.build_cnn_args(_shared_namespace())
        assert isinstance(result, SimpleNamespace)
        assert set(vars(result).keys()) == CNN_EXPECTED_FIELDS

    def test_build_segmentation_args_preserves_values(self):
        ns = _shared_namespace(confidence=0.42, input_dir="some/path")
        result = run_auto_annotate.build_segmentation_args(ns)
        assert result.confidence == 0.42
        assert result.input_dir == "some/path"

    def test_build_cnn_args_preserves_values(self):
        ns = _shared_namespace(checkpoint="my/checkpoint.pt", device="cpu")
        result = run_auto_annotate.build_cnn_args(ns)
        assert result.checkpoint == "my/checkpoint.pt"
        assert result.device == "cpu"

    def test_build_cnn_args_does_not_leak_segmentation_only_field(self):
        result = run_auto_annotate.build_cnn_args(_shared_namespace())
        assert "simplify" not in vars(result)

    def test_build_segmentation_args_does_not_leak_cnn_only_fields(self):
        result = run_auto_annotate.build_segmentation_args(_shared_namespace())
        assert "checkpoint" not in vars(result)
        assert "device" not in vars(result)
        assert "imgsz" not in vars(result)


# ---------------------------------------------------------------------------
# main() dispatch: --output_dir default resolution, --pipeline choices,
# and --input_dir fallback -- exercised via mocked backend modules so no
# real annotation logic executes.
# ---------------------------------------------------------------------------


def _install_mock_backends():
    """Seed sys.modules with mocks for both backends so main()'s dynamic
    `import auto_annotate as backend` / `import cnn_auto_annotate as backend`
    bind these mocks instead of loading the real files."""
    seg_mock = MagicMock()
    cnn_mock = MagicMock()
    sys.modules["auto_annotate"] = seg_mock
    sys.modules["cnn_auto_annotate"] = cnn_mock
    return seg_mock, cnn_mock


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Prevent mocked 'auto_annotate'/'cnn_auto_annotate' entries installed by
    _install_mock_backends() from leaking into other test files that may run
    later in the same pytest session and expect the real modules."""
    saved = {
        name: sys.modules.get(name) for name in ("auto_annotate", "cnn_auto_annotate")
    }
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestMainDispatch:
    def test_default_output_dir_for_segmentation(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--pipeline", "segmentation", "--dry_run"],
        )
        run_auto_annotate.main()
        called_args = seg_mock.run.call_args[0][0]
        assert called_args.output_dir == "image_data_annotated"

    def test_default_output_dir_for_cnn(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--pipeline", "cnn", "--dry_run"],
        )
        run_auto_annotate.main()
        called_args = cnn_mock.run.call_args[0][0]
        assert called_args.output_dir == "image_data_annotated_cnn"

    def test_explicit_output_dir_is_preserved_for_either_pipeline(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--pipeline", "cnn",
             "--output_dir", "custom_output", "--dry_run"],
        )
        run_auto_annotate.main()
        called_args = cnn_mock.run.call_args[0][0]
        assert called_args.output_dir == "custom_output"

    def test_default_pipeline_is_segmentation(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(sys, "argv", ["run_auto_annotate.py", "--dry_run"])
        run_auto_annotate.main()
        seg_mock.run.assert_called_once()
        cnn_mock.run.assert_not_called()

    def test_verify_flag_routes_to_run_verify(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--pipeline", "cnn", "--verify"],
        )
        run_auto_annotate.main()
        cnn_mock.run_verify.assert_called_once()
        cnn_mock.run.assert_not_called()

    def test_invalid_pipeline_choice_exits_nonzero(self, monkeypatch):
        _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--pipeline", "not_a_real_pipeline"],
        )
        with pytest.raises(SystemExit) as exc_info:
            run_auto_annotate.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --input_dir default fallback (image_data_normal -> roboflow data)
# ---------------------------------------------------------------------------


class TestInputDirDefaultFallback:
    def test_defaults_to_image_data_normal_when_present(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        with patch.object(Path, "exists", return_value=True):
            monkeypatch.setattr(sys, "argv", ["run_auto_annotate.py", "--dry_run"])
            run_auto_annotate.main()
        called_args = seg_mock.run.call_args[0][0]
        assert called_args.input_dir == "image_data_normal"

    def test_falls_back_to_roboflow_data_when_absent(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        with patch.object(Path, "exists", return_value=False):
            monkeypatch.setattr(sys, "argv", ["run_auto_annotate.py", "--dry_run"])
            run_auto_annotate.main()
        called_args = seg_mock.run.call_args[0][0]
        assert called_args.input_dir == "roboflow data"

    def test_explicit_input_dir_overrides_default(self, monkeypatch):
        seg_mock, cnn_mock = _install_mock_backends()
        monkeypatch.setattr(
            sys, "argv",
            ["run_auto_annotate.py", "--input_dir", "some/other/dataset", "--dry_run"],
        )
        run_auto_annotate.main()
        called_args = seg_mock.run.call_args[0][0]
        assert called_args.input_dir == "some/other/dataset"
