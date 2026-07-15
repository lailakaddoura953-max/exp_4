"""
Unit tests for hazard_detection.evaluation module.

Tests the individual plot functions directly using synthetic numpy data
(17 classes, no real YOLO model required).

Visual outputs saved to tests/output/:
  - eval_confusion_matrix.png
  - eval_class_spread.png
  - eval_feature_distributions.png
  - eval_roc_curves.png
  - eval_confidence_distribution.png
  - eval_summary.json

Requirements: 13.2, 13.6
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hazard_detection.evaluation import (
    NUM_CLASSES,
    ROBOFLOW_CLASS_NAMES,
    plot_classification_report,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    save_evaluation_summary,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_samples: int = 300, n_classes: int = NUM_CLASSES, seed: int = 0):
    """
    Build synthetic y_true, y_pred, y_probs arrays with ``n_classes`` classes.

    y_true  — random class indices drawn uniformly from [0, n_classes)
    y_pred  — ~80 % correct predictions plus ~20 % random misclassifications
    y_probs — softmax-like probability matrix (row sums to 1)
    """
    rng = np.random.default_rng(seed)

    # Ensure every class has at least one sample so ROC curves can be drawn
    reps = max(1, n_samples // n_classes)
    y_true = np.tile(np.arange(n_classes), reps)
    # Pad to exactly n_samples
    extra = n_samples - len(y_true)
    if extra > 0:
        y_true = np.concatenate([y_true, rng.integers(0, n_classes, size=extra)])
    rng.shuffle(y_true)
    y_true = y_true[:n_samples].astype(np.int64)

    # ~80 % correct
    y_pred = y_true.copy()
    flip_mask = rng.random(n_samples) < 0.20
    y_pred[flip_mask] = rng.integers(0, n_classes, size=int(flip_mask.sum()))
    y_pred = y_pred.astype(np.int64)

    # Probability matrix: high score for predicted class, noise elsewhere
    raw = rng.uniform(0.01, 0.1, size=(n_samples, n_classes)).astype(np.float32)
    for i, pred in enumerate(y_pred):
        raw[i, pred] += rng.uniform(0.5, 0.9)
    row_sums = raw.sum(axis=1, keepdims=True)
    y_probs = raw / row_sums  # normalise to sum-to-1

    return y_true, y_pred, y_probs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """Module-scoped synthetic detection data with all 17 classes represented."""
    return _make_synthetic_data(n_samples=340, n_classes=NUM_CLASSES, seed=42)


# ---------------------------------------------------------------------------
# Tests — plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def test_creates_png_file(self, synthetic_data, output_dir):
        y_true, y_pred, _ = synthetic_data
        out_path = output_dir / "eval_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists(), "Confusion matrix PNG was not created"

    def test_png_is_non_empty(self, output_dir):
        out_path = output_dir / "eval_confusion_matrix.png"
        assert out_path.stat().st_size > 0, "Confusion matrix PNG is empty"

    def test_works_with_subset_of_classes(self, output_dir):
        """Confusion matrix should handle data that only uses a subset of classes."""
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 5, size=50).astype(np.int64)
        y_pred = rng.integers(0, 5, size=50).astype(np.int64)
        out_path = output_dir / "_eval_cm_subset.png"
        # Should not raise
        plot_confusion_matrix(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests — plot_classification_report  (class spread)
# ---------------------------------------------------------------------------

class TestPlotClassificationReport:
    def test_creates_png_file(self, synthetic_data, output_dir):
        y_true, y_pred, _ = synthetic_data
        out_path = output_dir / "eval_class_spread.png"
        plot_classification_report(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists(), "Class spread PNG was not created"

    def test_png_is_non_empty(self, output_dir):
        out_path = output_dir / "eval_class_spread.png"
        assert out_path.stat().st_size > 0, "Class spread PNG is empty"

    def test_handles_perfect_predictions(self, output_dir):
        """Should not crash when all predictions are correct."""
        y_true = np.arange(NUM_CLASSES, dtype=np.int64)
        y_pred = y_true.copy()
        out_path = output_dir / "_eval_perfect_spread.png"
        plot_classification_report(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests — plot_class_distribution  (feature distributions)
# ---------------------------------------------------------------------------

class TestPlotClassDistribution:
    def test_creates_png_file(self, synthetic_data, output_dir):
        y_true, y_pred, _ = synthetic_data
        out_path = output_dir / "eval_feature_distributions.png"
        plot_class_distribution(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists(), "Feature distributions PNG was not created"

    def test_png_is_non_empty(self, output_dir):
        out_path = output_dir / "eval_feature_distributions.png"
        assert out_path.stat().st_size > 0, "Feature distributions PNG is empty"

    def test_ground_truth_counts_match_input(self, output_dir):
        """
        The distribution plot must faithfully count ground-truth occurrences.
        We use a controlled array and verify the file is created without error.
        """
        y_true = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 1, 2, 0, 2], dtype=np.int64)
        out_path = output_dir / "_eval_dist_controlled.png"
        plot_class_distribution(y_true, y_pred, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests — plot_roc_curves
# ---------------------------------------------------------------------------

class TestPlotRocCurves:
    def test_creates_png_file(self, synthetic_data, output_dir):
        y_true, _, y_probs = synthetic_data
        out_path = output_dir / "eval_roc_curves.png"
        plot_roc_curves(y_true, y_probs, ROBOFLOW_CLASS_NAMES, out_path)
        assert out_path.exists(), "ROC curves PNG was not created"

    def test_png_is_non_empty(self, output_dir):
        out_path = output_dir / "eval_roc_curves.png"
        assert out_path.stat().st_size > 0, "ROC curves PNG is empty"

    def test_returns_auc_scores_dict(self, synthetic_data, output_dir):
        y_true, _, y_probs = synthetic_data
        out_path = output_dir / "_eval_roc_auc_check.png"
        auc_scores = plot_roc_curves(y_true, y_probs, ROBOFLOW_CLASS_NAMES, out_path)
        assert isinstance(auc_scores, dict), "plot_roc_curves should return a dict"
        # All AUC values must be in [0, 1]
        for class_name, auc_val in auc_scores.items():
            assert 0.0 <= auc_val <= 1.0, (
                f"AUC for '{class_name}' out of range: {auc_val}"
            )

    def test_auc_covers_all_represented_classes(self, synthetic_data, output_dir):
        """Every class that has at least one positive sample should appear in AUC dict."""
        y_true, _, y_probs = synthetic_data
        out_path = output_dir / "_eval_roc_coverage.png"
        auc_scores = plot_roc_curves(y_true, y_probs, ROBOFLOW_CLASS_NAMES, out_path)
        classes_with_positives = {
            ROBOFLOW_CLASS_NAMES[i]
            for i in range(NUM_CLASSES)
            if (y_true == i).any()
        }
        for cls in classes_with_positives:
            assert cls in auc_scores, f"Class '{cls}' missing from AUC scores"


# ---------------------------------------------------------------------------
# Tests — plot_confidence_distribution
# ---------------------------------------------------------------------------

class TestPlotConfidenceDistribution:
    def test_creates_png_file(self, synthetic_data, output_dir):
        y_true, y_pred, y_probs = synthetic_data
        out_path = output_dir / "eval_confidence_distribution.png"
        plot_confidence_distribution(y_true, y_pred, y_probs, out_path)
        assert out_path.exists(), "Confidence distribution PNG was not created"

    def test_png_is_non_empty(self, output_dir):
        out_path = output_dir / "eval_confidence_distribution.png"
        assert out_path.stat().st_size > 0, "Confidence distribution PNG is empty"

    def test_handles_all_correct_predictions(self, output_dir):
        """Should not crash when there are no incorrect predictions."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_pred = y_true.copy()
        rng = np.random.default_rng(3)
        y_probs = np.eye(NUM_CLASSES, dtype=np.float32)[:3]
        out_path = output_dir / "_eval_conf_all_correct.png"
        plot_confidence_distribution(y_true, y_pred, y_probs, out_path)
        assert out_path.exists()

    def test_handles_all_incorrect_predictions(self, output_dir):
        """Should not crash when all predictions are wrong."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        y_pred = np.array([1, 2, 0], dtype=np.int64)
        y_probs = np.ones((3, NUM_CLASSES), dtype=np.float32) / NUM_CLASSES
        out_path = output_dir / "_eval_conf_all_wrong.png"
        plot_confidence_distribution(y_true, y_pred, y_probs, out_path)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests — save_evaluation_summary  (JSON)
# ---------------------------------------------------------------------------

class TestSaveEvaluationSummary:
    def test_creates_json_file(self, synthetic_data, output_dir):
        y_true, y_pred, y_probs = synthetic_data
        out_path = output_dir / "eval_summary.json"
        save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=None, evaluation_duration_s=1.23, output_path=out_path
        )
        assert out_path.exists(), "Evaluation summary JSON was not created"

    def test_json_is_valid(self, output_dir):
        out_path = output_dir / "eval_summary.json"
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_top_level_keys(self, output_dir):
        out_path = output_dir / "eval_summary.json"
        with open(out_path) as f:
            data = json.load(f)
        required_keys = {
            "overall_accuracy",
            "num_classes",
            "total_samples",
            "evaluation_duration_seconds",
            "per_class_metrics",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"JSON summary missing top-level keys: {missing}"

    def test_json_has_all_17_classes(self, output_dir):
        out_path = output_dir / "eval_summary.json"
        with open(out_path) as f:
            data = json.load(f)
        per_class = data["per_class_metrics"]
        assert len(per_class) == NUM_CLASSES, (
            f"Expected {NUM_CLASSES} classes in per_class_metrics, got {len(per_class)}"
        )

    def test_json_per_class_metric_structure(self, output_dir):
        out_path = output_dir / "eval_summary.json"
        with open(out_path) as f:
            data = json.load(f)
        per_class = data["per_class_metrics"]
        required_metric_keys = {"precision", "recall", "f1_score", "support"}
        for cls_name, metrics in per_class.items():
            missing = required_metric_keys - set(metrics.keys())
            assert not missing, (
                f"Class '{cls_name}' missing metric keys: {missing}"
            )

    def test_json_accuracy_in_valid_range(self, output_dir):
        out_path = output_dir / "eval_summary.json"
        with open(out_path) as f:
            data = json.load(f)
        acc = data["overall_accuracy"]
        assert 0.0 <= acc <= 1.0, f"overall_accuracy {acc} is outside [0, 1]"

    def test_json_total_samples_matches_input(self, synthetic_data, output_dir):
        y_true, y_pred, y_probs = synthetic_data
        out_path = output_dir / "_eval_summary_count_check.json"
        save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=None, evaluation_duration_s=0.5, output_path=out_path
        )
        with open(out_path) as f:
            data = json.load(f)
        assert data["total_samples"] == len(y_true)

    def test_json_with_auc_scores_included(self, synthetic_data, output_dir):
        y_true, y_pred, y_probs = synthetic_data
        # Generate real AUC scores first
        roc_path = output_dir / "_eval_roc_for_summary.png"
        auc_scores = plot_roc_curves(y_true, y_probs, ROBOFLOW_CLASS_NAMES, roc_path)

        out_path = output_dir / "_eval_summary_with_auc.json"
        summary = save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=auc_scores, evaluation_duration_s=2.0, output_path=out_path
        )
        with open(out_path) as f:
            data = json.load(f)
        # At least one class should have a non-null AUC
        auc_values = [
            metrics["auc"]
            for metrics in data["per_class_metrics"].values()
            if metrics["auc"] is not None
        ]
        assert len(auc_values) > 0, "No AUC values were stored in summary JSON"


# ---------------------------------------------------------------------------
# Tests — correctness of metrics on controlled data
# ---------------------------------------------------------------------------

class TestMetricsCorrectness:
    """Verify metric values match expected output on fully predictable data."""

    def test_perfect_classifier_accuracy(self, output_dir):
        """A classifier that always predicts the true class must achieve 100 % accuracy."""
        n = NUM_CLASSES * 10  # 170 samples, 10 per class
        y_true = np.repeat(np.arange(NUM_CLASSES), 10).astype(np.int64)
        y_pred = y_true.copy()

        # Build trivially perfect probability matrix
        y_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        for i, cls in enumerate(y_true):
            y_probs[i, cls] = 1.0

        out_path = output_dir / "_eval_summary_perfect.json"
        summary = save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=None, evaluation_duration_s=0.1, output_path=out_path
        )
        assert summary["overall_accuracy"] == pytest.approx(1.0), (
            "Perfect classifier should have accuracy 1.0"
        )

    def test_worst_classifier_low_accuracy(self, output_dir):
        """A classifier that never predicts the true class should have low accuracy."""
        n = NUM_CLASSES * 10
        y_true = np.repeat(np.arange(NUM_CLASSES), 10).astype(np.int64)
        # Always predict next class (mod n_classes) — never correct
        y_pred = (y_true + 1) % NUM_CLASSES

        y_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        for i, cls in enumerate(y_pred):
            y_probs[i, cls] = 1.0

        out_path = output_dir / "_eval_summary_worst.json"
        summary = save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=None, evaluation_duration_s=0.1, output_path=out_path
        )
        assert summary["overall_accuracy"] == pytest.approx(0.0), (
            "Worst classifier should have accuracy 0.0"
        )

    def test_per_class_f1_perfect(self, output_dir):
        """Each class should have F1=1.0 for a perfect classifier."""
        y_true = np.arange(NUM_CLASSES, dtype=np.int64)
        y_pred = y_true.copy()
        y_probs = np.eye(NUM_CLASSES, dtype=np.float32)

        out_path = output_dir / "_eval_summary_f1_perfect.json"
        summary = save_evaluation_summary(
            y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
            auc_scores=None, evaluation_duration_s=0.1, output_path=out_path
        )
        for cls_name, metrics in summary["per_class_metrics"].items():
            if metrics["support"] > 0:
                assert metrics["f1_score"] == pytest.approx(1.0), (
                    f"F1 for '{cls_name}' should be 1.0 with perfect predictions"
                )

    def test_num_classes_constant_matches_roboflow_list(self):
        """NUM_CLASSES must equal the length of ROBOFLOW_CLASS_NAMES."""
        assert NUM_CLASSES == len(ROBOFLOW_CLASS_NAMES) == 17


# ---------------------------------------------------------------------------
# Tests — all required output files are present after running suite
# ---------------------------------------------------------------------------

class TestRequiredOutputFiles:
    """
    These tests act as a final contract check: the canonical output filenames
    required by task 14.2 must all exist after the test suite runs.
    They depend on earlier tests having already generated the files.
    """

    REQUIRED_PNGS = [
        "eval_confusion_matrix.png",
        "eval_class_spread.png",
        "eval_feature_distributions.png",
        "eval_roc_curves.png",
        "eval_confidence_distribution.png",
    ]
    REQUIRED_JSONS = [
        "eval_summary.json",
    ]

    @pytest.mark.parametrize("filename", REQUIRED_PNGS)
    def test_required_png_exists(self, filename, output_dir):
        path = output_dir / filename
        # Generate if not yet present (handles standalone run of this class)
        if not path.exists():
            y_true, y_pred, y_probs = _make_synthetic_data()
            if "confusion_matrix" in filename:
                plot_confusion_matrix(y_true, y_pred, ROBOFLOW_CLASS_NAMES, path)
            elif "class_spread" in filename:
                plot_classification_report(y_true, y_pred, ROBOFLOW_CLASS_NAMES, path)
            elif "feature_distributions" in filename:
                plot_class_distribution(y_true, y_pred, ROBOFLOW_CLASS_NAMES, path)
            elif "roc_curves" in filename:
                plot_roc_curves(y_true, y_probs, ROBOFLOW_CLASS_NAMES, path)
            elif "confidence_distribution" in filename:
                plot_confidence_distribution(y_true, y_pred, y_probs, path)
        assert path.exists(), f"Required output file not found: {filename}"
        assert path.stat().st_size > 0, f"Required output file is empty: {filename}"

    @pytest.mark.parametrize("filename", REQUIRED_JSONS)
    def test_required_json_exists(self, filename, output_dir):
        path = output_dir / filename
        if not path.exists():
            y_true, y_pred, y_probs = _make_synthetic_data()
            save_evaluation_summary(
                y_true, y_pred, y_probs, ROBOFLOW_CLASS_NAMES,
                auc_scores=None, evaluation_duration_s=1.0, output_path=path
            )
        assert path.exists(), f"Required output file not found: {filename}"
        assert path.stat().st_size > 0, f"Required output file is empty: {filename}"

        with open(path) as f:
            data = json.load(f)
        assert "per_class_metrics" in data
        assert "overall_accuracy" in data
