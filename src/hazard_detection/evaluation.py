"""
Model Evaluation Module for the Hazard Detection System.

Evaluates a trained YOLOv12 model on the 17-class Roboflow dataset and
generates comprehensive visualizations:
  - Confusion matrix for all 17 detection classes
  - Precision/Recall/F1 per-class bar charts
  - Class distribution comparison (ground truth vs predictions)
  - ROC curves for multi-class detection
  - Confidence score distribution (correct vs incorrect predictions)

All plots are saved as PNG to a configurable output directory.
An evaluation summary JSON with per-class metrics is also saved.

Requirements covered:
  - 13.1: YOLO_Detector trained on Roboflow 17-class dataset
  - 13.2: Detections include bbox, class label, confidence
  - 13.6: 17-class taxonomy from Roboflow data.yaml
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .diagnostics import PerformanceTimer, get_logger

# ---------------------------------------------------------------------------
# Class definitions matching the Roboflow data.yaml taxonomy
# ---------------------------------------------------------------------------
ROBOFLOW_CLASS_NAMES: List[str] = [
    "Boat - With Cargo",
    "Container - Misaligned",
    "Container - Open",
    "Container - Picked",
    "Container - Reefer",
    "Container - Water Drop",
    "Container - Separate",
    "Container - Stacked",
    "Crane",
    "Human",
    "Human - No Safety Clothes",
    "Truck - No Container",
    "Truck - With Container",
    "Vehicle",
    "Yard - Dropoff zone",
    "Yard - No People",
    "Yard - Operation Zone",
]
NUM_CLASSES: int = len(ROBOFLOW_CLASS_NAMES)  # 17


# ---------------------------------------------------------------------------
# Individual plot functions (mirrors evaluate_model.py pattern)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap for all present classes."""
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    labels_present = [int(c) for c in unique_classes]
    names_present = [class_names[i] for i in labels_present]

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)

    fig_width = max(12, len(names_present) * 0.8)
    fig_height = max(10, len(names_present) * 0.7)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names_present,
        yticklabels=names_present,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix — Hazard Detection (17 Classes)", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    if len(labels_present) < NUM_CLASSES:
        _logger = get_logger("evaluation")
        _logger.info(
            f"Confusion matrix: only {len(labels_present)}/{NUM_CLASSES} classes present in data"
        )


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate and save precision/recall/F1 bar charts for all present classes."""
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    labels_present = [int(c) for c in unique_classes]
    names_present = [class_names[i] for i in labels_present]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0
    )

    x = np.arange(len(names_present))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(names_present) * 0.9), 7))
    ax.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax.bar(x, recall, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1-Score", color="#e74c3c")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 by Class", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names_present, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.15])
    ax.grid(axis="y", alpha=0.3)

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
        ax.text(i + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate class distribution comparison (ground truth vs predictions)."""
    true_counts = np.bincount(y_true, minlength=NUM_CLASSES)
    pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(16, NUM_CLASSES * 0.9), 7))
    bars1 = ax.bar(x - width / 2, true_counts, width, label="Ground Truth", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, pred_counts, width, label="Predictions", color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Class Distribution: Ground Truth vs Predictions", fontsize=16, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Generate ROC curves for multi-class detection (one-vs-rest)."""
    n_classes = y_probs.shape[1]
    classes_present = list(range(n_classes))

    # Binarise labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=classes_present)
    if y_true_bin.shape[1] != n_classes:
        # Binary edge case — expand manually
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate distinct colours for all 17 classes
    cmap = plt.cm.get_cmap("tab20", n_classes)
    colors = [cmap(i) for i in range(n_classes)]

    auc_scores: Dict[str, float] = {}
    for i, (class_name, color) in enumerate(zip(class_names[:n_classes], colors)):
        # Skip classes with no positive samples in y_true
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{class_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Hazard Detection (17 Classes)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return auc_scores


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confidence score distribution for correct vs incorrect predictions."""
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_conf = np.max(y_probs[correct_mask], axis=1) if correct_mask.any() else np.array([])
    incorrect_conf = np.max(y_probs[incorrect_mask], axis=1) if incorrect_mask.any() else np.array([])

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(correct_conf) > 0:
        ax.hist(correct_conf, bins=30, alpha=0.7, label="Correct Predictions",
                color="#2ecc71", edgecolor="black")
        ax.axvline(
            float(np.mean(correct_conf)), color="#2ecc71", linestyle="--", linewidth=2,
            label=f"Mean Correct: {np.mean(correct_conf):.3f}",
        )
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=30, alpha=0.7, label="Incorrect Predictions",
                color="#e74c3c", edgecolor="black")
        ax.axvline(
            float(np.mean(incorrect_conf)), color="#e74c3c", linestyle="--", linewidth=2,
            label=f"Mean Incorrect: {np.mean(incorrect_conf):.3f}",
        )

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Confidence Distribution: Correct vs Incorrect Predictions",
        fontsize=16, fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    auc_scores: Optional[Dict[str, float]],
    evaluation_duration_s: float,
    output_path: Path,
) -> Dict:
    """Save evaluation summary as JSON with per-class metrics."""
    from sklearn.metrics import accuracy_score

    accuracy = float(accuracy_score(y_true, y_pred))
    labels_all = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_all, zero_division=0
    )

    per_class: Dict[str, Dict] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1_score": round(float(f1[i]), 6),
            "support": int(support[i]),
            "auc": round(float(auc_scores.get(name, 0.0)), 6) if auc_scores else None,
        }

    summary = {
        "overall_accuracy": round(accuracy, 6),
        "num_classes": len(class_names),
        "total_samples": int(len(y_true)),
        "evaluation_duration_seconds": round(evaluation_duration_s, 3),
        "per_class_metrics": per_class,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# ModelEvaluator class
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    Evaluates a trained YOLOv12 model on the Roboflow 17-class hazard detection dataset.

    Follows the evaluate_model.py pattern: loads model, runs inference on the
    validation split, collects predictions, and generates a suite of visualizations
    plus a JSON summary saved to a configurable output directory.

    Args:
        checkpoint_path: Path to the trained YOLO model weights (.pt file).
        data_yaml: Path to the Roboflow data.yaml defining the dataset splits.
        device: Inference device — "cuda", "cpu", or "auto" (auto-selects GPU if available).
        output_dir: Directory where all PNGs and the JSON summary are saved.
        class_names: Override the default 17-class Roboflow list. Mainly for testing.
        conf_threshold: Confidence threshold below which detections are ignored (0.0–1.0).
        iou_threshold: IoU threshold for NMS during YOLO inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_yaml: str,
        device: str = "auto",
        output_dir: str = "evaluation_results",
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir)
        self.class_names: List[str] = class_names or ROBOFLOW_CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._logger = get_logger("evaluation")

        # Resolve device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"ModelEvaluator: checkpoint not found at '{self.checkpoint_path}'"
            )
        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"ModelEvaluator: data_yaml not found at '{self.data_yaml}'"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            f"ModelEvaluator initialised | checkpoint={self.checkpoint_path} "
            f"| data_yaml={self.data_yaml} | device={self.device} "
            f"| output_dir={self.output_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict:
        """
        Run full evaluation pipeline:
          1. Load the YOLO model
          2. Run inference on the validation split defined in data_yaml
          3. Collect ground-truth labels and model predictions/probabilities
          4. Generate all visualisation plots
          5. Save and return the evaluation summary dict

        Returns:
            Evaluation summary dictionary (same structure as evaluation_summary.json).
        """
        eval_start = time.perf_counter()
        self._logger.info("Starting model evaluation")

        with PerformanceTimer("model_load", logger=self._logger):
            model = self._load_model()

        with PerformanceTimer("inference", logger=self._logger):
            y_true, y_pred, y_probs = self._run_inference(model)

        eval_duration = time.perf_counter() - eval_start
        self._logger.info(
            f"Inference complete: {len(y_true)} samples in {eval_duration:.2f}s"
        )

        with PerformanceTimer("plot_generation", logger=self._logger):
            auc_scores = self._generate_plots(y_true, y_pred, y_probs)

        summary_path = self.output_dir / "evaluation_summary.json"
        summary = save_evaluation_summary(
            y_true, y_pred, y_probs,
            self.class_names, auc_scores, eval_duration,
            summary_path,
        )

        self._logger.info(
            f"Evaluation complete | accuracy={summary['overall_accuracy']:.4f} "
            f"| duration={eval_duration:.2f}s | results saved to {self.output_dir}"
        )
        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load YOLO model from checkpoint using Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics package is required for ModelEvaluator. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._logger.info(f"Loading model from {self.checkpoint_path}")
        model = YOLO(str(self.checkpoint_path))
        return model

    def _run_inference(self, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run YOLO inference on the validation split and collect per-image
        ground-truth labels, predicted labels, and class probability vectors.

        For each image we use the highest-confidence detection as the predicted
        class, and build a soft-score vector from the per-class max confidence
        observed across all detections in that image.

        Returns:
            y_true:  (N,) integer array of ground-truth class indices
            y_pred:  (N,) integer array of predicted class indices
            y_probs: (N, C) float array of per-class max confidence scores
        """
        import yaml

        # Parse data.yaml to locate validation images and labels
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)

        data_root = self.data_yaml.parent
        val_images_rel = data_cfg.get("val", "../valid/images")

        # data.yaml val paths are written relative to the split subdirectories
        # (e.g. "../valid/images" means "from train/ go up to roboflow data/
        # then into valid/images"). We resolve from data_root directly by
        # normalizing the path properly.
        # First try: resolve val path as-is from data_root
        # Second try: the path may use ../ notation meant for split-level,
        #             so strip leading ../ and resolve from data_root
        import re
        val_clean = re.sub(r'^(\.\./)+', '', val_images_rel)  # strip leading ../

        candidates = [
            (data_root / val_clean).resolve(),               # most common: valid/images from data.yaml folder
            (data_root / val_images_rel.lstrip("../")).resolve(),
            Path(val_images_rel).resolve(),
        ]
        val_images_path = None
        for candidate in candidates:
            if candidate.exists():
                val_images_path = candidate
                break

        if val_images_path is None:
            # Last resort: search for valid/images anywhere under data_root
            fallback = data_root / "valid" / "images"
            if fallback.exists():
                val_images_path = fallback.resolve()

        if val_images_path is None:
            raise FileNotFoundError(
                f"Validation images directory not found. Tried:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + f"\n\nPass the absolute path to data.yaml, e.g.:\n"
                + f'  python scripts/evaluate_yolo.py --data "{data_root.resolve()}\\data.yaml"'
            )

        val_labels_path = Path(str(val_images_path).replace("images", "labels"))

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            p for p in val_images_path.iterdir() if p.suffix.lower() in image_exts
        )

        if not image_files:
            raise ValueError(f"No images found in validation directory: {val_images_path}")

        self._logger.info(
            f"Running inference on {len(image_files)} validation images "
            f"from {val_images_path}"
        )

        n_classes = len(self.class_names)
        y_true_list: List[int] = []
        y_pred_list: List[int] = []
        y_probs_list: List[np.ndarray] = []

        batch_start = time.perf_counter()

        for idx, img_path in enumerate(image_files):
            # --- Ground truth ---
            label_file = val_labels_path / (img_path.stem + ".txt")
            gt_class = self._read_gt_class(label_file)
            if gt_class is None:
                continue  # Skip images with no annotations

            # --- Inference ---
            batch_t = time.perf_counter()
            results = model.predict(
                source=str(img_path),
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            batch_ms = (time.perf_counter() - batch_t) * 1000

            pred_class, prob_vector = self._extract_prediction(results, n_classes)

            y_true_list.append(gt_class)
            y_pred_list.append(pred_class)
            y_probs_list.append(prob_vector)

            if (idx + 1) % 10 == 0:
                elapsed = time.perf_counter() - batch_start
                self._logger.info(
                    f"  [{idx+1}/{len(image_files)}] last batch: {batch_ms:.1f}ms "
                    f"| elapsed: {elapsed:.1f}s"
                )

        if not y_true_list:
            raise ValueError("No valid ground-truth annotations found in validation set.")

        return (
            np.array(y_true_list, dtype=np.int64),
            np.array(y_pred_list, dtype=np.int64),
            np.array(y_probs_list, dtype=np.float32),
        )

    def _read_gt_class(self, label_file: Path) -> Optional[int]:
        """
        Read the dominant ground-truth class index from a YOLO label file.

        Returns the class that appears most frequently in the label file,
        or None if the file doesn't exist or is empty.
        """
        if not label_file.exists():
            return None
        try:
            classes: List[int] = []
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.append(int(parts[0]))
            if not classes:
                return None
            # Use most-frequent class as the image-level label
            return max(set(classes), key=classes.count)
        except (ValueError, OSError):
            return None

    def _extract_prediction(
        self, results, n_classes: int
    ) -> Tuple[int, np.ndarray]:
        """
        Extract predicted class index and probability vector from Ultralytics results.

        Builds a probability vector by recording the max confidence seen for each
        class across all detections in the image, then normalises so it sums to 1.

        Returns:
            (predicted_class_index, probability_vector of shape (n_classes,))
        """
        prob_vector = np.zeros(n_classes, dtype=np.float32)
        predicted_class = 0  # default to class 0 when no detection

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)

                # Build per-class max confidence
                for cls_id, conf in zip(cls_ids, confs):
                    if 0 <= cls_id < n_classes:
                        prob_vector[cls_id] = max(prob_vector[cls_id], float(conf))

                # Predicted class = highest-confidence detection
                best_idx = int(np.argmax(confs))
                predicted_class = int(cls_ids[best_idx]) if 0 <= int(cls_ids[best_idx]) < n_classes else 0

        # Normalise to pseudo-probability (sum to 1)
        total = prob_vector.sum()
        if total > 0:
            prob_vector /= total
        else:
            # No detections: uniform distribution
            prob_vector[:] = 1.0 / n_classes

        return predicted_class, prob_vector

    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """Generate all evaluation plots and return AUC scores dict."""
        self._logger.info("Generating evaluation plots")

        plot_confusion_matrix(
            y_true, y_pred, self.class_names,
            self.output_dir / "confusion_matrix.png",
        )
        self._logger.info("✓ confusion_matrix.png saved")

        plot_classification_report(
            y_true, y_pred, self.class_names,
            self.output_dir / "classification_metrics.png",
        )
        self._logger.info("✓ classification_metrics.png saved")

        plot_class_distribution(
            y_true, y_pred, self.class_names,
            self.output_dir / "class_distribution.png",
        )
        self._logger.info("✓ class_distribution.png saved")

        auc_scores: Optional[Dict[str, float]] = None
        try:
            auc_scores = plot_roc_curves(
                y_true, y_probs, self.class_names,
                self.output_dir / "roc_curves.png",
            )
            self._logger.info("✓ roc_curves.png saved")
        except Exception as exc:  # pylint: disable=broad-except
            self._logger.warning(f"ROC curve generation skipped: {exc}")

        plot_confidence_distribution(
            y_true, y_pred, y_probs,
            self.output_dir / "confidence_distribution.png",
        )
        self._logger.info("✓ confidence_distribution.png saved")

        return auc_scores

    @staticmethod
    def _print_summary(summary: Dict) -> None:
        """Print a human-readable evaluation summary to stdout."""
        print("\n" + "=" * 70)
        print("HAZARD DETECTION MODEL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Overall accuracy : {summary['overall_accuracy']:.4f}")
        print(f"  Total samples    : {summary['total_samples']}")
        print(f"  Evaluation time  : {summary['evaluation_duration_seconds']:.2f}s")
        print("\n  Per-class F1 scores:")
        for cls_name, metrics in summary["per_class_metrics"].items():
            support = metrics["support"]
            if support > 0:
                print(f"    {cls_name:<40} F1={metrics['f1_score']:.4f}  (n={support})")
        print("=" * 70)
        print(f"\nResults saved to: {summary.get('output_dir', 'evaluation_results/')}")


# ---------------------------------------------------------------------------
# CLI entry point (mirrors evaluate_model.py usage)
# ---------------------------------------------------------------------------


def main() -> int:
    """
    CLI entry point for standalone evaluation.

    Usage:
        python -m hazard_detection.evaluation \\
            --checkpoint checkpoints/yolov12_best.pt \\
            --data_yaml "roboflow data/data.yaml" \\
            --output_dir evaluation_results \\
            [--device auto] [--conf 0.25] [--iou 0.45]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv12 hazard detection model on the Roboflow dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained YOLO model weights (.pt)"
    )
    parser.add_argument(
        "--data_yaml", type=str, default="roboflow data/data.yaml",
        help="Path to Roboflow data.yaml"
    )
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device: cuda / cpu / auto"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections (0.0–1.0)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (0.0–1.0)"
    )

    args = parser.parse_args()

    from .diagnostics import setup_logging
    setup_logging()

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_yaml=args.data_yaml,
        device=args.device,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    evaluator.evaluate()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
