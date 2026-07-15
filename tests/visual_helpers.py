"""
Reusable plotting utilities for test visual diagnostics.

Follows the project's evaluate_model.py pattern using matplotlib/seaborn/numpy
for generating visual diagnostic artifacts during testing. All outputs are saved
to the tests/output/ directory as PNGs or JSON files.

Functions:
- plot_confusion_matrix: Confusion matrix heatmap
- plot_class_distribution: Bar chart of class counts
- plot_confidence_histogram: Histogram of confidence scores
- plot_iou_distribution: Histogram of IoU values
- plot_annotated_frame: Frame image with bounding boxes and zone overlays
- save_json_report: Save structured data as JSON
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use non-interactive backend for headless test environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# Style defaults (matching evaluate_model.py aesthetic)
# ---------------------------------------------------------------------------

_DPI = 300
_COLORS = {
    "primary": "#3498db",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "info": "#9b59b6",
    "dark": "#2c3e50",
}

_ZONE_COLORS = {
    "no_people": (1.0, 0.2, 0.2, 0.25),      # Red semi-transparent
    "operation": (0.2, 0.8, 0.2, 0.25),       # Green semi-transparent
    "dropoff": (0.2, 0.4, 1.0, 0.25),         # Blue semi-transparent
}

_CLASS_COLORS = {
    "Human": "#e74c3c",
    "Human - No Safety Clothes": "#c0392b",
    "Container - Misaligned": "#f39c12",
    "Container - Open": "#e67e22",
    "Container - Picked": "#3498db",
    "Container - Stacked": "#2ecc71",
    "Container - Reefer": "#1abc9c",
    "Container - Water Drop": "#16a085",
    "Container - Separate": "#27ae60",
    "Crane": "#9b59b6",
    "Truck - No Container": "#8e44ad",
    "Truck - With Container": "#2980b9",
    "Vehicle": "#34495e",
    "Boat - With Cargo": "#7f8c8d",
    "Yard - No People": "#e74c3c",
    "Yard - Operation Zone": "#2ecc71",
    "Yard - Dropoff zone": "#3498db",
}


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Args:
        confusion_matrix: 2D numpy array of shape (n_classes, n_classes)
        class_names: List of class name strings
        output_path: Path to save the PNG file
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Path,
    title: str = "Class Distribution",
) -> None:
    """
    Generate and save a bar chart of class counts.

    Args:
        class_counts: Dictionary mapping class names to their counts
        output_path: Path to save the PNG file
        title: Plot title
    """
    names = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        range(len(names)),
        counts,
        color=_COLORS["primary"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close()


def plot_confidence_histogram(
    confidences: List[float],
    output_path: Path,
    title: str = "Confidence Score Distribution",
    bins: int = 30,
    threshold: Optional[float] = None,
) -> None:
    """
    Generate and save a histogram of confidence scores.

    Args:
        confidences: List of confidence values in [0.0, 1.0]
        output_path: Path to save the PNG file
        title: Plot title
        bins: Number of histogram bins
        threshold: Optional threshold line to draw on the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        confidences,
        bins=bins,
        alpha=0.7,
        color=_COLORS["primary"],
        edgecolor="black",
        linewidth=0.5,
    )

    if threshold is not None:
        ax.axvline(
            threshold,
            color=_COLORS["danger"],
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {threshold:.2f}",
        )
        ax.legend(fontsize=11)

    # Add mean line
    mean_conf = np.mean(confidences) if confidences else 0
    ax.axvline(
        mean_conf,
        color=_COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_conf:.3f}",
    )
    ax.legend(fontsize=11)

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close()


def plot_iou_distribution(
    ious: List[float],
    output_path: Path,
    title: str = "IoU Distribution",
    bins: int = 25,
    threshold: Optional[float] = 0.5,
) -> None:
    """
    Generate and save a histogram of IoU (Intersection over Union) values.

    Args:
        ious: List of IoU values in [0.0, 1.0]
        output_path: Path to save the PNG file
        title: Plot title
        bins: Number of histogram bins
        threshold: Optional IoU threshold line to draw
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        ious,
        bins=bins,
        alpha=0.7,
        color=_COLORS["info"],
        edgecolor="black",
        linewidth=0.5,
    )

    if threshold is not None:
        ax.axvline(
            threshold,
            color=_COLORS["danger"],
            linestyle="--",
            linewidth=2,
            label=f"IoU Threshold: {threshold:.2f}",
        )

    mean_iou = np.mean(ious) if ious else 0
    ax.axvline(
        mean_iou,
        color=_COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Mean IoU: {mean_iou:.3f}",
    )
    ax.legend(fontsize=11)

    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close()


def plot_annotated_frame(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    zones: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    title: str = "Annotated Frame",
) -> None:
    """
    Generate and save an annotated frame with bounding boxes and zone overlays.

    Args:
        frame: Numpy array of shape (H, W, 3) in RGB or BGR format
        detections: List of dicts with keys: 'bbox' (x_center, y_center, w, h normalized),
                    'class_label', 'confidence'
        zones: Optional list of dicts with keys: 'vertices' (list of (x, y) normalized),
               'zone_type'
        output_path: Path to save the PNG file. If None, displays the plot.
        title: Plot title
    """
    h, w = frame.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(frame)

    # Draw zone overlays first (behind detections)
    if zones:
        for zone in zones:
            vertices = zone.get("vertices", [])
            zone_type = zone.get("zone_type", "no_people")
            color = _ZONE_COLORS.get(zone_type, (0.5, 0.5, 0.5, 0.2))

            # Convert normalized vertices to pixel coordinates
            polygon_points = [(x * w, y * h) for x, y in vertices]
            if polygon_points:
                polygon = plt.Polygon(
                    polygon_points,
                    closed=True,
                    facecolor=color,
                    edgecolor=color[:3] + (0.8,),
                    linewidth=2,
                )
                ax.add_patch(polygon)

    # Draw detection bounding boxes
    for det in detections:
        bbox = det.get("bbox", {})
        x_center = bbox.get("x_center", 0.5) if isinstance(bbox, dict) else bbox.x_center
        y_center = bbox.get("y_center", 0.5) if isinstance(bbox, dict) else bbox.y_center
        bw = bbox.get("width", 0.1) if isinstance(bbox, dict) else bbox.width
        bh = bbox.get("height", 0.1) if isinstance(bbox, dict) else bbox.height

        class_label = det.get("class_label", "Unknown")
        confidence = det.get("confidence", 0.0)

        # Convert normalized coords to pixel coords
        x1 = (x_center - bw / 2) * w
        y1 = (y_center - bh / 2) * h
        box_w = bw * w
        box_h = bh * h

        # Pick color based on class
        color = _CLASS_COLORS.get(class_label, "#95a5a6")

        rect = mpatches.FancyBboxPatch(
            (x1, y1),
            box_w,
            box_h,
            boxstyle="round,pad=0",
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label
        label_text = f"{class_label} ({confidence:.2f})"
        ax.text(
            x1,
            y1 - 5,
            label_text,
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close()


def save_json_report(
    data: Any,
    output_path: Path,
    indent: int = 2,
) -> None:
    """
    Save structured data as a JSON report file.

    Args:
        data: Data to serialize (must be JSON-serializable)
        output_path: Path to save the JSON file
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle numpy types for serialization
    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=_default)
