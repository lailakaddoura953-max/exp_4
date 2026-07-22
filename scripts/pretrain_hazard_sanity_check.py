"""
Pre-Training Hazard Sanity Check for the CNN Fallback Pipeline
================================================================
Before relying on the CNN fallback (scripts/annotation/cnn_auto_annotate.py)
to annotate the real "normal operations" dataset at scale, this script
answers one question: can a traditional-CNN (YOLOv12) detector actually
learn to find the hazard classes we care about, when they're present?

The "normal operations" dataset (image_data_normal) is, by definition,
free of real hazards — that's what makes it useful as background footage,
but it also means there's nothing to validate detection against. So this
script:

  1. Injects synthetic instances of the hazard classes into copies of the
     normal images, using the same asset-extraction/compositing logic as
     scripts/generate_hazard_augmentations.py (imported directly — no
     duplicated logic). This gives us images with KNOWN ground-truth
     hazard locations.

  2. Splits the result into a small train/val YOLO dataset.

  3. Trains (or fine-tunes, if --base_checkpoint is given) a YOLOv12 model
     for a small number of epochs — enough to tell us whether the signal
     is learnable, not a production-quality model.

  4. Evaluates the trained model against the KNOWN injected ground truth
     and reports per-class recall for the hazard classes.

  5. Prints a pass/fail verdict: if recall clears --recall_threshold for
     every hazard class, the CNN pipeline is validated as viable and you
     can move on to running cnn_auto_annotate.py / run_auto_annotate.py
     at full scale. If it doesn't, that's a signal to add more synthetic
     examples, train longer, or fall back to the segmentation pipeline
     (auto_annotate.py) instead.

Hazard classes checked (matches SAFETY_CRITICAL_CLASSES in
scripts/annotation/auto_annotate.py and HAZARD_CLASSES in
scripts/generate_hazard_augmentations.py):
    2  — Container - Open
    9  — Human
    10 — Human - No Safety Clothes

Usage:
    # Full run: generate synthetic data, train a quick model, evaluate it
    python scripts/pretrain_hazard_sanity_check.py

    # Fine-tune an existing checkpoint instead of training from scratch
    python scripts/pretrain_hazard_sanity_check.py \\
        --base_checkpoint runs/train/hazard_yolo/weights/best.pt

    # Reuse a previously generated synthetic dataset (skip regeneration)
    python scripts/pretrain_hazard_sanity_check.py --skip_generation

    # Reuse a previously trained sanity-check model (skip training)
    python scripts/pretrain_hazard_sanity_check.py --skip_training \\
        --checkpoint runs/train/hazard_sanity_check/weights/best.pt
"""

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Sibling script (scripts/generate_hazard_augmentations.py) — reused directly
# rather than duplicated, so both stay in sync on hazard-class definitions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_hazard_augmentations as hazard_gen  # noqa: E402

# hazard_detection package (training pipeline)
sys.path.insert(0, "src")


# ─────────────────────────────────────────────────────────────────────────────
# Input dataset resolution
# ─────────────────────────────────────────────────────────────────────────────
# image_data_normal (the real "normal operations" dataset) lives on a
# separate, access-restricted device and may not be present on this machine.
# Fall back to "roboflow data" as background imagery so the sanity check can
# still run end-to-end for setup/testing purposes. Note this is a stand-in
# only — Roboflow images already contain real hazard instances, so recall
# numbers from a roboflow-data-as-background run are not representative of
# how the pipeline will perform on the real hazard-free normal-operations
# footage. Treat a PASS in that mode as "the mechanics work", not "the
# fallback pipeline is production-viable".
PREFERRED_NORMAL_DIR = "image_data_normal"
FALLBACK_NORMAL_DIR = "roboflow data"


def resolve_default_normal_dir() -> Path:
    """Return image_data_normal if present, otherwise roboflow data."""
    return Path(PREFERRED_NORMAL_DIR) if Path(PREFERRED_NORMAL_DIR).exists() else Path(FALLBACK_NORMAL_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Synthetic hazard injection (delegates to generate_hazard_augmentations)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_hazards(
    roboflow_dir: Path,
    normal_dir: Path,
    synthetic_dir: Path,
    injections_per_image: int,
    max_images: Optional[int],
    seed: int,
) -> None:
    print("\n[Step 1/4] Injecting synthetic hazards into normal-operations images...")
    hazard_gen.run(
        roboflow_dir=roboflow_dir,
        normal_dir=normal_dir,
        output_dir=synthetic_dir,
        injections_per_image=injections_per_image,
        max_images=max_images,
        seed=seed,
        dry_run=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build a train/val YOLO split + data.yaml from the synthetic set
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(synthetic_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (image, label) pairs produced by generate_hazard_augmentations.py.

    Deduplicated by resolved path: on case-insensitive filesystems (Windows,
    default macOS) a pattern like "*.png" already matches "IMG.PNG", so
    iterating both "*.png" and "*.PNG" would otherwise discover the same
    file twice. Undetected, that duplication lets an identical image land
    in both the train and val splits after shuffling, silently leaking
    validation data into training and inflating the sanity check's recall.
    """
    seen: set[Path] = set()
    pairs = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"):
        for img_path in synthetic_dir.rglob(ext):
            resolved = img_path.resolve()
            if resolved in seen:
                continue
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                seen.add(resolved)
                pairs.append((img_path, label_path))
    return sorted(pairs)


def build_yolo_split(
    synthetic_dir: Path,
    split_dir: Path,
    val_fraction: float,
    seed: int,
    class_names: list[str],
) -> tuple[Path, list[tuple[Path, Path]]]:
    """
    Copy synthetic image/label pairs into split_dir/train and split_dir/val,
    write a YOLO data.yaml pointing at them, and return (data_yaml_path,
    val_pairs) so the caller can evaluate against known val ground truth.
    """
    print("\n[Step 2/4] Building train/val YOLO split from synthetic hazard data...")

    pairs = discover_pairs(synthetic_dir)
    if not pairs:
        print(f"ERROR: No image/label pairs found under {synthetic_dir}. "
              f"Did synthetic generation run successfully?")
        sys.exit(1)

    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    if not train_pairs:
        # Extremely small datasets: guarantee at least one training example
        train_pairs = [val_pairs.pop()]

    if split_dir.exists():
        shutil.rmtree(split_dir)

    def _copy_pairs(pairs_subset: list[tuple[Path, Path]], subset_name: str) -> list[tuple[Path, Path]]:
        """
        Copy image/label pairs into split_dir/<subset_name>, renaming each
        file to a short, deterministic "{subset_name}_{index:05d}" stem
        rather than preserving the original filename.

        This matters on Windows: some Roboflow-derived source filenames
        (e.g. "stock-photo-forklift-handling-container-box-loading-to-
        truck-in-shipping-yard-...aug01_hazard_0093.png") are well over 100
        characters on their own. Combined with the split_dir path, the full
        destination path can exceed the 260-character MAX_PATH limit, which
        shutil.copy2 surfaces as a misleading "FileNotFoundError: [WinError
        3] The system cannot find the path specified" rather than a clear
        path-length error. Short deterministic names sidestep this
        entirely and also guarantee no filename collisions within a subset.

        Returns the list of (copied_image_path, copied_label_path) actually
        written, in the same order as pairs_subset, so callers can build an
        accurate mapping back to split_dir locations.
        """
        img_dir = split_dir / subset_name / "images"
        lbl_dir = split_dir / subset_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        copied: list[tuple[Path, Path]] = []
        for i, (img_path, label_path) in enumerate(pairs_subset):
            dest_img = img_dir / f"{subset_name}_{i:05d}{img_path.suffix}"
            dest_lbl = lbl_dir / f"{subset_name}_{i:05d}.txt"
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)
            copied.append((dest_img, dest_lbl))
        return copied

    _copy_pairs(train_pairs, "train")
    val_pairs_in_split = _copy_pairs(val_pairs, "val")

    data_yaml_path = split_dir / "data.yaml"
    yaml_text = (
        f"train: train/images\n"
        f"val: val/images\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names)}\n"
    )
    data_yaml_path.write_text(yaml_text, encoding="utf-8")

    print(f"  Train images : {len(train_pairs)}")
    print(f"  Val images   : {len(val_pairs)}")
    print(f"  data.yaml    : {data_yaml_path}")

    # val_pairs_in_split already references the copied (split_dir) locations
    # under their short deterministic names, since that's what _copy_pairs
    # returned above -- that's what evaluation will read images/labels from.
    return data_yaml_path, val_pairs_in_split


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Quick training / fine-tuning run
# ─────────────────────────────────────────────────────────────────────────────

def train_sanity_check_model(
    data_yaml: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    imgsz: int,
    device: str,
    base_checkpoint: Optional[str],
    project_name: str,
) -> Path:
    from hazard_detection.data_pipeline.training_pipeline import (
        TrainingConfig,
        YOLOTrainingPipeline,
    )

    print(f"\n[Step 3/4] Training sanity-check model ({epochs} epochs)...")

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_resolution=imgsz,
        checkpoint_interval=max(1, epochs // 2),
        data_yaml=str(data_yaml),
        output_dir="runs/train",
        project_name=project_name,
        device=device,
    )
    pipeline = YOLOTrainingPipeline(config)

    if base_checkpoint:
        pipeline.fine_tune(
            pretrained_checkpoint=base_checkpoint,
            additional_data_yaml=str(data_yaml),
        )
    else:
        pipeline.train(data_yaml=str(data_yaml))

    best = pipeline.best_checkpoint_path or pipeline.last_checkpoint_path
    if best is None or not best.exists():
        print("ERROR: Training completed but no checkpoint was produced.")
        sys.exit(1)

    print(f"  Sanity-check checkpoint: {best}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Evaluate recall on the known synthetic hazards
# ─────────────────────────────────────────────────────────────────────────────

def parse_label_line(line: str) -> Optional[tuple[int, list[tuple[float, float]]]]:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            return None
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        return class_id, points
    except ValueError:
        return None


def polygon_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Convert a normalised polygon to its normalised axis-aligned bbox (x1,y1,x2,y2)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ClassRecall:
    class_id: int
    class_name: str
    ground_truth_count: int = 0
    detected_count: int = 0

    @property
    def recall(self) -> float:
        if self.ground_truth_count == 0:
            return 1.0  # nothing to find, vacuously satisfied
        return self.detected_count / self.ground_truth_count


def evaluate_hazard_recall(
    checkpoint_path: Path,
    val_pairs: list[tuple[Path, Path]],
    hazard_classes: dict[int, str],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    imgsz: int,
) -> dict[int, ClassRecall]:
    from ultralytics import YOLO

    print(f"\n[Step 4/4] Evaluating detection recall on {len(val_pairs)} val images "
          f"with KNOWN synthetic hazard ground truth...")

    model = YOLO(str(checkpoint_path))
    results_by_class: dict[int, ClassRecall] = {
        cid: ClassRecall(class_id=cid, class_name=name)
        for cid, name in hazard_classes.items()
    }

    for img_path, label_path in val_pairs:
        gt_annotations = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed and parsed[0] in hazard_classes:
                gt_annotations.append(parsed)

        if not gt_annotations:
            continue

        results = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf_threshold,
            device=device, verbose=False,
        )

        pred_boxes_by_class: dict[int, list[tuple[float, float, float, float]]] = {}
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for box, cid in zip(xyxyn, cls_ids):
                pred_boxes_by_class.setdefault(int(cid), []).append(tuple(box.tolist()))

        for class_id, points in gt_annotations:
            gt_bbox = polygon_to_bbox(points)
            results_by_class[class_id].ground_truth_count += 1

            candidates = pred_boxes_by_class.get(class_id, [])
            best_iou = max((iou_xyxy(gt_bbox, pred) for pred in candidates), default=0.0)
            if best_iou >= iou_threshold:
                results_by_class[class_id].detected_count += 1

    return results_by_class


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b — Multi-hazard vs single-hazard recall breakdown
# ─────────────────────────────────────────────────────────────────────────────
# The CNN fallback produces bounding boxes only, with no pixel mask. When
# multiple hazard types co-occur in the same frame (already possible today
# via --injections_per_image >= 2 in generate_hazard_augmentations.py, which
# composites more than one hazard object per background image), the
# segmentation pipeline can disambiguate overlapping instances at the pixel
# level; the fallback cannot -- it only has box regression to separate them.
# This breakdown answers a narrower, more useful question than "does the
# fallback have this limitation" (yes, structurally, always): does recall on
# THIS checkpoint actually degrade on multi-hazard frames versus
# single-hazard ones, given the data it was trained on. It does not change
# the per-class recall metric or PASS/FAIL verdict computed by
# evaluate_hazard_recall() above -- that gate is unchanged. This is purely
# an additional diagnostic split of the same known ground truth.

@dataclass
class HazardGroupRecall:
    group_name: str  # "single_hazard" or "multi_hazard"
    image_count: int = 0
    ground_truth_count: int = 0
    detected_count: int = 0

    @property
    def recall(self) -> float:
        if self.ground_truth_count == 0:
            return 1.0  # nothing to find, vacuously satisfied
        return self.detected_count / self.ground_truth_count


def evaluate_multi_hazard_breakdown(
    checkpoint_path: Path,
    val_pairs: list[tuple[Path, Path]],
    hazard_classes: dict[int, str],
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    imgsz: int,
) -> dict[str, HazardGroupRecall]:
    """
    Split known-ground-truth val images into "single_hazard" (exactly one
    distinct hazard class present) and "multi_hazard" (2+ distinct hazard
    classes present, i.e. hazards co-occurring in one frame), and compute
    the same IoU-based detection recall independently within each group.

    Returns {"single_hazard": HazardGroupRecall, "multi_hazard": HazardGroupRecall}.
    """
    from ultralytics import YOLO

    print(f"\n[Step 4b/4] Evaluating multi-hazard vs single-hazard recall breakdown "
          f"on {len(val_pairs)} val images...")

    model = YOLO(str(checkpoint_path))
    groups: dict[str, HazardGroupRecall] = {
        "single_hazard": HazardGroupRecall(group_name="single_hazard"),
        "multi_hazard": HazardGroupRecall(group_name="multi_hazard"),
    }

    for img_path, label_path in val_pairs:
        gt_annotations = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_label_line(line)
            if parsed and parsed[0] in hazard_classes:
                gt_annotations.append(parsed)

        if not gt_annotations:
            continue

        distinct_classes = {cid for cid, _ in gt_annotations}
        group = groups["multi_hazard"] if len(distinct_classes) >= 2 else groups["single_hazard"]
        group.image_count += 1

        results = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf_threshold,
            device=device, verbose=False,
        )

        pred_boxes_by_class: dict[int, list[tuple[float, float, float, float]]] = {}
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxyn = boxes.xyxyn.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for box, cid in zip(xyxyn, cls_ids):
                pred_boxes_by_class.setdefault(int(cid), []).append(tuple(box.tolist()))

        for class_id, points in gt_annotations:
            gt_bbox = polygon_to_bbox(points)
            group.ground_truth_count += 1

            candidates = pred_boxes_by_class.get(class_id, [])
            best_iou = max((iou_xyxy(gt_bbox, pred) for pred in candidates), default=0.0)
            if best_iou >= iou_threshold:
                group.detected_count += 1

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-training sanity check: verify the CNN fallback pipeline "
                     "can detect synthetic hazards injected into normal-operations images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--roboflow_dir", type=Path, default=Path("roboflow data"),
                   help="Roboflow dataset used as the source of hazard asset patches")
    p.add_argument("--normal_dir", type=Path, default=resolve_default_normal_dir(),
                   help="Normal-operations screenshot dataset (no real hazards). Defaults to "
                        f"'{PREFERRED_NORMAL_DIR}' if present, otherwise falls back to "
                        f"'{FALLBACK_NORMAL_DIR}' when the real dataset is unavailable on this "
                        "machine (e.g. it lives on a separate, access-restricted device)")
    p.add_argument("--synthetic_dir", type=Path, default=Path("image_data_normal_hazard_sanity"),
                   help="Where synthetic hazard-injected images are written")
    p.add_argument("--split_dir", type=Path, default=Path("image_data_normal_hazard_sanity_split"),
                   help="Where the train/val YOLO split + data.yaml is written")
    p.add_argument("--injections_per_image", type=int, default=2,
                   help="Hazard objects injected per background image")
    p.add_argument("--max_images", type=int, default=300,
                   help="Cap on normal images used to build the synthetic set "
                        "(sanity check doesn't need the full dataset)")
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="Fraction of synthetic images held out for evaluation")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=15,
                   help="Epochs for the quick sanity-check training run "
                        "(kept small — this is a viability check, not a final model)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="If given, fine-tune this checkpoint instead of training from scratch "
                        "(e.g. runs/train/hazard_yolo/weights/best.pt)")
    p.add_argument("--project_name", type=str, default="hazard_sanity_check")

    p.add_argument("--conf_threshold", type=float, default=0.35,
                   help="Confidence threshold for evaluation-time inference")
    p.add_argument("--iou_threshold", type=float, default=0.4,
                   help="Minimum IoU between prediction and known ground truth to count as detected")
    p.add_argument("--recall_threshold", type=float, default=0.5,
                   help="Per-class recall required to PASS the sanity check")

    p.add_argument("--skip_generation", action="store_true",
                   help="Reuse an existing --synthetic_dir instead of regenerating it")
    p.add_argument("--skip_training", action="store_true",
                   help="Reuse an existing --checkpoint instead of training")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint to evaluate when --skip_training is set")

    args = p.parse_args()

    print("=== CNN Fallback Pipeline — Pre-Training Hazard Sanity Check ===")
    print(f"  Hazard classes checked: {hazard_gen.HAZARD_CLASSES}")

    # ── Step 1: synthetic hazard injection ──────────────────────────────────
    if not args.skip_generation:
        if not args.normal_dir.exists():
            if str(args.normal_dir) == PREFERRED_NORMAL_DIR and Path(FALLBACK_NORMAL_DIR).exists():
                print(f"  [warn] '{PREFERRED_NORMAL_DIR}' not found on this machine "
                      f"(it lives on a separate device) — falling back to '{FALLBACK_NORMAL_DIR}'")
                args.normal_dir = Path(FALLBACK_NORMAL_DIR)
            else:
                print(f"ERROR: --normal_dir does not exist: {args.normal_dir}")
                sys.exit(1)
        if str(args.normal_dir) == FALLBACK_NORMAL_DIR:
            print(f"  [warn] Using '{FALLBACK_NORMAL_DIR}' as background imagery. "
                  f"This dataset already contains real hazard instances, so recall results "
                  f"from this run confirm the mechanics work but are NOT representative of "
                  f"performance on the real hazard-free normal-operations footage.")
        generate_synthetic_hazards(
            roboflow_dir=args.roboflow_dir,
            normal_dir=args.normal_dir,
            synthetic_dir=args.synthetic_dir,
            injections_per_image=args.injections_per_image,
            max_images=args.max_images,
            seed=args.seed,
        )
    else:
        print(f"\n[Step 1/4] Skipped (--skip_generation) — reusing {args.synthetic_dir}")
        if not args.synthetic_dir.exists():
            print(f"ERROR: --synthetic_dir does not exist and --skip_generation was set: "
                  f"{args.synthetic_dir}")
            sys.exit(1)

    # ── Step 2: build train/val split ────────────────────────────────────────
    data_yaml, val_pairs = build_yolo_split(
        synthetic_dir=args.synthetic_dir,
        split_dir=args.split_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        class_names=hazard_gen.CLASS_NAMES,
    )

    # ── Step 3: train or reuse a checkpoint ──────────────────────────────────
    if args.skip_training:
        if not args.checkpoint:
            print("ERROR: --skip_training requires --checkpoint to be provided.")
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: --checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        print(f"\n[Step 3/4] Skipped (--skip_training) — using {checkpoint_path}")
    else:
        checkpoint_path = train_sanity_check_model(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            imgsz=args.imgsz,
            device=args.device,
            base_checkpoint=args.base_checkpoint,
            project_name=args.project_name,
        )

    # ── Step 4: evaluate recall against known synthetic ground truth ───────
    recall_results = evaluate_hazard_recall(
        checkpoint_path=checkpoint_path,
        val_pairs=val_pairs,
        hazard_classes=hazard_gen.HAZARD_CLASSES,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        imgsz=args.imgsz,
    )

    # ── Step 4b: diagnostic-only multi-hazard vs single-hazard breakdown ────
    # Does not affect the PASS/FAIL verdict below -- that is still decided
    # solely by per-class recall from evaluate_hazard_recall() above.
    hazard_group_results = evaluate_multi_hazard_breakdown(
        checkpoint_path=checkpoint_path,
        val_pairs=val_pairs,
        hazard_classes=hazard_gen.HAZARD_CLASSES,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        imgsz=args.imgsz,
    )

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n=== Sanity Check Results ===")
    all_passed = True
    summary_classes = {}
    for cid, result in recall_results.items():
        status = "PASS" if result.recall >= args.recall_threshold else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{cid}] {result.class_name:<28} "
              f"recall={result.recall:.2f}  "
              f"({result.detected_count}/{result.ground_truth_count} detected)  [{status}]")
        summary_classes[result.class_name] = {
            "class_id": cid,
            "ground_truth_count": result.ground_truth_count,
            "detected_count": result.detected_count,
            "recall": round(result.recall, 4),
            "status": status,
        }

    verdict = "PASS" if all_passed else "FAIL"
    print(f"\n  Overall verdict: {verdict}")
    if all_passed:
        print("  The CNN fallback pipeline can detect the injected hazard classes.")
        print("  It's viable to proceed with cnn_auto_annotate.py / run_auto_annotate.py "
              "--pipeline cnn on the full normal-operations dataset.")
    else:
        print("  At least one hazard class fell below the recall threshold.")
        print("  Consider: more synthetic training examples, more epochs, or falling back")
        print("  to the segmentation pipeline (auto_annotate.py) for those classes.")

    # ── Multi-hazard breakdown (diagnostic only, does not affect verdict) ──
    print("\n=== Multi-Hazard Co-Occurrence Breakdown (diagnostic only) ===")
    print("  The fallback detects boxes only; it cannot pixel-disambiguate")
    print("  overlapping hazards the way the segmentation pipeline can. This")
    print("  shows whether recall on THIS checkpoint holds up when multiple")
    print("  hazard classes co-occur in the same frame vs. when only one does.")
    summary_hazard_groups = {}
    for group_name in ("single_hazard", "multi_hazard"):
        group = hazard_group_results[group_name]
        print(f"  [{group_name:<13}] {group.image_count:3d} images, "
              f"recall={group.recall:.2f}  ({group.detected_count}/{group.ground_truth_count} detected)")
        summary_hazard_groups[group_name] = {
            "image_count": group.image_count,
            "ground_truth_count": group.ground_truth_count,
            "detected_count": group.detected_count,
            "recall": round(group.recall, 4),
        }
    multi = hazard_group_results["multi_hazard"]
    single = hazard_group_results["single_hazard"]
    if multi.image_count == 0:
        print("  No multi-hazard images found in the validation split -- increase "
              "--injections_per_image to exercise this case.")
    elif multi.recall < single.recall:
        print(f"  Recall drops by {single.recall - multi.recall:.2f} on multi-hazard frames "
              f"relative to single-hazard frames.")
    else:
        print("  No recall drop observed on multi-hazard frames relative to single-hazard frames.")

    summary = {
        "checkpoint": str(checkpoint_path),
        "synthetic_dir": str(args.synthetic_dir),
        "split_dir": str(args.split_dir),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "hazard_group_breakdown": summary_hazard_groups,
        "recall_threshold": args.recall_threshold,
        "classes": summary_classes,
        "verdict": verdict,
    }
    summary_path = args.split_dir / "sanity_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary written to: {summary_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
