"""
CNN Auto-Annotation Pipeline — Fallback for Grounded SAM 2
============================================================
Runs a traditional CNN object detector (Ultralytics YOLOv12) over the raw
normal-operations image dataset and produces YOLO rectangle-polygon label
files, in the same auto_accepted / review_queue / rejected layout used by
auto_annotate.py (scripts/annotation/auto_annotate.py).

Why this exists
----------------
The primary pipeline (Grounded SAM 2 + Grounding DINO, see SETUP.md) is a
zero-shot, promptable segmentation stack. It needs a compiled CUDA build of
Grounding DINO, multi-GB checkpoints, and a dedicated virtual environment
(.venv_annotation). If that stack turns out to be unstable, too slow on the
private machine, or not worth maintaining, this script is the fallback:

    - Uses Ultralytics YOLOv12, a conventional single-stage CNN detector,
      already installed in the project's main virtual environment (.venv).
    - Requires a YOLO checkpoint trained on "roboflow data/data.yaml"
      (the same 17-class dataset used everywhere else in this repo).
      Train one first with scripts/train_yolo.py if you don't have one.
    - No SAM 2, no Grounding DINO, no second venv, no CUDA compilation step.
    - Produces bounding boxes only (encoded as 4-corner rectangle polygons
      so the label files and review_annotations.py continue to work
      unmodified) rather than pixel-accurate masks.

This script is intentionally self-contained (it does not import from
auto_annotate.py) so the fallback keeps working even if the segmentation
pipeline's dependencies are broken or removed.

Use scripts/annotation/run_auto_annotate.py to pick between this pipeline
and the segmentation one from a single entry point.

Usage:
    python scripts/annotation/cnn_auto_annotate.py \\
        --input_dir  image_data_normal \\
        --output_dir image_data_annotated_cnn \\
        --checkpoint runs/train/hazard_yolo/weights/best.pt \\
        --confidence 0.35 \\
        --review_threshold 0.55

    # Verify the checkpoint loads and can run inference, without touching
    # any real data:
    python scripts/annotation/cnn_auto_annotate.py --verify \\
        --checkpoint runs/train/hazard_yolo/weights/best.pt

See scripts/annotation/SETUP.md, section "Fallback: CNN-Based Pipeline".
"""

# ─────────────────────────────────────────────────────────────────────────────
# The 17 classes, in the exact order used by roboflow data/data.yaml and
# scripts/generate_hazard_augmentations.py. Must stay in this order — it is
# the order a checkpoint trained via scripts/train_yolo.py will emit.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Boat - With Cargo",         # 0
    "Container - Misaligned",    # 1
    "Container - Open",          # 2
    "Container - Picked",        # 3
    "Container - Reefer",        # 4
    "Container - Water Drop",    # 5
    "Container -Separate",       # 6
    "Container -Stacked",        # 7
    "Crane",                     # 8
    "Human",                     # 9
    "Human - No Safety Clothes", # 10
    "Truck - No Container",      # 11
    "Truck - With Container",    # 12
    "Vehicle",                   # 13
    "Yard - Dropoff zone",       # 14
    "Yard - No People",          # 15
    "Yard - Operation Zone",     # 16
]

# Same safety-critical set as auto_annotate.py — lower effective accept bar,
# these route to human review more readily than other classes.
SAFETY_CRITICAL_CLASSES = {2, 9, 10}

DEFAULT_CHECKPOINT = "runs/train/hazard_yolo/weights/best.pt"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Input dataset resolution
# ─────────────────────────────────────────────────────────────────────────────
# image_data_normal (the real "normal operations" dataset) lives on a
# separate, access-restricted device and is not always present on this
# machine. Rather than fail outright, default to the Roboflow dataset
# ("roboflow data") as a stand-in input so the pipeline still has real
# images to exercise for setup/testing purposes.
PREFERRED_INPUT_DIR = "image_data_normal"
FALLBACK_INPUT_DIR = "roboflow data"


def resolve_default_input_dir() -> str:
    """Return image_data_normal if present, otherwise roboflow data."""
    return PREFERRED_INPUT_DIR if Path(PREFERRED_INPUT_DIR).exists() else FALLBACK_INPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Chrome cropping — identical convention to auto_annotate.py so both
# pipelines treat Ocularis screenshot UI chrome the same way. Duplicated
# rather than imported to keep this fallback independent of the
# segmentation script.
# ─────────────────────────────────────────────────────────────────────────────

def crop_chrome(
    image: np.ndarray,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop UI chrome margins from a screenshot.
    Returns (cropped_image, (offset_x, offset_y)) where offset is the
    pixel position of the crop's top-left corner in the original image.
    """
    h, w = image.shape[:2]
    y1 = min(margin_top, h - 1)
    y2 = max(h - margin_bottom, y1 + 1)
    x1 = min(margin_left, w - 1)
    x2 = max(w - margin_right, x1 + 1)
    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1)


def offset_boxes_to_original(
    boxes_xyxy: np.ndarray,
    offset_xy: tuple[int, int],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Shift pixel-space boxes from cropped-image space back to original image space."""
    ox, oy = offset_xy
    shifted = boxes_xyxy.copy().astype(float)
    shifted[:, 0] += ox
    shifted[:, 1] += oy
    shifted[:, 2] += ox
    shifted[:, 3] += oy
    shifted[:, 0] = np.clip(shifted[:, 0], 0, orig_w)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, orig_h)
    shifted[:, 2] = np.clip(shifted[:, 2], 0, orig_w)
    shifted[:, 3] = np.clip(shifted[:, 3], 0, orig_h)
    return shifted


# ─────────────────────────────────────────────────────────────────────────────
# Label I/O — a plain CNN detector only produces boxes, not masks, so we
# encode each box as a 4-corner rectangle polygon. This keeps the label
# format identical to auto_annotate.py's polygon output and lets
# review_annotations.py work on either pipeline's output unmodified.
# ─────────────────────────────────────────────────────────────────────────────

def box_to_yolo_polygon(box_xyxy: np.ndarray, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Convert a pixel-space xyxy box into a normalised 4-corner polygon."""
    x1, y1, x2, y2 = box_xyxy
    nx1, ny1 = x1 / img_w, y1 / img_h
    nx2, ny2 = x2 / img_w, y2 / img_h
    return [(nx1, ny1), (nx2, ny1), (nx2, ny2), (nx1, ny2)]


def polygon_to_label_line(class_id: int, points: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def write_label_file(path: Path, annotations: list[tuple[int, list[tuple[float, float]]]]) -> None:
    lines = [polygon_to_label_line(cid, pts) for cid, pts in annotations]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_metadata(path: Path, detections: list[dict]) -> None:
    path.write_text(json.dumps(detections, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _resolve_device(requested_device: str) -> str:
    """Fall back to CPU if CUDA was requested but is unavailable."""
    if requested_device == "cuda" and not _cuda_available():
        print("  [warn] CUDA requested but not available — falling back to CPU (slow)")
        return "cpu"
    return requested_device


def load_model(checkpoint_path: str, device: str):
    """Load an Ultralytics YOLO checkpoint. Returns (model, resolved_device)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics package is required for the CNN fallback pipeline. "
            "It should already be installed in the project's main .venv "
            "(pip install ultralytics)."
        ) from exc

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"YOLO checkpoint not found at '{checkpoint_path}'. "
            f"Train one first with:\n"
            f"    python scripts/train_yolo.py\n"
            f"which writes to runs/train/hazard_yolo/weights/best.pt"
        )

    resolved_device = _resolve_device(device)
    print(f"  Loading YOLO checkpoint from '{checkpoint_path}' on device '{resolved_device}'")
    model = YOLO(str(checkpoint))
    return model, resolved_device


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    model,
    device: str,
    args,
) -> dict:
    """
    Run the CNN detector on one image. Mirrors auto_annotate.py's
    process_image() routing so both pipelines behave the same way for
    downstream consumers (review_annotations.py, training scripts).
    """
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return {"status": "unreadable", "accepted": [], "review": [], "rejected": []}

    orig_h, orig_w = image_bgr.shape[:2]

    cropped, offset_xy = crop_chrome(
        image_bgr,
        margin_top=args.chrome_top,
        margin_bottom=args.chrome_bottom,
        margin_left=args.chrome_left,
        margin_right=args.chrome_right,
    )

    results = model.predict(
        source=cropped,
        imgsz=args.imgsz,
        conf=args.confidence,
        device=device,
        verbose=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return {"status": "no_detections", "accepted": [], "review": [], "rejected": []}

    boxes = results[0].boxes
    target_classes = set(args.classes) if args.classes else None

    boxes_xyxy_crop = boxes.xyxy.cpu().numpy()
    boxes_xyxy_orig = offset_boxes_to_original(boxes_xyxy_crop, offset_xy, orig_w, orig_h)

    accepted, review, rejected = [], [], []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])

        if class_id >= len(CLASS_NAMES):
            rejected.append({"class_id": class_id, "score": confidence, "reason": "unknown_class"})
            continue
        if target_classes is not None and class_id not in target_classes:
            continue

        polygon = box_to_yolo_polygon(boxes_xyxy_orig[i], orig_w, orig_h)

        entry = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "score": confidence,
            "polygon": polygon,
            "box_px": boxes_xyxy_orig[i].tolist(),
        }

        effective_threshold = (
            min(args.review_threshold, args.review_threshold * 0.85)
            if class_id in SAFETY_CRITICAL_CLASSES
            else args.review_threshold
        )

        if confidence >= effective_threshold:
            accepted.append(entry)
        else:
            review.append(entry)

    return {"status": "ok", "accepted": accepted, "review": review, "rejected": rejected}


# ─────────────────────────────────────────────────────────────────────────────
# Output writing — identical directory layout to auto_annotate.py
# ─────────────────────────────────────────────────────────────────────────────

def save_results(img_path: Path, input_root: Path, output_root: Path, result: dict) -> None:
    try:
        rel = img_path.relative_to(input_root)
    except ValueError:
        rel = Path(img_path.name)

    accepted = result["accepted"]
    review = result["review"]

    if accepted:
        acc_dir = output_root / "auto_accepted" / rel.parent
        acc_dir.mkdir(parents=True, exist_ok=True)
        dest_img = acc_dir / img_path.name
        cv2.imwrite(str(dest_img), cv2.imread(str(img_path)))
        annotations = [(e["class_id"], e["polygon"]) for e in accepted]
        write_label_file(dest_img.with_suffix(".txt"), annotations)

    if review:
        rev_dir = output_root / "review_queue" / rel.parent
        rev_dir.mkdir(parents=True, exist_ok=True)
        dest_img = rev_dir / img_path.name
        cv2.imwrite(str(dest_img), cv2.imread(str(img_path)))
        annotations = [(e["class_id"], e["polygon"]) for e in review]
        write_label_file(dest_img.with_suffix(".txt"), annotations)
        meta_path = rev_dir / (img_path.stem + ".meta.json")
        write_review_metadata(meta_path, review)


# ─────────────────────────────────────────────────────────────────────────────
# Verify command
# ─────────────────────────────────────────────────────────────────────────────

def run_verify(args) -> None:
    print("\n=== Verifying CNN fallback pipeline setup ===")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ERROR: the 'ultralytics' package is required for the CNN fallback pipeline "
            "but is not importable in the active environment.\n"
            "Install it with:\n"
            "    pip install ultralytics"
        )
        sys.exit(1)

    print(f"  CUDA available: {_cuda_available()}")

    try:
        model, device = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"\n  ERROR loading model: {e}")
        sys.exit(1)

    print(f"  Model loaded OK on device '{device}'")

    test_img = np.full((480, 640, 3), 80, dtype=np.uint8)
    results = model.predict(source=test_img, imgsz=args.imgsz, conf=args.confidence, device=device, verbose=False)
    n_dets = 0 if not results or results[0].boxes is None else len(results[0].boxes)
    print(f"  Test inference on blank image: {n_dets} detections (expected 0)")
    print("\n  Setup verified. Ready to annotate.")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        if args.input_dir == PREFERRED_INPUT_DIR and Path(FALLBACK_INPUT_DIR).exists():
            print(f"  [warn] '{PREFERRED_INPUT_DIR}' not found on this machine "
                  f"(it lives on a separate device) — falling back to '{FALLBACK_INPUT_DIR}'")
            input_root = Path(FALLBACK_INPUT_DIR)
        else:
            print(f"ERROR: --input_dir does not exist: {input_root}")
            sys.exit(1)

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ERROR: the 'ultralytics' package is required for the CNN fallback pipeline "
            "but is not importable in the active environment.\n"
            "Install it with:\n"
            "    pip install ultralytics"
        )
        sys.exit(1)

    print("\n=== CNN Fallback Auto-Annotation (Ultralytics YOLO) ===")
    print(f"  Input      : {input_root}")
    print(f"  Output     : {output_root}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Confidence threshold : {args.confidence}")
    print(f"  Review threshold     : {args.review_threshold}")
    print(f"  Classes              : {args.classes if args.classes else 'all 17'}")

    # Deduplicated by resolved path: on case-insensitive filesystems
    # (Windows, default macOS) "*.png" already matches "IMG.PNG", so
    # iterating both patterns would otherwise discover -- and process --
    # every image twice.
    seen_images: set[Path] = set()
    all_images: list[Path] = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"):
        for img_path in input_root.rglob(ext):
            resolved = img_path.resolve()
            if resolved not in seen_images:
                seen_images.add(resolved)
                all_images.append(img_path)
    all_images = sorted(all_images)

    if args.limit:
        all_images = all_images[: args.limit]

    print(f"  Images to process    : {len(all_images)}")

    if args.dry_run:
        print("\n  [DRY RUN] Would process:")
        for p in all_images[:5]:
            print(f"    {p}")
        if len(all_images) > 5:
            print(f"    ... and {len(all_images) - 5} more")
        return

    print("\n  Loading model...")
    t0 = time.time()
    try:
        model, device = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        sys.exit(1)
    print(f"  Model loaded in {time.time() - t0:.1f}s on {device}")

    stats = {
        "total": len(all_images),
        "ok": 0,
        "no_detections": 0,
        "unreadable": 0,
        "accepted_annotations": 0,
        "review_annotations": 0,
        "rejected_annotations": 0,
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(all_images, desc="Annotating (CNN)", unit="img"):
        try:
            result = process_image(img_path, model, device, args)
        except Exception as e:
            tqdm.write(f"  [error] {img_path.name}: {e}")
            result = {"status": "error", "accepted": [], "review": [], "rejected": []}

        if result["status"] == "ok":
            stats["ok"] += 1
        elif result["status"] == "no_detections":
            stats["no_detections"] += 1
        elif result["status"] == "unreadable":
            stats["unreadable"] += 1

        stats["accepted_annotations"] += len(result["accepted"])
        stats["review_annotations"] += len(result["review"])
        stats["rejected_annotations"] += len(result["rejected"])

        if result["accepted"] or result["review"]:
            save_results(img_path, input_root, output_root, result)

    print("\n=== Done ===")
    print(f"  Images processed     : {stats['total']}")
    print(f"  With detections      : {stats['ok']}")
    print(f"  No detections        : {stats['no_detections']}")
    print(f"  Unreadable           : {stats['unreadable']}")
    print(f"  Auto-accepted labels : {stats['accepted_annotations']}")
    print(f"  Review-queue labels  : {stats['review_annotations']}")
    print(f"  Rejected (no match)  : {stats['rejected_annotations']}")
    print(f"\n  Output: {output_root}")
    print("    auto_accepted/ — ready for training")
    print("    review_queue/  — run review_annotations.py to process")

    summary_path = output_root / "annotation_summary.json"
    summary_path.write_text(json.dumps({
        "pipeline": "cnn_fallback",
        "checkpoint": args.checkpoint,
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "confidence": args.confidence,
        "review_threshold": args.review_threshold,
        "chrome_margins": {
            "top": args.chrome_top, "bottom": args.chrome_bottom,
            "left": args.chrome_left, "right": args.chrome_right,
        },
        "stats": stats,
    }, indent=2), encoding="utf-8")
    print(f"\n  Summary written to: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — deliberately mirrors auto_annotate.py's arguments so the two
# pipelines are drop-in swappable via run_auto_annotate.py.
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fallback auto-annotation using a traditional CNN detector (Ultralytics YOLO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir", type=str, default=resolve_default_input_dir(),
                   help="Root directory of raw screenshot images. Defaults to "
                        f"'{PREFERRED_INPUT_DIR}' if present, otherwise falls back "
                        f"to '{FALLBACK_INPUT_DIR}' (e.g. when the real normal-operations "
                        "dataset lives on a separate, access-restricted device)")
    p.add_argument("--output_dir", type=str, default="image_data_annotated_cnn",
                   help="Root output directory for label files")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to a YOLOv12 checkpoint trained on roboflow data/data.yaml")
    p.add_argument("--confidence", type=float, default=0.35,
                   help="Minimum detection confidence to keep")
    p.add_argument("--review_threshold", type=float, default=0.55,
                   help="Confidence at or above which a detection is auto-accepted. "
                        "Below this it goes to the review queue.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference resolution (square)")
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--chrome_top", type=int, default=60)
    p.add_argument("--chrome_bottom", type=int, default=30)
    p.add_argument("--chrome_left", type=int, default=220)
    p.add_argument("--chrome_right", type=int, default=10)
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Space-separated class IDs to keep. Default: all 17.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images")
    p.add_argument("--dry_run", action="store_true",
                   help="List images that would be processed without running inference")
    p.add_argument("--verify", action="store_true",
                   help="Load the model and run a sanity check, then exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        run_verify(args)
    else:
        run(args)
