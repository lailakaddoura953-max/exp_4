"""
Auto-Annotation Pipeline — Grounded SAM 2
==========================================
Runs Grounding DINO + SAM 2 over the raw normal-operations image dataset and
produces YOLO polygon label files. Detections are split into two queues:

  auto_accepted/  — confidence >= review_threshold  → treat as ground truth
  review_queue/   — confidence_threshold <= conf < review_threshold → human review
  rejected/       — conf < confidence_threshold → discarded (logged only)

Output mirrors the input directory tree under --output_dir.

Usage:
    python scripts/annotation/auto_annotate.py \\
        --input_dir  image_data_normal \\
        --output_dir image_data_annotated \\
        --confidence 0.35 \\
        --review_threshold 0.55

    # Verify model setup without processing real data:
    python scripts/annotation/auto_annotate.py --verify

See scripts/annotation/SETUP.md for installation instructions.
"""

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — edit these paths before running on the private machine
# ─────────────────────────────────────────────────────────────────────────────
GSAM2_REPO       = r"C:\path\to\Grounded-SAM-2"
SAM2_CHECKPOINT  = r"C:\path\to\Grounded-SAM-2\checkpoints\sam2.1_hiera_large.pt"
SAM2_CONFIG      = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_CHECKPOINT = r"C:\path\to\Grounded-SAM-2\gdino_checkpoints\groundingdino_swint_ogc.pth"
GDINO_CONFIG     = r"C:\path\to\Grounded-SAM-2\grounding_dino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DEVICE           = "cuda"   # "cuda" or "cpu"
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Text prompts for each YOLO class we want to detect.
# Grounding DINO uses these as open-vocabulary queries.
# Tune these if detections are poor for a particular class.
# Format: dot-separated phrases work best with Grounding DINO.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_PROMPTS = {
    0:  "boat with cargo",
    1:  "misaligned shipping container",
    2:  "shipping container with open door. open container door",
    3:  "container being lifted by crane. picked container",
    4:  "reefer container. refrigerated container",
    5:  "wet container. water on container",
    6:  "separate shipping container",
    7:  "stacked shipping containers",
    8:  "crane. port crane. container crane",
    9:  "person wearing safety vest and helmet. worker in hi-vis",
    10: "person without safety equipment. worker without vest or helmet",
    11: "truck without container. empty truck",
    12: "truck with shipping container",
    13: "vehicle. forklift. reach stacker",
    14: "dropoff zone marking. painted ground marking",
    15: "empty yard area. restricted zone",
    16: "operation zone. active work area",
}

# Classes that are safety-critical — lower auto-accept threshold applied
SAFETY_CRITICAL_CLASSES = {2, 9, 10}  # open container, human variants

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add Grounded-SAM-2 repo to path so its modules resolve correctly
sys.path.insert(0, GSAM2_REPO)
sys.path.insert(0, os.path.join(GSAM2_REPO, "grounding_dino"))

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — only loaded after path is set up.
# Wrapped in functions so --verify can catch import errors cleanly.
# ─────────────────────────────────────────────────────────────────────────────
def _load_models():
    """Load Grounding DINO and SAM 2 image predictor. Returns (gdino, sam2)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model as load_gdino

    device = DEVICE if DEVICE == "cuda" and _cuda_available() else "cpu"
    if device == "cpu":
        print("  [warn] CUDA not available — running on CPU (slow)")

    print(f"  Loading Grounding DINO from {GDINO_CHECKPOINT}")
    gdino = load_gdino(GDINO_CONFIG, GDINO_CHECKPOINT)
    gdino = gdino.to(device).eval()

    print(f"  Loading SAM 2 from {SAM2_CHECKPOINT}")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return gdino, sam2_predictor, device


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Chrome cropping
# Ocularis screenshots include UI chrome (title bar, sidebar, status bar).
# We crop a configurable margin from each edge before running inference,
# then map detections back to original image coordinates for labelling.
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
    """
    Shift bounding boxes from cropped-image space back to original image space.
    boxes_xyxy: (N, 4) array in pixel coords relative to cropped image.
    Returns (N, 4) array in pixel coords relative to original image.
    """
    ox, oy = offset_xy
    shifted = boxes_xyxy.copy().astype(float)
    shifted[:, 0] += ox
    shifted[:, 1] += oy
    shifted[:, 2] += ox
    shifted[:, 3] += oy
    # Clamp to original image bounds
    shifted[:, 0] = np.clip(shifted[:, 0], 0, orig_w)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, orig_h)
    shifted[:, 2] = np.clip(shifted[:, 2], 0, orig_w)
    shifted[:, 3] = np.clip(shifted[:, 3], 0, orig_h)
    return shifted

# ─────────────────────────────────────────────────────────────────────────────
# Grounding DINO detection
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_prompt(class_ids: list[int]) -> tuple[str, dict[str, int]]:
    """
    Build a single Grounding DINO text prompt from all target classes.
    Returns (prompt_string, {phrase -> class_id}).
    Grounding DINO expects dot-separated phrases.
    """
    phrase_to_class: dict[str, int] = {}
    parts = []
    for cid in class_ids:
        phrase = CLASS_PROMPTS[cid]
        # Use only the first sub-phrase as the lookup key
        key = phrase.split(".")[0].strip().lower()
        phrase_to_class[key] = cid
        parts.append(phrase)
    prompt = " . ".join(parts) + " ."
    return prompt, phrase_to_class


def run_gdino(
    gdino_model,
    image_pil: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Run Grounding DINO inference.
    Returns (boxes_xyxy_norm, scores, phrases).
    boxes are normalised [0,1] in xyxy format.
    """
    from groundingdino.util.inference import predict as gdino_predict

    boxes, logits, phrases = gdino_predict(
        model=gdino_model,
        image=image_pil,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    # boxes from gdino are (cx, cy, w, h) normalised → convert to xyxy
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), []

    boxes_np = boxes.cpu().numpy()
    cx, cy, bw, bh = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    scores = logits.cpu().numpy()
    return boxes_xyxy, scores, phrases


def match_phrase_to_class(phrase: str, phrase_to_class: dict[str, int]) -> Optional[int]:
    """
    Map a Grounding DINO output phrase back to a YOLO class ID.
    Uses substring matching since gdino may truncate or paraphrase.
    Returns None if no match found.
    """
    phrase_lower = phrase.lower().strip()
    # Exact key match first
    if phrase_lower in phrase_to_class:
        return phrase_to_class[phrase_lower]
    # Substring: does any known key appear in the phrase?
    for key, cid in phrase_to_class.items():
        if key in phrase_lower or phrase_lower in key:
            return cid
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SAM 2 segmentation
# ─────────────────────────────────────────────────────────────────────────────

def run_sam2(
    sam2_predictor,
    image_np: np.ndarray,
    boxes_xyxy_px: np.ndarray,
) -> list[np.ndarray]:
    """
    Run SAM 2 with bounding box prompts.
    image_np: BGR numpy array (full original-resolution image).
    boxes_xyxy_px: (N, 4) pixel-space xyxy boxes.
    Returns list of binary masks (H x W bool), one per box.
    """
    import torch

    if len(boxes_xyxy_px) == 0:
        return []

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(image_rgb)

    boxes_tensor = torch.tensor(boxes_xyxy_px, dtype=torch.float32)

    masks_out, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_tensor,
        multimask_output=False,
    )
    # masks_out shape: (N, 1, H, W) or (N, H, W) depending on SAM version
    results = []
    for i in range(masks_out.shape[0]):
        m = masks_out[i]
        if m.ndim == 3:
            m = m[0]
        results.append(m.astype(bool))
    return results


def mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int, simplify_epsilon: float = 2.0) -> Optional[list[tuple[float, float]]]:
    """
    Convert a binary mask to a normalised YOLO polygon.
    Uses the largest contour. Returns None if mask is empty or too small.
    simplify_epsilon: Douglas-Peucker epsilon in pixels (larger = fewer points).
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take the largest contour by area
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 16:  # too tiny
        return None

    # Simplify to reduce point count
    epsilon = simplify_epsilon
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    if len(simplified) < 3:
        return None

    # Normalise to [0, 1]
    pts = simplified.reshape(-1, 2)
    norm_pts = [(float(pt[0]) / img_w, float(pt[1]) / img_h) for pt in pts]
    return norm_pts

# ─────────────────────────────────────────────────────────────────────────────
# YOLO label I/O
# ─────────────────────────────────────────────────────────────────────────────

def polygon_to_label_line(class_id: int, points: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def write_label_file(path: Path, annotations: list[tuple[int, list[tuple[float, float]]]]) -> None:
    lines = [polygon_to_label_line(cid, pts) for cid, pts in annotations]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_metadata(path: Path, detections: list[dict]) -> None:
    """Write a sidecar JSON with confidence scores for the review tool."""
    path.write_text(json.dumps(detections, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    gdino_model,
    sam2_predictor,
    device: str,
    text_prompt: str,
    phrase_to_class: dict[str, int],
    args,
) -> dict:
    """
    Run the full pipeline on one image.
    Returns a result dict with keys: status, accepted, review, rejected.
    """
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        return {"status": "unreadable", "accepted": [], "review": [], "rejected": []}

    orig_h, orig_w = image_bgr.shape[:2]

    # ── 1. Crop UI chrome ────────────────────────────────────────────────────
    cropped, offset_xy = crop_chrome(
        image_bgr,
        margin_top=args.chrome_top,
        margin_bottom=args.chrome_bottom,
        margin_left=args.chrome_left,
        margin_right=args.chrome_right,
    )
    crop_h, crop_w = cropped.shape[:2]

    # ── 2. Grounding DINO ────────────────────────────────────────────────────
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(cropped_rgb)

    boxes_norm, scores, phrases = run_gdino(
        gdino_model, image_pil, text_prompt,
        box_threshold=args.confidence,
        text_threshold=args.confidence * 0.8,
        device=device,
    )

    if len(boxes_norm) == 0:
        return {"status": "no_detections", "accepted": [], "review": [], "rejected": []}

    # Convert normalised boxes → pixel boxes in cropped space
    boxes_px_crop = boxes_norm.copy()
    boxes_px_crop[:, [0, 2]] *= crop_w
    boxes_px_crop[:, [1, 3]] *= crop_h

    # Shift back to original image space
    boxes_px_orig = offset_boxes_to_original(boxes_px_crop, offset_xy, orig_w, orig_h)

    # ── 3. SAM 2 segmentation ─────────────────────────────────────────────────
    masks = run_sam2(sam2_predictor, image_bgr, boxes_px_orig)

    # ── 4. Route each detection ──────────────────────────────────────────────
    accepted, review, rejected = [], [], []

    for i, (score, phrase, mask, box_orig) in enumerate(
        zip(scores, phrases, masks, boxes_px_orig)
    ):
        class_id = match_phrase_to_class(phrase, phrase_to_class)
        if class_id is None:
            rejected.append({"phrase": phrase, "score": float(score), "reason": "no_class_match"})
            continue

        polygon = mask_to_polygon(mask, orig_w, orig_h, simplify_epsilon=args.simplify)
        if polygon is None:
            rejected.append({"class_id": class_id, "phrase": phrase, "score": float(score), "reason": "empty_mask"})
            continue

        entry = {
            "class_id": class_id,
            "phrase": phrase,
            "score": float(score),
            "polygon": polygon,
            "box_px": box_orig.tolist(),
        }

        # Safety-critical classes use a higher review bar
        effective_threshold = (
            min(args.review_threshold, args.review_threshold * 0.85)
            if class_id in SAFETY_CRITICAL_CLASSES
            else args.review_threshold
        )

        if score >= effective_threshold:
            accepted.append(entry)
        else:
            review.append(entry)

    return {
        "status": "ok",
        "accepted": accepted,
        "review": review,
        "rejected": rejected,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    result: dict,
) -> None:
    """
    Write image + label files to the appropriate output subdirectory.

    auto_accepted/ → high-confidence annotations, ready for training
    review_queue/  → uncertain, needs human review
    Both directories mirror the source tree structure.
    """
    try:
        rel = img_path.relative_to(input_root)
    except ValueError:
        rel = Path(img_path.name)

    accepted = result["accepted"]
    review    = result["review"]

    # Write auto-accepted labels
    if accepted:
        acc_dir = output_root / "auto_accepted" / rel.parent
        acc_dir.mkdir(parents=True, exist_ok=True)
        # Copy image
        dest_img = acc_dir / img_path.name
        cv2.imwrite(str(dest_img), cv2.imread(str(img_path)))
        # Write label
        annotations = [(e["class_id"], e["polygon"]) for e in accepted]
        write_label_file(dest_img.with_suffix(".txt"), annotations)

    # Write review-queue items (image + labels + sidecar JSON with scores)
    if review:
        rev_dir = output_root / "review_queue" / rel.parent
        rev_dir.mkdir(parents=True, exist_ok=True)
        dest_img = rev_dir / img_path.name
        cv2.imwrite(str(dest_img), cv2.imread(str(img_path)))
        # Draft labels (annotator will edit these)
        annotations = [(e["class_id"], e["polygon"]) for e in review]
        write_label_file(dest_img.with_suffix(".txt"), annotations)
        # Sidecar with confidence scores for the review tool
        meta_path = rev_dir / (img_path.stem + ".meta.json")
        write_review_metadata(meta_path, review)


# ─────────────────────────────────────────────────────────────────────────────
# Verify command — loads models and runs on a synthetic test image
# ─────────────────────────────────────────────────────────────────────────────

def run_verify() -> None:
    print("\n=== Verifying Grounded SAM 2 setup ===")
    print("  Checking CUDA...")
    cuda_ok = _cuda_available()
    print(f"  CUDA available: {cuda_ok}")

    print("  Loading models (this may take 30-60s on first load)...")
    try:
        gdino, sam2, device = _load_models()
    except Exception as e:
        print(f"\n  ERROR loading models: {e}")
        print("  Check SETUP.md for installation steps.")
        sys.exit(1)

    print("  Models loaded OK")
    print(f"  Device: {device}")

    # Run a trivial inference on a blank image to confirm the pipeline works
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (80, 80, 80)  # grey
    test_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

    prompt, p2c = build_combined_prompt([9, 10])
    boxes, scores, phrases = run_gdino(gdino, test_pil, prompt, 0.35, 0.28, device)
    print(f"  Grounding DINO test inference: {len(boxes)} detections on blank image (expected 0)")

    masks = run_sam2(sam2, test_img, np.zeros((0, 4)))
    print(f"  SAM 2 test inference: {len(masks)} masks on empty boxes (expected 0)")

    print("\n  Setup verified. Ready to annotate.")

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"ERROR: --input_dir does not exist: {input_root}")
        sys.exit(1)

    print("\n=== Grounded SAM 2 Auto-Annotation ===")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"  Confidence threshold : {args.confidence}")
    print(f"  Review threshold     : {args.review_threshold}")
    print(f"  Classes              : {args.classes if args.classes else 'all 17'}")

    # ── Discover images ──────────────────────────────────────────────────────
    all_images: list[Path] = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"):
        all_images.extend(input_root.rglob(ext))
    all_images = sorted(all_images)

    if args.limit:
        all_images = all_images[:args.limit]

    print(f"  Images to process    : {len(all_images)}")

    if args.dry_run:
        print("\n  [DRY RUN] Would process:")
        for p in all_images[:5]:
            print(f"    {p}")
        if len(all_images) > 5:
            print(f"    ... and {len(all_images) - 5} more")
        return

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n  Loading models...")
    t0 = time.time()
    try:
        gdino_model, sam2_predictor, device = _load_models()
    except Exception as e:
        print(f"ERROR: Could not load models: {e}")
        print("Run with --verify first to diagnose the issue.")
        sys.exit(1)
    print(f"  Models loaded in {time.time() - t0:.1f}s on {device}")

    # ── Build combined text prompt ────────────────────────────────────────────
    target_classes = args.classes if args.classes else list(CLASS_PROMPTS.keys())
    text_prompt, phrase_to_class = build_combined_prompt(target_classes)
    print(f"  Text prompt: {text_prompt[:80]}...")

    # ── Process images ────────────────────────────────────────────────────────
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

    for img_path in tqdm(all_images, desc="Annotating", unit="img"):
        try:
            result = process_image(
                img_path, gdino_model, sam2_predictor, device,
                text_prompt, phrase_to_class, args,
            )
        except Exception as e:
            tqdm.write(f"  [error] {img_path.name}: {e}")
            result = {"status": "error", "accepted": [], "review": [], "rejected": []}

        stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1
        if result["status"] == "ok":
            stats["ok"] += 1
        elif result["status"] == "no_detections":
            stats["no_detections"] += 1
        elif result["status"] == "unreadable":
            stats["unreadable"] += 1

        stats["accepted_annotations"] += len(result["accepted"])
        stats["review_annotations"]   += len(result["review"])
        stats["rejected_annotations"] += len(result["rejected"])

        if result["accepted"] or result["review"]:
            save_results(img_path, input_root, output_root, result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Done ===")
    print(f"  Images processed     : {stats['total']}")
    print(f"  With detections      : {stats['ok']}")
    print(f"  No detections        : {stats['no_detections']}")
    print(f"  Unreadable           : {stats['unreadable']}")
    print(f"  Auto-accepted labels : {stats['accepted_annotations']}")
    print(f"  Review-queue labels  : {stats['review_annotations']}")
    print(f"  Rejected (no match)  : {stats['rejected_annotations']}")
    print(f"\n  Output: {output_root}")
    print(f"    auto_accepted/ — ready for training")
    print(f"    review_queue/  — run review_annotations.py to process")

    # Write summary JSON
    summary_path = output_root / "annotation_summary.json"
    summary_path.write_text(json.dumps({
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-annotate images using Grounded SAM 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  type=str, default="image_data_normal",
                   help="Root directory of raw screenshot images")
    p.add_argument("--output_dir", type=str, default="image_data_annotated",
                   help="Root output directory for label files")
    p.add_argument("--confidence", type=float, default=0.35,
                   help="Minimum Grounding DINO box confidence to keep a detection")
    p.add_argument("--review_threshold", type=float, default=0.55,
                   help="Confidence at or above which a detection is auto-accepted. "
                        "Below this it goes to the review queue.")
    # Chrome crop margins (pixels)
    p.add_argument("--chrome_top",    type=int, default=60,
                   help="Pixels to crop from top (Ocularis title bar)")
    p.add_argument("--chrome_bottom", type=int, default=30,
                   help="Pixels to crop from bottom (status bar)")
    p.add_argument("--chrome_left",   type=int, default=220,
                   help="Pixels to crop from left (camera tree sidebar)")
    p.add_argument("--chrome_right",  type=int, default=10,
                   help="Pixels to crop from right")
    # Polygon simplification
    p.add_argument("--simplify", type=float, default=2.0,
                   help="Douglas-Peucker epsilon for polygon simplification (pixels). "
                        "Higher = fewer polygon points.")
    # Which classes to detect (default: all 17)
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Space-separated class IDs to detect. Default: all 17. "
                        "Example: --classes 2 9 10  (open container, human variants only)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images (useful for testing)")
    p.add_argument("--dry_run", action="store_true",
                   help="List images that would be processed without running inference")
    p.add_argument("--verify", action="store_true",
                   help="Load models and run a sanity check, then exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        run_verify()
    else:
        run(args)
