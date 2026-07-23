"""
Synthetic Hazard Injection Script
==================================
Takes normal-operations screenshots and injects hazard objects extracted from
the Roboflow dataset (Container - Open, Human, Human - No Safety Clothes) to
produce synthetic hazard training images with YOLO polygon labels.

Usage:
    python scripts/generate_hazard_augmentations.py \
        --roboflow_dir "roboflow data" \
        --normal_dir "image_data_normal" \
        --output_dir "image_data_normal_augmented_hazards" \
        --injections_per_image 2 \
        --max_images 500 \
        --seed 42

Output structure mirrors the input normal_dir tree:
    image_data_normal_augmented_hazards/
        berth_401/normal_operations/day/
            Screenshot_2026XXX_hazard_001.png
            Screenshot_2026XXX_hazard_001.txt   <- YOLO polygon labels
        ...

Label format (YOLO polygon, same as roboflow dataset):
    class_id x1 y1 x2 y2 ... xN yN   (normalised 0-1)

Notes:
- Annotations are polygon format (not bbox), matching the roboflow dataset style.
- Objects are extracted using their polygon mask so they composite cleanly.
- Basic brightness/contrast matching is applied to help objects blend.
- The Ocularis UI chrome at the edges of screenshots is avoided during placement.
- Each output image gets at least one hazard object; class is chosen randomly
  from those that have available assets.
- Run with --dry_run to see counts without writing files.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────── Class IDs (from data.yaml) ───────────────────────
# Sourced from src/hazard_detection/rule_engine/class_taxonomy.py — the single
# shared definition of the class list, per requirements.md Requirement 12.5.
# This is deliberately imported rather than re-typed here, so this script's
# class indices can never drift out of sync with the shared taxonomy (or with
# scripts/pretrain_hazard_sanity_check.py, which imports the same module).
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from hazard_detection.rule_engine.class_taxonomy import FULL_CLASS_NAMES as CLASS_NAMES  # noqa: E402

#some classes are likely going to be removed or added as the project evolves
#and the needs of supervisors/stakeholders changes
#
# NOTE: this script currently injects against the FULL 17-class taxonomy
# (CLASS_NAMES = FULL_CLASS_NAMES), since it extracts hazard assets from
# roboflow data/, which is still labeled with the original 17 classes
# (requirements.md Requirement 12.6 — existing datasets are not required to
# drop classes in place). HAZARD_CLASSES below references indices 2, 9, 10,
# which are unaffected by the Reduced_Class_Set's drops and keep the same
# index in both FULL_CLASS_NAMES and REDUCED_CLASS_SET's source positions.

# The three hazard classes we inject
HAZARD_CLASSES = {
    2:  "Container - Open",
    9:  "Human",
    10: "Human - No Safety Clothes",
}

# Minimum polygon area (normalised) to accept an extracted asset.
# Tiny detections (far-away objects) make poor injection assets.
MIN_NORM_AREA = 0.002


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Asset Extraction
# ══════════════════════════════════════════════════════════════════════════════

def polygon_to_pixel(points_norm: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    """Convert normalised polygon points to pixel coordinates."""
    pts = [(int(x * w), int(y * h)) for x, y in points_norm]
    return np.array(pts, dtype=np.int32)


def polygon_area_norm(points_norm: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area in normalised coordinates."""
    n = len(points_norm)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points_norm[i]
        x2, y2 = points_norm[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def extract_asset(image: np.ndarray, points_norm: list[tuple[float, float]]) -> Optional[np.ndarray]:
    """
    Extract an object from an image using its polygon mask.

    Returns an RGBA patch (height x width x 4) where alpha=0 outside the
    polygon, or None if the polygon is degenerate.
    """
    h, w = image.shape[:2]
    pts_px = polygon_to_pixel(points_norm, w, h)

    # Bounding box of polygon
    x, y, bw, bh = cv2.boundingRect(pts_px)

    # Guard against zero-size or out-of-bounds boxes
    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 4 or bh <= 4:
        return None

    # Build mask for the full image, then crop
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_full, [pts_px], 255)

    crop_bgr = image[y:y+bh, x:x+bw].copy()
    crop_mask = mask_full[y:y+bh, x:x+bw]

    # Combine into RGBA
    crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = crop_mask

    return crop_bgra


def parse_label_line(line: str) -> Optional[tuple[int, list[tuple[float, float]]]]:
    """
    Parse one YOLO polygon label line.

    Format: class_id x1 y1 x2 y2 ... xN yN
    Returns (class_id, [(x1,y1), (x2,y2), ...]) or None on bad input.
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class + at least 3 xy pairs
        return None
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            return None
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        return class_id, points
    except ValueError:
        return None


def build_asset_pool(roboflow_dir: Path, target_classes: set[int], splits: list[str] = None) -> dict[int, list[np.ndarray]]:
    """
    Walk all splits in the roboflow dataset and extract RGBA patches for each
    target class. Returns {class_id: [rgba_patch, ...]}
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    pool: dict[int, list[np.ndarray]] = {c: [] for c in target_classes}
    total_scanned = 0
    total_extracted = 0

    for split in splits:
        images_dir = roboflow_dir / split / "images"
        labels_dir = roboflow_dir / split / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"  [skip] Split '{split}' not found at {roboflow_dir / split}")
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            # Find corresponding image (try jpg and png)
            stem = label_file.stem
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            total_scanned += 1
            lines = label_file.read_text().splitlines()

            for line in lines:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                class_id, points = parsed

                if class_id not in target_classes:
                    continue

                # Skip tiny objects — poor injection assets
                if polygon_area_norm(points) < MIN_NORM_AREA:
                    continue

                patch = extract_asset(image, points)
                if patch is None:
                    continue

                pool[class_id].append(patch)
                total_extracted += 1

    print(f"  Asset extraction: scanned {total_scanned} images, extracted {total_extracted} patches")
    for cid, patches in pool.items():
        print(f"    Class {cid} ({CLASS_NAMES[cid]}): {len(patches)} patches")

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Composition
# ══════════════════════════════════════════════════════════════════════════════

# Fraction of image edges to avoid (accounts for Ocularis UI chrome)
EDGE_MARGIN = 0.08


def brightness_match(patch_bgra: np.ndarray, background_region: np.ndarray) -> np.ndarray:
    """
    Adjust patch brightness/contrast to loosely match the background region.
    Works in LAB colour space on the L channel only.
    """
    patch_bgr = patch_bgra[:, :, :3].astype(np.float32)
    alpha = patch_bgra[:, :, 3]

    # Only consider non-transparent pixels
    mask = alpha > 128
    if not np.any(mask):
        return patch_bgra

    patch_lab = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(background_region.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    patch_L = patch_lab[:, :, 0][mask]
    bg_L = bg_lab[:, :, 0].flatten()

    if patch_L.std() < 1e-3 or bg_L.std() < 1e-3:
        return patch_bgra

    # Scale patch L channel to match background mean/std (clamped)
    scale = bg_L.std() / patch_L.std()
    shift = bg_L.mean() - patch_L.mean() * scale
    # Apply conservatively (blend 40% correction to avoid over-adjustment)
    alpha_blend = 0.4
    scale = 1.0 * (1 - alpha_blend) + scale * alpha_blend
    shift = 0.0 * (1 - alpha_blend) + shift * alpha_blend

    new_L = np.clip(patch_lab[:, :, 0] * scale + shift, 0, 255)
    patch_lab[:, :, 0] = new_L
    corrected_bgr = cv2.cvtColor(patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    result = patch_bgra.copy()
    result[:, :, :3] = corrected_bgr
    return result


def paste_patch(
    background: np.ndarray,
    patch_bgra: np.ndarray,
    top_left: tuple[int, int],
) -> np.ndarray:
    """
    Alpha-composite a BGRA patch onto a BGR background at top_left (x, y).
    Returns a new BGR image.
    """
    bg = background.copy()
    bh, bw = bg.shape[:2]
    ph, pw = patch_bgra.shape[:2]
    x, y = top_left

    # Clip patch to image bounds
    x_end = min(x + pw, bw)
    y_end = min(y + ph, bh)
    x = max(x, 0)
    y = max(y, 0)
    if x >= x_end or y >= y_end:
        return bg

    # Corresponding region of patch
    px_start = x - top_left[0]
    py_start = y - top_left[1]
    patch_crop = patch_bgra[py_start:py_start+(y_end-y), px_start:px_start+(x_end-x)]

    alpha = patch_crop[:, :, 3:4].astype(np.float32) / 255.0
    fg = patch_crop[:, :, :3].astype(np.float32)
    dst = bg[y:y_end, x:x_end].astype(np.float32)

    blended = fg * alpha + dst * (1.0 - alpha)
    bg[y:y_end, x:x_end] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def place_patch_random(
    background: np.ndarray,
    patch_bgra: np.ndarray,
    rng: random.Random,
    scale_range: tuple[float, float] = (0.12, 0.30),
    existing_boxes: list[tuple[int, int, int, int]] = None,
    max_attempts: int = 10,
) -> Optional[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """
    Resize patch to a random scale fraction of the background height, then
    place it at a random non-overlapping position (avoiding image edges).

    Returns (composited_image, (x, y, w, h)) in pixel coords, or None if
    no valid placement found within max_attempts.
    """
    bh, bw = background.shape[:2]
    ph, pw = patch_bgra.shape[:2]

    if existing_boxes is None:
        existing_boxes = []

    for _ in range(max_attempts):
        # Random scale relative to background height
        scale = rng.uniform(*scale_range)
        new_h = int(bh * scale)
        if new_h < 10:
            continue
        aspect = pw / ph if ph > 0 else 1.0
        new_w = int(new_h * aspect)
        if new_w < 4 or new_w > bw:
            continue

        resized = cv2.resize(patch_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Random position avoiding edges
        margin_x = int(bw * EDGE_MARGIN)
        margin_y = int(bh * EDGE_MARGIN)
        x_min = margin_x
        x_max = bw - new_w - margin_x
        y_min = margin_y
        y_max = bh - new_h - margin_y

        if x_max <= x_min or y_max <= y_min:
            continue

        # Brightness match against the background region before pasting
        bg_region = background[y_min:y_max, x_min:x_max]
        if bg_region.size > 0:
            sample_h = min(new_h, bg_region.shape[0])
            sample_w = min(new_w, bg_region.shape[1])
            bg_sample = bg_region[:sample_h, :sample_w]
            resized = brightness_match(resized, bg_sample)

        for attempt in range(max_attempts):
            x = rng.randint(x_min, x_max)
            y = rng.randint(y_min, y_max)

            # Check overlap with existing placed boxes
            new_box = (x, y, new_w, new_h)
            overlap = False
            for ex, ey, ew, eh in existing_boxes:
                ix1 = max(x, ex)
                iy1 = max(y, ey)
                ix2 = min(x + new_w, ex + ew)
                iy2 = min(y + new_h, ey + eh)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    own_area = new_w * new_h
                    if inter_area / own_area > 0.25:
                        overlap = True
                        break

            if not overlap:
                composited = paste_patch(background, resized, (x, y))
                return composited, new_box

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Annotation Writing
# ══════════════════════════════════════════════════════════════════════════════

def box_to_yolo_polygon(box: tuple[int, int, int, int], img_w: int, img_h: int) -> list[tuple[float, float]]:
    """
    Convert a pixel bounding box (x, y, w, h) to a 4-point normalised polygon.
    We use a rectangle polygon since we've lost the original mask after compositing.
    YOLO polygon format accepts rectangular polygons.
    """
    x, y, w, h = box
    x1 = x / img_w
    y1 = y / img_h
    x2 = (x + w) / img_w
    y2 = (y + h) / img_h
    # 4 corners: top-left, top-right, bottom-right, bottom-left
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_to_label_line(class_id: int, points: list[tuple[float, float]]) -> str:
    """Format a polygon as a YOLO label line."""
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{class_id} {coords}"


def write_label_file(
    label_path: Path,
    annotations: list[tuple[int, list[tuple[float, float]]]],
) -> None:
    """Write a YOLO polygon label file."""
    lines = [polygon_to_label_line(cid, pts) for cid, pts in annotations]
    label_path.write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Normal Image Discovery
# ══════════════════════════════════════════════════════════════════════════════

def discover_normal_images(normal_dir: Path) -> list[Path]:
    """
    Recursively find all PNG/JPG images under normal_dir.
    Expected structure: normal_dir / <location> / normal_operations / <day|night> / Screenshot*.png
    Any image file anywhere under the tree is accepted.
    """
    # Deduplicated by resolved path: on case-insensitive filesystems
    # (Windows, default macOS) "*.png" already matches "IMG.PNG", so
    # iterating both patterns would otherwise discover -- and inject
    # hazards into -- every image twice.
    seen: set[Path] = set()
    images = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"):
        for img_path in normal_dir.rglob(ext):
            resolved = img_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(img_path)
    return sorted(images)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(
    roboflow_dir: Path,
    normal_dir: Path,
    output_dir: Path,
    injections_per_image: int,
    max_images: Optional[int],
    seed: int,
    dry_run: bool,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    print("\n=== Synthetic Hazard Injection ===")
    print(f"  Roboflow data : {roboflow_dir}")
    print(f"  Normal images : {normal_dir}")
    print(f"  Output        : {output_dir}")
    print(f"  Injections/img: {injections_per_image}")
    print(f"  Seed          : {seed}")
    print(f"  Dry run       : {dry_run}")

    # ── Step 1: Build asset pool ──────────────────────────────────────────────
    print("\n[1/4] Extracting hazard asset patches from Roboflow dataset...")
    asset_pool = build_asset_pool(roboflow_dir, set(HAZARD_CLASSES.keys()))

    # Check we have at least one class with assets
    available_classes = [cid for cid, patches in asset_pool.items() if patches]
    if not available_classes:
        print("ERROR: No assets extracted. Check roboflow_dir and that label files exist.")
        sys.exit(1)

    print(f"  Available injection classes: {[CLASS_NAMES[c] for c in available_classes]}")

    # ── Step 2: Discover normal images ───────────────────────────────────────
    print("\n[2/4] Discovering normal operations images...")
    all_images = discover_normal_images(normal_dir)
    if not all_images:
        print(f"ERROR: No images found under {normal_dir}")
        sys.exit(1)

    if max_images and len(all_images) > max_images:
        rng_for_sample = random.Random(seed)
        all_images = rng_for_sample.sample(all_images, max_images)

    print(f"  Found {len(all_images)} images (using {len(all_images)})")

    if dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  {len(all_images)} augmented images")
        print(f"  ~{len(all_images) * injections_per_image} injected hazard objects")
        return

    # ── Step 3: Generate augmented images ────────────────────────────────────
    print("\n[3/4] Compositing hazard objects onto normal images...")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images_processed": 0, "images_skipped": 0, "injections": {c: 0 for c in HAZARD_CLASSES}}

    for img_idx, img_path in enumerate(all_images):
        if img_idx % 50 == 0:
            print(f"  Processing {img_idx + 1}/{len(all_images)}...")

        background = cv2.imread(str(img_path))
        if background is None:
            stats["images_skipped"] += 1
            continue

        bh, bw = background.shape[:2]

        # Mirror directory structure under output_dir
        try:
            rel_path = img_path.relative_to(normal_dir)
        except ValueError:
            rel_path = Path(img_path.name)

        out_img_dir = output_dir / rel_path.parent
        out_img_dir.mkdir(parents=True, exist_ok=True)

        # Generate one augmented image per normal image
        composited = background.copy()
        annotations: list[tuple[int, list[tuple[float, float]]]] = []
        placed_boxes: list[tuple[int, int, int, int]] = []

        # Randomly pick which hazard classes to inject for this image
        # Weighted slightly toward human hazards (more safety-relevant)
        injection_classes = rng.choices(
            available_classes,
            weights=[1 if c == 2 else 2 for c in available_classes],
            k=injections_per_image,
        )

        for class_id in injection_classes:
            patches = asset_pool[class_id]
            if not patches:
                continue

            patch = rng.choice(patches).copy()

            # Scale range depends on class: containers are larger than people
            if class_id == 2:  # Container - Open
                scale_range = (0.15, 0.35)
            else:              # Human variants
                scale_range = (0.08, 0.22)

            result = place_patch_random(
                composited, patch, rng,
                scale_range=scale_range,
                existing_boxes=placed_boxes,
            )

            if result is None:
                continue

            composited, box = result
            placed_boxes.append(box)
            stats["injections"][class_id] += 1

            # Write annotation as a 4-corner rectangle polygon
            poly_norm = box_to_yolo_polygon(box, bw, bh)
            annotations.append((class_id, poly_norm))

        # Only save if at least one injection succeeded
        if not annotations:
            stats["images_skipped"] += 1
            continue

        # Build output filename
        stem = img_path.stem
        out_stem = f"{stem}_hazard_{img_idx:04d}"
        out_img_path = out_img_dir / f"{out_stem}.png"
        out_lbl_path = out_img_dir / f"{out_stem}.txt"

        cv2.imwrite(str(out_img_path), composited)
        write_label_file(out_lbl_path, annotations)
        stats["images_processed"] += 1

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    print("\n[4/4] Done.")
    print(f"\n  Images generated : {stats['images_processed']}")
    print(f"  Images skipped   : {stats['images_skipped']}")
    print(f"  Injections by class:")
    for cid, count in stats["injections"].items():
        print(f"    [{cid}] {CLASS_NAMES[cid]}: {count}")

    # Write a summary JSON alongside the output
    summary_path = output_dir / "augmentation_summary.json"
    summary = {
        "roboflow_dir": str(roboflow_dir),
        "normal_dir": str(normal_dir),
        "output_dir": str(output_dir),
        "injections_per_image": injections_per_image,
        "seed": seed,
        "asset_pool_sizes": {CLASS_NAMES[c]: len(p) for c, p in asset_pool.items()},
        "stats": {
            "images_processed": stats["images_processed"],
            "images_skipped": stats["images_skipped"],
            "injections": {CLASS_NAMES[c]: v for c, v in stats["injections"].items()},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary written to: {summary_path}")
    print(f"  Output directory  : {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject synthetic hazard objects into normal-operations screenshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--roboflow_dir",
        type=Path,
        default=Path("roboflow data"),
        help="Root of the Roboflow dataset (contains train/, valid/, test/ and data.yaml)",
    )
    p.add_argument(
        "--normal_dir",
        type=Path,
        default=Path("image_data_normal"),
        help="Root of the normal-operations screenshot dataset",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("image_data_normal_augmented_hazards"),
        help="Where to write augmented images and YOLO label files",
    )
    p.add_argument(
        "--injections_per_image",
        type=int,
        default=2,
        help="How many hazard objects to inject per background image (1-5 recommended)",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of normal images processed (useful for quick tests)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        roboflow_dir=args.roboflow_dir,
        normal_dir=args.normal_dir,
        output_dir=args.output_dir,
        injections_per_image=args.injections_per_image,
        max_images=args.max_images,
        seed=args.seed,
        dry_run=args.dry_run,
    )
