"""
Augment train and test splits of a YOLO dataset.

Generates augmented copies of images with random transforms (rotation,
horizontal flip, brightness, contrast, noise, blur) and recalculates
bounding box coordinates for flipped images. Labels stay in sync.

Leaves valid/ completely untouched. Originals are never modified.

Usage:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --dataset "roboflow data" --copies 5
    python scripts/augment_dataset.py --dataset image_data_with_synth --copies 3
    python scripts/augment_dataset.py --dry-run
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment train and test splits of a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="roboflow data",
                        help="Path to dataset root (must contain train/images/ and train/labels/)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Which splits to augment")
    parser.add_argument("--copies", type=int, default=3,
                        help="Number of augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing files")
    return parser.parse_args()


# ============================================================================
# Augmentation transforms
# ============================================================================

def random_brightness(image, rng, low=0.7, high=1.3):
    """Adjust brightness by a random factor."""
    factor = rng.uniform(low, high)
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def random_contrast(image, rng, low=0.7, high=1.3):
    """Adjust contrast by a random factor."""
    factor = rng.uniform(low, high)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_noise(image, rng, std_range=(5, 25)):
    """Add Gaussian noise."""
    std = rng.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_blur(image, rng, max_k=5):
    """Apply Gaussian blur with a random kernel size."""
    k = rng.choice([3, 5]) if max_k >= 5 else 3
    return cv2.GaussianBlur(image, (k, k), 0)


def horizontal_flip(image):
    """Flip image horizontally."""
    return cv2.flip(image, 1)


def flip_label_line(line):
    """
    Flip a YOLO label line horizontally (mirror x coordinates).
    Works for both bbox format (class x_center y_center w h) and
    polygon format (class x1 y1 x2 y2 ... xN yN).
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return line

    class_id = parts[0]
    coords = [float(v) for v in parts[1:]]

    # Flip all x coordinates (even indices: 0, 2, 4, ...)
    flipped = []
    for i, val in enumerate(coords):
        if i % 2 == 0:  # x coordinate
            flipped.append(1.0 - val)
        else:  # y coordinate (unchanged)
            flipped.append(val)

    return class_id + " " + " ".join(f"{v:.6f}" for v in flipped)


def augment_image(image, rng):
    """
    Apply a random combination of 2-4 transforms to an image.
    Returns (augmented_image, was_flipped).
    """
    transforms = [
        ("brightness", lambda img: random_brightness(img, rng)),
        ("contrast", lambda img: random_contrast(img, rng)),
        ("noise", lambda img: random_noise(img, rng)),
        ("blur", lambda img: random_blur(img, rng)),
        ("flip", lambda img: horizontal_flip(img)),
    ]

    num_transforms = rng.randint(2, 4)
    selected = rng.sample(transforms, min(num_transforms, len(transforms)))

    was_flipped = False
    result = image.copy()

    for name, fn in selected:
        result = fn(result)
        if name == "flip":
            was_flipped = True

    return result, was_flipped


# ============================================================================
# Main augmentation logic
# ============================================================================

def augment_split(dataset_root, split, copies, seed, dry_run):
    """Augment one split (train or test)."""
    images_dir = Path(dataset_root) / split / "images"
    labels_dir = Path(dataset_root) / split / "labels"

    if not images_dir.exists():
        print(f"  [skip] {split}/images/ not found at {images_dir}")
        return 0

    # Collect image files
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and "_aug" not in f.stem  # don't re-augment previous augmentations
    ])

    if not image_files:
        print(f"  [skip] No original images found in {images_dir}")
        return 0

    print(f"  {len(image_files)} original images found in {split}/images/")
    print(f"  Generating {copies} augmented copies per image ({len(image_files) * copies} new images)...")

    if dry_run:
        return len(image_files) * copies

    rng = random.Random(seed)
    created = 0

    for img_path in image_files:
        # Read original image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Read matching label (if exists)
        label_path = labels_dir / (img_path.stem + ".txt")
        label_lines = []
        if label_path.exists():
            label_lines = label_path.read_text(encoding="utf-8").strip().splitlines()

        for copy_idx in range(copies):
            suffix = f"_aug{copy_idx + 1:02d}"
            dest_img = images_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
            dest_lbl = labels_dir / f"{img_path.stem}{suffix}.txt"

            # Apply augmentation
            aug_image, was_flipped = augment_image(image, rng)

            # Write augmented image
            cv2.imwrite(str(dest_img), aug_image)

            # Write label (with flipped coordinates if horizontally flipped)
            if label_lines:
                if was_flipped:
                    flipped_lines = [flip_label_line(line) for line in label_lines]
                    dest_lbl.write_text("\n".join(flipped_lines) + "\n", encoding="utf-8")
                else:
                    dest_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            created += 1

    return created


def main():
    args = parse_args()
    rng_seed = args.seed

    print()
    print("=" * 70)
    print(" YOLO DATASET AUGMENTATION")
    print("=" * 70)
    print(f" Dataset   : {args.dataset}")
    print(f" Splits    : {args.splits}")
    print(f" Copies    : {args.copies} per original image")
    if args.dry_run:
        print(" Mode      : DRY RUN (no files will be written)")
    print("=" * 70)
    print()

    total_originals = 0
    total_created = 0

    for split in args.splits:
        print(f"[{split}]")
        created = augment_split(args.dataset, split, args.copies, rng_seed, args.dry_run)
        total_created += created
        # Count originals for reporting
        images_dir = Path(args.dataset) / split / "images"
        if images_dir.exists():
            originals = len([
                f for f in images_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                and "_aug" not in f.stem
            ])
            total_originals += originals
            print(f"  Done: {originals} images processed, {created} copies created.")
        print()

    print()
    print(" AUGMENTATION COMPLETE")
    print(f" Total originals processed : {total_originals}")
    print(f" Total copies created      : {total_created}")
    print("=" * 70)


if __name__ == "__main__":
    main()
