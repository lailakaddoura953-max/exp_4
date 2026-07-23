"""Package image_data_with_synth/ into a trainable YOLO train/val/test split.

image_data_with_synth/ is organized by CONCERN, not by train/val/test:

    image_data_with_synth/
      augmented_hazards/<location>/<day|night>/*.PNG (+ labels)
      normal_operations/
        augmented_normal/<location>/<day|night>/*.PNG (+ labels)
        auto_accepted/<location>/... (+ labels)   <- depth/day-night shape
                                                       may differ; this
                                                       script doesn't care,
                                                       it recurses.

Ultralytics YOLO training expects a train/images + train/labels (+ val/,
optionally test/) layout with a data.yaml describing it. This script
performs ONLY that packaging step:

  - Discovers every (image, label) pair across all three buckets
  - Shuffles deterministically and splits into train/val(/test)
  - COPIES (never moves) files into a new output directory, using short
    deterministic filenames — the same fix already used in
    scripts/pretrain_hazard_sanity_check.py's build_yolo_split() for the
    Windows MAX_PATH issue with long source filenames
  - Writes a data.yaml pointing at the new split

This script does NOT:
  - Re-evaluate or change any label's hazard/normal status
  - Touch bounding box / polygon coordinates
  - Modify anything under image_data_with_synth/ itself — it is read-only
    input; all output goes to a separate directory (--output_dir)

By default the written data.yaml uses the full 17-class taxonomy
(FULL_CLASS_NAMES from src/hazard_detection/rule_engine/class_taxonomy.py),
matching the class indices already present in the label files, unmodified.

Pass --reduced_classes to instead filter out the 5 dropped classes and
remap the remaining indices to the 12-class Reduced_Class_Set — this is
an INDEX remap/filter only (never touches box/polygon coordinates).
Images whose only label lines reference dropped classes become
empty-label (background) images in that mode, not deleted or skipped.

Usage:
    python scripts/package_image_data_with_synth.py
    python scripts/package_image_data_with_synth.py --reduced_classes
    python scripts/package_image_data_with_synth.py --val_fraction 0.15 --test_fraction 0.1
    python scripts/package_image_data_with_synth.py --synth_dir "image_data_with_synth" --output_dir "image_data_with_synth_split"
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, "src")

from hazard_detection.rule_engine.class_taxonomy import (  # noqa: E402
    DROPPED_CLASS_INDICES,
    FULL_CLASS_NAMES,
    FULL_TO_REDUCED_INDEX,
    REDUCED_CLASS_SET,
)

# The three buckets under image_data_with_synth/ that contain real
# image+label pairs to package. augmented_hazards/ holds injected-hazard
# examples; the two normal_operations/ subfolders hold non-hazard
# examples (augmented_normal/ from the same injection pipeline as
# augmented_hazards/, auto_accepted/ from the CNN/segmentation
# auto-annotation pipelines — see scripts/annotation/auto_annotate.py).
BUCKETS = [
    "augmented_hazards",
    "normal_operations/augmented_normal",
    "normal_operations/auto_accepted",
]

IMAGE_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


def discover_pairs(synth_dir: Path) -> Tuple[List[Tuple[Path, Path, str]], int]:
    """
    Find every (image, label, bucket_name) triple across all three
    buckets, recursively — deliberately not assuming a fixed
    <location>/<day|night>/ depth, since auto_accepted/'s depth mirrors
    whatever its source tree looked like and may not match
    augmented_hazards/'s and augmented_normal/'s shape.

    Deduplicated by resolved path: on case-insensitive filesystems
    (Windows, default macOS), globbing both "*.png" and "*.PNG" would
    otherwise discover the same file twice.

    Images with no matching label file are SKIPPED (not packaged as
    empty-label background images) — this mirrors
    pretrain_hazard_sanity_check.py's discover_pairs() convention. Skipped
    counts are reported by the caller, not silently dropped.
    """
    seen: set = set()
    pairs: List[Tuple[Path, Path, str]] = []
    skipped_no_label = 0

    for bucket in BUCKETS:
        bucket_dir = synth_dir / bucket
        if not bucket_dir.exists():
            continue

        for ext in IMAGE_EXTENSIONS:
            for img_path in bucket_dir.rglob(f"*{ext}"):
                resolved = img_path.resolve()
                if resolved in seen:
                    continue
                label_path = img_path.with_suffix(".txt")
                if not label_path.exists():
                    seen.add(resolved)
                    skipped_no_label += 1
                    continue
                seen.add(resolved)
                pairs.append((img_path, label_path, bucket))

    return sorted(pairs, key=lambda t: str(t[0])), skipped_no_label


def _remap_label_line(line: str) -> Optional[str]:
    """
    Remap a single YOLO label line's class_id from the full 17-class
    index to the 12-class Reduced_Class_Set index. Returns None if the
    line's class is one of the dropped classes (caller drops the line
    entirely) or if the line can't be parsed. Coordinates are copied
    through byte-for-byte, never recalculated.
    """
    parts = line.strip().split()
    if not parts:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        return None

    if class_id in DROPPED_CLASS_INDICES:
        return None

    new_class_id = FULL_TO_REDUCED_INDEX.get(class_id)
    if new_class_id is None:
        return None

    return " ".join([str(new_class_id)] + parts[1:])


def _write_label(dest_path: Path, source_label_path: Path, reduced_classes: bool) -> None:
    """
    Copy a label file to dest_path, optionally remapping class indices
    to the Reduced_Class_Set (index-only remap; coordinates untouched).
    """
    if not reduced_classes:
        shutil.copy2(source_label_path, dest_path)
        return

    original_lines = source_label_path.read_text(encoding="utf-8").splitlines()
    remapped_lines = [
        remapped for line in original_lines
        if (remapped := _remap_label_line(line)) is not None
    ]
    # Write even if remapped_lines is empty — an image whose only labels
    # were dropped classes becomes a valid empty-label background image,
    # not a deleted/skipped one.
    dest_path.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")


def build_split(
    synth_dir: Path,
    output_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    reduced_classes: bool,
    dry_run: bool,
) -> Tuple[Path, dict]:
    """
    Discover pairs, split into train/val(/test), copy into output_dir
    with short deterministic filenames, and write output_dir/data.yaml.

    Returns (data_yaml_path, report) where report is a dict of counts
    suitable for printing or JSON-dumping.
    """
    pairs, skipped_no_label = discover_pairs(synth_dir)

    if not pairs:
        print(f"ERROR: No image/label pairs found under {synth_dir}. "
              f"Checked buckets: {BUCKETS}")
        sys.exit(1)

    bucket_counts: dict = {}
    for _, _, bucket in pairs:
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(1, int(n_total * val_fraction)) if val_fraction > 0 else 0
    n_test = int(n_total * test_fraction) if test_fraction > 0 else 0

    val_pairs = shuffled[:n_val]
    test_pairs = shuffled[n_val:n_val + n_test]
    train_pairs = shuffled[n_val + n_test:]

    if not train_pairs:
        # Extremely small datasets: guarantee at least one training example.
        train_pairs = [val_pairs.pop()] if val_pairs else [test_pairs.pop()]

    class_names = REDUCED_CLASS_SET if reduced_classes else FULL_CLASS_NAMES

    report = {
        "input_dir": str(synth_dir),
        "output_dir": str(output_dir),
        "per_bucket_counts": bucket_counts,
        "skipped_no_matching_label": skipped_no_label,
        "reduced_classes": reduced_classes,
        "class_count": len(class_names),
        "split_counts": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "dry_run": dry_run,
    }

    if dry_run:
        return output_dir / "data.yaml", report

    if output_dir.exists():
        shutil.rmtree(output_dir)

    def _copy_subset(subset_pairs: List[Tuple[Path, Path, str]], subset_name: str) -> None:
        img_dir = output_dir / subset_name / "images"
        lbl_dir = output_dir / subset_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_path, label_path, _bucket) in enumerate(subset_pairs):
            dest_img = img_dir / f"{subset_name}_{i:05d}{img_path.suffix}"
            dest_lbl = lbl_dir / f"{subset_name}_{i:05d}.txt"
            shutil.copy2(img_path, dest_img)
            _write_label(dest_lbl, label_path, reduced_classes)

    _copy_subset(train_pairs, "train")
    _copy_subset(val_pairs, "val")
    if test_pairs:
        _copy_subset(test_pairs, "test")

    data_yaml_path = output_dir / "data.yaml"
    yaml_lines = [
        "train: train/images",
        "val: val/images",
    ]
    if test_pairs:
        yaml_lines.append("test: test/images")
    yaml_lines += [
        "",
        f"nc: {len(class_names)}",
        f"names: {json.dumps(class_names)}",
    ]
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return data_yaml_path, report


def print_report(report: dict) -> None:
    print()
    print("=" * 70)
    print(" IMAGE_DATA_WITH_SYNTH/ PACKAGING REPORT")
    print("=" * 70)
    print(f" Input directory  : {report['input_dir']}")
    print(f" Output directory : {report['output_dir']}")
    print(f" Class list       : "
          f"{'Reduced_Class_Set (12 classes)' if report['reduced_classes'] else 'FULL_CLASS_NAMES (17 classes)'}")
    print()
    print(" Pairs found per bucket:")
    for bucket, count in report["per_bucket_counts"].items():
        print(f"   {bucket:40s}: {count:5d}")
    print(f"   {'(skipped, no matching label file)':40s}: {report['skipped_no_matching_label']:5d}")
    print()
    print(" Split sizes:")
    for split_name, count in report["split_counts"].items():
        print(f"   {split_name:6s}: {count:5d}")
    print("=" * 70)
    if report["dry_run"]:
        print(" DRY RUN — no files were written.")
        print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package image_data_with_synth/ into a trainable YOLO train/val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synth_dir", type=Path, default=Path("image_data_with_synth"),
                        help="Root of the image_data_with_synth/ tree to package.")
    parser.add_argument("--output_dir", type=Path, default=Path("image_data_with_synth_split"),
                        help="Where to write the packaged train/val/test split + data.yaml. "
                             "This is a SEPARATE directory — image_data_with_synth/ itself is "
                             "never modified.")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of pairs held out for validation.")
    parser.add_argument("--test_fraction", type=float, default=0.0,
                        help="Fraction of pairs held out for testing (0 = no test split).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reduced_classes", action="store_true",
                        help="Write data.yaml with the 12-class Reduced_Class_Set and remap "
                             "label indices accordingly (index-only remap; box/polygon "
                             "coordinates are never changed). Without this flag, the full "
                             "17-class taxonomy is used, matching label files unmodified.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report counts without writing any files.")
    args = parser.parse_args()

    if not args.synth_dir.exists():
        print(
            f"ERROR: '{args.synth_dir}' was not found on this machine. "
            f"image_data_with_synth/ lives on a separate device per the project's "
            f"documented setup — check on your device for this folder and the exact "
            f"path needed."
        )
        sys.exit(1)

    data_yaml_path, report = build_split(
        synth_dir=args.synth_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        reduced_classes=args.reduced_classes,
        dry_run=args.dry_run,
    )

    print_report(report)

    if not args.dry_run:
        print()
        print(f" data.yaml written to: {data_yaml_path}")
        print()
        print(" To train against this split:")
        print(f'   python scripts/train_yolo.py --data "{data_yaml_path}"')
        print()
        print(" To evaluate a checkpoint against this split:")
        print(f'   python scripts/evaluate_yolo.py --data "{data_yaml_path}"')


if __name__ == "__main__":
    main()
