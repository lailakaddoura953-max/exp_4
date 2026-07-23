"""Step 2 - Inspect dataset splits.

Two reporting modes (requirements.md Requirement 15.4):

  1. Roboflow mode (default) - flat train/valid/test image+label counts,
     matching the original behavior of this script unchanged.

  2. image_data_with_synth/ mode (--source image_data_with_synth) - a
     DISTINCT reporting mode, since that dataset's directory shape (per
     location, per day/night, hazard vs. normal) does not match the flat
     roboflow data/ train/valid/test split. This is not a forced reuse of
     the roboflow counting logic.

Usage:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --source "roboflow data"
    python scripts/check_dataset.py --source image_data_with_synth

image_data_with_synth/ is NOT present in this workspace checkout (it lives
on a separate device per the project's documented setup). If you run this
script with --source image_data_with_synth and it reports "NOT FOUND",
check on your device for this folder and the exact path needed.
"""
import argparse
import os
import sys

IMAGE_DATA_WITH_SYNTH_MARKER = "image_data_with_synth"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _count_images_and_labels(images_dir: str, labels_dir: str) -> tuple[int, int]:
    imgs = len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]) if os.path.exists(images_dir) else 0
    lbls = len([
        f for f in os.listdir(labels_dir) if f.endswith(".txt")
    ]) if os.path.exists(labels_dir) else 0
    return imgs, lbls


def check_roboflow_dataset(dataset_root: str) -> None:
    """
    Report flat train/valid/test image+label counts for a roboflow-style
    dataset. Unchanged from this script's original behavior.
    """
    print(f"Roboflow-style dataset at: {dataset_root}\n")
    if not os.path.exists(dataset_root):
        print(f"  [warn] '{dataset_root}' was not found on this machine.")
        return

    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(dataset_root, split, "images")
        labels_dir = os.path.join(dataset_root, split, "labels")
        if os.path.exists(images_dir):
            imgs, lbls = _count_images_and_labels(images_dir, labels_dir)
            print(f"{split:6s}: {imgs:4d} images,  {lbls:4d} labels")
        else:
            print(f"{split:6s}: NOT FOUND at {images_dir}")


def check_image_data_with_synth(dataset_root: str) -> None:
    """
    Report per-location, per-day/night, hazard-vs-normal counts for
    image_data_with_synth/'s directory structure:

        image_data_with_synth/
          augmented_hazards/<location>/<day|night>/*.PNG (+ labels)
          normal_operations/
            augmented_normal/<location>/<day|night>/*.PNG (+ labels)
            auto_accepted/<location>/<day|night>/*.PNG (+ labels)

    This is a DISTINCT reporting mode from check_roboflow_dataset() —
    the directory shapes are not the same, so the counting logic is not
    forced to reuse the flat train/valid/test walk (requirements.md
    Requirement 15.4).
    """
    print(f"image_data_with_synth/-style dataset at: {dataset_root}\n")

    if not os.path.exists(dataset_root):
        print(
            f"  [warn] '{dataset_root}' was not found on this machine. "
            f"image_data_with_synth/ lives on a separate device per the "
            f"project's documented setup — check on your device for this "
            f"folder and the exact path needed."
        )
        return

    buckets = {
        "augmented_hazards": os.path.join(dataset_root, "augmented_hazards"),
        "normal_operations/augmented_normal": os.path.join(
            dataset_root, "normal_operations", "augmented_normal"
        ),
        "normal_operations/auto_accepted": os.path.join(
            dataset_root, "normal_operations", "auto_accepted"
        ),
    }

    grand_total_images = 0
    grand_total_labels = 0

    for bucket_name, bucket_path in buckets.items():
        print(f"[{bucket_name}]")
        if not os.path.exists(bucket_path):
            print(f"  NOT FOUND at {bucket_path}\n")
            continue

        bucket_total_images = 0
        bucket_total_labels = 0

        locations = sorted(
            d for d in os.listdir(bucket_path)
            if os.path.isdir(os.path.join(bucket_path, d))
        )
        if not locations:
            print("  (no location folders found)\n")
            continue

        for location in locations:
            location_path = os.path.join(bucket_path, location)
            for day_night in ("day", "night"):
                dn_path = os.path.join(location_path, day_night)
                if not os.path.exists(dn_path):
                    continue
                # image_data_with_synth/ images/labels sit directly in the
                # <location>/<day|night>/ folder, not in images/labels
                # subfolders, per requirements.md Requirement 11.1's
                # documented structure.
                imgs = len([
                    f for f in os.listdir(dn_path)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                ])
                lbls = len([
                    f for f in os.listdir(dn_path) if f.endswith(".txt")
                ])
                if imgs or lbls:
                    print(
                        f"  {location:30s} {day_night:6s}: "
                        f"{imgs:4d} images,  {lbls:4d} labels"
                    )
                bucket_total_images += imgs
                bucket_total_labels += lbls

        print(
            f"  {'TOTAL':30s} {'':6s}: {bucket_total_images:4d} images,  "
            f"{bucket_total_labels:4d} labels\n"
        )
        grand_total_images += bucket_total_images
        grand_total_labels += bucket_total_labels

    print(
        f"[grand total across all buckets] {grand_total_images:4d} images,  "
        f"{grand_total_labels:4d} labels"
    )
    print(
        f"  hazard examples come from 'augmented_hazards' only; "
        f"'normal_operations/*' buckets are non-hazard by construction."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect dataset splits (roboflow data/ or image_data_with_synth/)."
    )
    parser.add_argument(
        "--source", type=str, default="roboflow data",
        help="Dataset root to inspect. Defaults to 'roboflow data'. Pass "
             "'image_data_with_synth' (or a path containing that name) to "
             "switch to the image_data_with_synth/ reporting mode.",
    )
    args = parser.parse_args()

    normalized = args.source.replace("\\", "/").lower()
    if IMAGE_DATA_WITH_SYNTH_MARKER in normalized:
        check_image_data_with_synth(args.source)
    else:
        check_roboflow_dataset(args.source)


if __name__ == "__main__":
    main()
