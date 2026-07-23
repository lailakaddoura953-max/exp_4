"""Step 8c - Run inference on test images (Roboflow or image_data_with_synth/).

Saves annotated images with bounding boxes to runs/detect/test_run/.

Usage:
    python scripts/run_on_test_images.py
    python scripts/run_on_test_images.py --checkpoint checkpoints/yolov12_best.pt
    python scripts/run_on_test_images.py --conf 0.3 --device cpu
    python scripts/run_on_test_images.py --source "image_data_with_synth/augmented_hazards"
                                                    # run inference against an
                                                    # image_data_with_synth/ subfolder or its
                                                    # full tree instead of roboflow data/test/images
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

# requirements.md Requirement 15.3: --source can resolve to any subfolder
# or the full tree of image_data_with_synth/, in addition to the existing
# default of roboflow data/test/images. Works for plain inference (no
# labels required) exactly as it does for roboflow data.
#
# image_data_with_synth/ is NOT present in this workspace checkout (it
# lives on a separate device per the project's documented setup). If you
# pass --source pointing into image_data_with_synth/, check on your
# device for this folder and the exact path needed.
IMAGE_DATA_WITH_SYNTH_MARKER = "image_data_with_synth"


def _describe_dataset_source(source_path: str) -> str:
    """Same dataset-source labeling convention as train_yolo.py/evaluate_yolo.py."""
    normalized = source_path.replace("\\", "/").lower()
    if IMAGE_DATA_WITH_SYNTH_MARKER in normalized:
        if "reclassified" in normalized or "corrected" in normalized:
            return "image_data_with_synth/ (corrected)"
        return "image_data_with_synth/ (raw)"
    return "roboflow data/"


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Run YOLO inference on test images.")
    parser.add_argument("--checkpoint", type=str,
                        default="runs/train/hazard_yolo/weights/best.pt")
    parser.add_argument("--source",     type=str,
                        default="roboflow data/test/images",
                        help="Image folder to run inference on. Defaults to "
                             "'roboflow data/test/images'. To run against "
                             "image_data_with_synth/ instead, pass any subfolder or "
                             "the full tree path directly -- check on your device for "
                             "this folder and the exact path needed, since "
                             "image_data_with_synth/ is not present in every checkout. "
                             "No labels are required for plain inference either way.")
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--name",       type=str,   default="test_run")
    args = parser.parse_args()

    dataset_source = _describe_dataset_source(args.source)
    print(f"Dataset source: {dataset_source}  (--source {args.source})")
    if not Path(args.source).exists():
        print(
            f"  [warn] '{args.source}' was not found on this machine. If this is "
            f"meant to point into image_data_with_synth/, check on your device "
            f"for this folder and the path needed -- it may live on a separate "
            f"device per the project's documented setup."
        )

    model = YOLO(args.checkpoint)
    results = model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        save=True,
        project="runs/detect",
        name=args.name,
    )
    print(f"\nProcessed {len(results)} images")
    print(f"Annotated results saved to: runs/detect/{args.name}/")


if __name__ == '__main__':
    main()
