"""Step 5 - Train YOLOv12 on the Roboflow dataset (or image_data_with_synth/).

Usage:
    python scripts/train_yolo.py                      # GPU, 100 epochs
    python scripts/train_yolo.py --device cpu         # CPU only
    python scripts/train_yolo.py --epochs 50          # fewer epochs
    python scripts/train_yolo.py --resume             # resume from last.pt
    python scripts/train_yolo.py --batch 8            # smaller batch (low VRAM)
    python scripts/train_yolo.py --data "image_data_with_synth_split/data.yaml"
                                                        # train against the packaged
                                                        # image_data_with_synth/ split

The best checkpoint is saved to:
    runs/train/hazard_yolo/weights/best.pt

Update config/hazard_detection.yaml to point at it after training:
    yolo:
      checkpoint_path: "runs/train/hazard_yolo/weights/best.pt"

NOTE on image_data_with_synth training:
    If you've already got image_data_with_synth_split/ sitting there from a
    previous run without --reduced_classes, the data.yaml will say nc=17 and
    training will use all 17 classes (hurting performance). To fix this, either:
      - Use launch_all.bat's Train option (it re-packages with --reduced_classes
        automatically), or
      - Manually run:
        python scripts/package_image_data_with_synth.py --reduced_classes --output_dir image_data_with_synth_split --val_fraction 0.15
    This regenerates data.yaml with nc=12 (the Reduced_Class_Set) and remaps
    all label indices accordingly.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

# requirements.md Requirement 15.1, 15.6: --data can point at either
# roboflow data/data.yaml (default, unchanged) or a data.yaml for the
# image_data_with_synth/ tree. This script does not merge/consolidate the
# two datasets — pick one data.yaml per training run.
#
# image_data_with_synth/ is NOT present in this workspace checkout (it
# lives on a separate device per the project's documented setup). If you
# pass --data pointing into image_data_with_synth/, check on your device
# for this folder and the exact data.yaml path needed — this script does
# not assume, generate, or validate that path beyond the basic existence
# check below.
IMAGE_DATA_WITH_SYNTH_MARKER = "image_data_with_synth"


def _describe_dataset_source(data_yaml_path: str) -> str:
    """
    Return a short, human-readable label for which dataset source a
    --data path points at, per Requirement 15.6 ("clearly indicate...
    whether a given --data/--source path points at corrected or raw
    image_data_with_synth/ labels").
    """
    normalized = data_yaml_path.replace("\\", "/").lower()
    if IMAGE_DATA_WITH_SYNTH_MARKER in normalized:
        if "reclassified" in normalized or "corrected" in normalized:
            return "image_data_with_synth/ (corrected)"
        return "image_data_with_synth/ (raw)"
    return "roboflow data/"


def main():
    from hazard_detection.data_pipeline.training_pipeline import (
        TrainingConfig, YOLOTrainingPipeline
    )

    parser = argparse.ArgumentParser(description="Train YOLOv12 hazard detection model.")
    parser.add_argument("--epochs",   type=int,   default=150) #was 100
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=0.0007) #was 0.001
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--device",   type=str,   default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--data",     type=str,   default="roboflow data/data.yaml",
                        help="Path to data.yaml. Defaults to 'roboflow data/data.yaml'. "
                             "To train against image_data_with_synth/ instead, pass its "
                             "data.yaml path directly, e.g. "
                             "'image_data_with_synth/data.yaml' -- check on your device "
                             "for this folder and the exact path needed, since "
                             "image_data_with_synth/ is not present in every checkout.")
    parser.add_argument("--name",     type=str,   default="hazard_yolo",
                        help="Run name (output goes to runs/train/<name>/)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from runs/train/<name>/weights/last.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

    dataset_source = _describe_dataset_source(args.data)
    print(f"Dataset source: {dataset_source}  (--data {args.data})")
    if not Path(args.data).exists():
        print(
            f"  [warn] '{args.data}' was not found on this machine. If this is "
            f"meant to point into image_data_with_synth/, check on your device "
            f"for this folder and the path needed -- it may live on a separate "
            f"device per the project's documented setup."
        )

    resume_path = None
    if args.resume:
        resume_path = f"runs/train/{args.name}/weights/last.pt"
        print(f"Resuming from: {resume_path}")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        image_resolution=args.imgsz,
        checkpoint_interval=args.checkpoint_interval,
        data_yaml=args.data,
        output_dir=str(Path("runs/train").resolve()),
        project_name=args.name,
        device=args.device,
        resume_checkpoint=resume_path,
    )

    pipeline = YOLOTrainingPipeline(config)
    pipeline.train(data_yaml=args.data)

    output_dir = str(Path("runs/train").resolve())
    print()
    print("Training complete.")
    print(f"Best checkpoint: {output_dir}\\{args.name}\\weights\\best.pt")
    print()
    print("To evaluate:")
    print(f'  python scripts/evaluate_yolo.py --checkpoint "{output_dir}\\{args.name}\\weights\\best.pt"')
    print()
    print("To update config/hazard_detection.yaml:")
    print(f'  checkpoint_path: "{output_dir}\\{args.name}\\weights\\best.pt"')


if __name__ == '__main__':
    main()
