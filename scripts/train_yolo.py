"""Step 5 - Train YOLOv12 on the Roboflow dataset.

Usage:
    python scripts/train_yolo.py                      # GPU, 100 epochs
    python scripts/train_yolo.py --device cpu         # CPU only
    python scripts/train_yolo.py --epochs 50          # fewer epochs
    python scripts/train_yolo.py --resume             # resume from last.pt
    python scripts/train_yolo.py --batch 8            # smaller batch (low VRAM)

The best checkpoint is saved to:
    runs/train/hazard_yolo/weights/best.pt

Update config/hazard_detection.yaml to point at it after training:
    yolo:
      checkpoint_path: "runs/train/hazard_yolo/weights/best.pt"
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

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
                        help="Path to data.yaml")
    parser.add_argument("--name",     type=str,   default="hazard_yolo",
                        help="Run name (output goes to runs/train/<name>/)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from runs/train/<name>/weights/last.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

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
