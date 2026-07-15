"""Step 5d - Fine-tune a pretrained checkpoint on supplemental or synthetic data.

Usage:
    python scripts/finetune_yolo.py
    python scripts/finetune_yolo.py --checkpoint checkpoints/yolov12_best.pt
    python scripts/finetune_yolo.py --data data/synthetic_output/data.yaml
    python scripts/finetune_yolo.py --epochs 30 --device cpu
"""
import sys
import argparse

sys.path.insert(0, 'src')

def main():
    from hazard_detection.data_pipeline.training_pipeline import (
        TrainingConfig, YOLOTrainingPipeline
    )

    parser = argparse.ArgumentParser(description="Fine-tune a YOLO checkpoint.")
    parser.add_argument("--checkpoint", type=str,
                        default="runs/train/hazard_yolo/weights/best.pt")
    parser.add_argument("--data",       type=str,
                        default="roboflow data/data.yaml")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=0.0001)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--imgsz",      type=int,   default=640)
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        image_resolution=args.imgsz,
        checkpoint_interval=5,
        device=args.device,
    )

    pipeline = YOLOTrainingPipeline(config)
    pipeline.fine_tune(
        pretrained_checkpoint=args.checkpoint,
        additional_data_yaml=args.data,
    )

    print()
    print("Fine-tuning complete.")
    print("Best checkpoint: runs/train/hazard_yolo_finetune/weights/best.pt")


if __name__ == '__main__':
    main()
