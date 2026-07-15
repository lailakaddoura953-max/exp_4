"""Step 8c - Run inference on the Roboflow test images (no live cameras needed).

Saves annotated images with bounding boxes to runs/detect/test_run/.

Usage:
    python scripts/run_on_test_images.py
    python scripts/run_on_test_images.py --checkpoint checkpoints/yolov12_best.pt
    python scripts/run_on_test_images.py --conf 0.3 --device cpu
"""
import sys
import argparse

sys.path.insert(0, 'src')

def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Run YOLO inference on Roboflow test images.")
    parser.add_argument("--checkpoint", type=str,
                        default="runs/train/hazard_yolo/weights/best.pt")
    parser.add_argument("--source",     type=str,
                        default="roboflow data/test/images")
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--name",       type=str,   default="test_run")
    args = parser.parse_args()

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
