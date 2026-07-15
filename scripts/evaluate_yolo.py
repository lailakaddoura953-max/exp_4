"""Step 6 - Evaluate a trained YOLO model and generate visual diagnostic charts.

Outputs saved to evaluation_results/ (or --output_dir):
    confusion_matrix.png
    classification_metrics.png
    class_distribution.png
    roc_curves.png
    confidence_distribution.png
    evaluation_summary.json

Usage:
    python scripts/evaluate_yolo.py
    python scripts/evaluate_yolo.py --checkpoint checkpoints/yolov12_best.pt
    python scripts/evaluate_yolo.py --conf 0.3 --device cpu
"""
import sys
import argparse

sys.path.insert(0, 'src')

def main():
    from hazard_detection.evaluation import ModelEvaluator

    parser = argparse.ArgumentParser(description="Evaluate a YOLO hazard detection model.")
    parser.add_argument("--checkpoint",  type=str,
                        default="runs/train/hazard_yolo/weights/best.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--data",        type=str,
                        default="roboflow data/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--output_dir",  type=str,
                        default="evaluation_results",
                        help="Where to save charts and summary")
    parser.add_argument("--device",      type=str,  default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--conf",        type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_yaml=args.data,
        device=args.device,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
    )
    evaluator.evaluate()


if __name__ == '__main__':
    main()
