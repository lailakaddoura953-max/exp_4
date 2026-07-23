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
    python scripts/evaluate_yolo.py --data "image_data_with_synth/data.yaml"
                                            # evaluate against image_data_with_synth/
                                            # in isolation from roboflow data/ (see --data below)
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

# requirements.md Requirement 15.2, 15.6: --data can point at either
# roboflow data/data.yaml (default, unchanged) or a data.yaml for the
# image_data_with_synth/ tree, so a checkpoint can be evaluated against
# that data source specifically.
#
# image_data_with_synth/ is NOT present in this workspace checkout (it
# lives on a separate device per the project's documented setup). If you
# pass --data pointing into image_data_with_synth/, check on your device
# for this folder and the exact data.yaml path needed.
IMAGE_DATA_WITH_SYNTH_MARKER = "image_data_with_synth"


def _describe_dataset_source(data_yaml_path: str) -> str:
    """Same dataset-source labeling convention as train_yolo.py."""
    normalized = data_yaml_path.replace("\\", "/").lower()
    if IMAGE_DATA_WITH_SYNTH_MARKER in normalized:
        if "reclassified" in normalized or "corrected" in normalized:
            return "image_data_with_synth/ (corrected)"
        return "image_data_with_synth/ (raw)"
    return "roboflow data/"


def main():
    from hazard_detection.evaluation import ModelEvaluator

    parser = argparse.ArgumentParser(description="Evaluate a YOLO hazard detection model.")
    parser.add_argument("--checkpoint",  type=str,
                        default="runs/train/hazard_yolo/weights/best.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--data",        type=str,
                        default="roboflow data/data.yaml",
                        help="Path to data.yaml. Defaults to 'roboflow data/data.yaml'. "
                             "To evaluate against image_data_with_synth/ instead, pass its "
                             "data.yaml path directly -- check on your device for this "
                             "folder and the exact path needed, since image_data_with_synth/ "
                             "is not present in every checkout.")
    parser.add_argument("--output_dir",  type=str,
                        default="evaluation_results",
                        help="Where to save charts and summary")
    parser.add_argument("--device",      type=str,  default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--conf",        type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
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
