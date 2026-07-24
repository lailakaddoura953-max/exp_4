"""
Inference Test — Run YOLO inference on a random dataset image and display results.

Randomly selects an image from image_data_with_synth/ (or roboflow data/ fallback),
runs the InferenceEngine, draws annotated bounding boxes (red=hazard, green=safe),
displays in an OpenCV window, prints text results to terminal, and optionally saves.

Usage:
    python scripts/inference_test.py
    python scripts/inference_test.py --save
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, "src")

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run inference on a random dataset image.")
    parser.add_argument("--save", action="store_true", help="Save annotated image to runs/inference_test/")
    args = parser.parse_args()

    from dashboard.frame_source import FrameSourceManager, load_map_config
    from dashboard.checkpoint_resolver import CheckpointResolver
    from dashboard.inference_engine import InferenceEngine
    from dashboard.models import InferenceEngineConfig

    workspace_root = Path(__file__).resolve().parent.parent
    synth_dir = workspace_root / "image_data_with_synth"
    fallback_dir = workspace_root / "roboflow data"
    map_config_path = workspace_root / "config" / "dashboard_map.json"
    map_config = load_map_config(map_config_path)

    # Get a random frame
    frame_source = FrameSourceManager(
        synth_dir=synth_dir,
        fallback_dir=fallback_dir,
        map_config=map_config,
    )

    frame_info = frame_source.get_random_frame()
    if frame_info is None:
        print("ERROR: No images found in image_data_with_synth/ or roboflow data/.")
        print("Check that at least one dataset folder exists with images.")
        return 1

    print(f"\nSelected image: {frame_info.source_path}")
    print(f"  Folder: {frame_info.folder_name}")
    print(f"  Location: #{frame_info.map_location}")
    print(f"  Bucket: {frame_info.bucket}")
    print(f"  Synthetic: {frame_info.is_synthetic}")
    print()

    # Resolve checkpoint
    config_checkpoint = None
    try:
        import yaml
        hd_config_path = workspace_root / "config" / "hazard_detection.yaml"
        if hd_config_path.is_file():
            with open(hd_config_path, "r") as f:
                hd_config = yaml.safe_load(f.read())
            config_checkpoint = (hd_config or {}).get("yolo", {}).get("checkpoint_path")
    except Exception:
        pass

    resolver = CheckpointResolver(
        config_path=config_checkpoint,
        discovery_pattern=str(workspace_root / "runs" / "train" / "*" / "weights" / "best.pt"),
    )
    checkpoint = resolver.resolve()

    # Fallback to workspace root .pt files
    if checkpoint is None:
        fallback_pts = sorted(workspace_root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if fallback_pts:
            checkpoint = fallback_pts[0]

    if checkpoint is None:
        print("ERROR: No YOLO checkpoint found. Train a model first or place a .pt file in the project root.")
        return 1

    print(f"Checkpoint: {checkpoint} (source: {resolver.source})")
    print()

    # Run inference
    try:
        config = InferenceEngineConfig(
            checkpoint_path=str(checkpoint),
            device="cpu",
            confidence_threshold=0.3,
        )
        engine = InferenceEngine(config)
    except Exception as e:
        print(f"ERROR: Failed to load InferenceEngine: {e}")
        return 1

    results = engine.run(
        frame_info.image,
        camera_id=f"location_{frame_info.map_location}",
        folder_name=frame_info.folder_name,
    )

    # Print text results
    print(f"{'='*60}")
    print(f" INFERENCE RESULTS ({len(results)} detections)")
    print(f"{'='*60}")
    hazard_count = 0
    for i, r in enumerate(results, 1):
        status = "HAZARD" if r.is_hazard else "OK"
        if r.is_hazard:
            hazard_count += 1
        print(f"  {i}. [{status:6s}] {r.class_label:30s} conf={r.confidence:.2f}  {r.hazard_reason}")
    print(f"{'='*60}")
    print(f" Total: {len(results)} detections, {hazard_count} hazards")
    print(f"{'='*60}")
    print()

    # Draw bounding boxes on image
    annotated = frame_info.image.copy()
    h, w = annotated.shape[:2]

    for r in results:
        bbox = r.bbox
        x1 = int((bbox.x_center - bbox.width / 2) * w)
        y1 = int((bbox.y_center - bbox.height / 2) * h)
        x2 = int((bbox.x_center + bbox.width / 2) * w)
        y2 = int((bbox.y_center + bbox.height / 2) * h)

        color = (0, 0, 255) if r.is_hazard else (0, 255, 0)  # red=hazard, green=safe
        thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{r.class_label} {r.confidence:.2f}"
        if r.is_hazard:
            label += f" [{r.hazard_reason}]"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 6), (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save if requested
    if args.save:
        save_dir = workspace_root / "runs" / "inference_test"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"result_{frame_info.folder_name}_{Path(frame_info.source_path).stem}.png"
        cv2.imwrite(str(save_path), annotated)
        print(f"Saved annotated image to: {save_path}")

    # Display in OpenCV window
    window_name = f"Inference Test — {frame_info.folder_name} (press any key to close)"
    cv2.imshow(window_name, annotated)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
