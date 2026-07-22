"""
Auto-Annotation Entry Point — Pipeline Selector
================================================
Single command that lets you pick which auto-annotation backend to run:

    --pipeline segmentation   Grounded SAM 2 + Grounding DINO (auto_annotate.py)
                              Pixel-accurate polygon masks. Needs the
                              .venv_annotation environment set up per SETUP.md
                              (SAM 2 + Grounding DINO compiled, multi-GB
                              checkpoints downloaded).

    --pipeline cnn            Traditional CNN detector fallback
                              (cnn_auto_annotate.py). Ultralytics YOLOv12,
                              already available in the project's main .venv.
                              Produces bounding-box labels (encoded as
                              4-corner rectangle polygons) instead of masks.
                              Requires a YOLO checkpoint trained on
                              "roboflow data/data.yaml" — train one with
                              scripts/train_yolo.py if you don't have one.

Both pipelines write output in the same auto_accepted / review_queue /
rejected directory layout, so review_annotations.py and downstream training
scripts work unchanged regardless of which one you pick.

Why this exists: the segmentation pipeline is the primary target (mask
quality matters for hazard shapes like open container doors), but it has a
heavy, fragile dependency chain. If it isn't viable or maintainable on the
private machine, switch to --pipeline cnn without changing anything else in
your workflow.

Usage:
    # Use the segmentation pipeline (same as running auto_annotate.py directly)
    python scripts/annotation/run_auto_annotate.py --pipeline segmentation \\
        --input_dir image_data_normal --output_dir image_data_annotated

    # Use the CNN fallback pipeline
    python scripts/annotation/run_auto_annotate.py --pipeline cnn \\
        --input_dir image_data_normal --output_dir image_data_annotated_cnn \\
        --checkpoint runs/train/hazard_yolo/weights/best.pt

    # Verify either backend's setup without touching real data
    python scripts/annotation/run_auto_annotate.py --pipeline cnn --verify \\
        --checkpoint runs/train/hazard_yolo/weights/best.pt
    python scripts/annotation/run_auto_annotate.py --pipeline segmentation --verify
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# Make sibling modules importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_segmentation_args(args: argparse.Namespace) -> SimpleNamespace:
    """Map shared CLI args onto auto_annotate.py's expected Namespace shape."""
    return SimpleNamespace(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence=args.confidence,
        review_threshold=args.review_threshold,
        chrome_top=args.chrome_top,
        chrome_bottom=args.chrome_bottom,
        chrome_left=args.chrome_left,
        chrome_right=args.chrome_right,
        simplify=args.simplify,
        classes=args.classes,
        limit=args.limit,
        dry_run=args.dry_run,
        verify=args.verify,
    )


def build_cnn_args(args: argparse.Namespace) -> SimpleNamespace:
    """Map shared CLI args onto cnn_auto_annotate.py's expected Namespace shape."""
    return SimpleNamespace(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        confidence=args.confidence,
        review_threshold=args.review_threshold,
        imgsz=args.imgsz,
        device=args.device,
        chrome_top=args.chrome_top,
        chrome_bottom=args.chrome_bottom,
        chrome_left=args.chrome_left,
        chrome_right=args.chrome_right,
        classes=args.classes,
        limit=args.limit,
        dry_run=args.dry_run,
        verify=args.verify,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run auto-annotation with a selectable backend pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pipeline", choices=["segmentation", "cnn"], default="segmentation",
        help="'segmentation' = Grounded SAM 2 + Grounding DINO (auto_annotate.py). "
             "'cnn' = traditional CNN detector fallback (cnn_auto_annotate.py).",
    )
    # Shared args
    # image_data_normal lives on a separate, access-restricted device and may
    # not be present here. Fall back to roboflow data as a stand-in input.
    default_input_dir = "image_data_normal" if Path("image_data_normal").exists() else "roboflow data"
    p.add_argument("--input_dir", type=str, default=default_input_dir,
                   help="Root directory of images to annotate. Defaults to 'image_data_normal' "
                        "if present, otherwise falls back to 'roboflow data'.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: image_data_annotated (segmentation) or "
                        "image_data_annotated_cnn (cnn)")
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--review_threshold", type=float, default=0.55)
    p.add_argument("--chrome_top", type=int, default=60)
    p.add_argument("--chrome_bottom", type=int, default=30)
    p.add_argument("--chrome_left", type=int, default=220)
    p.add_argument("--chrome_right", type=int, default=10)
    p.add_argument("--classes", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verify", action="store_true")
    # Segmentation-only
    p.add_argument("--simplify", type=float, default=2.0,
                   help="[segmentation only] Douglas-Peucker polygon simplification epsilon")
    # CNN-only
    p.add_argument("--checkpoint", type=str, default="runs/train/hazard_yolo/weights/best.pt",
                   help="[cnn only] Path to a YOLOv12 checkpoint")
    p.add_argument("--imgsz", type=int, default=640,
                   help="[cnn only] Inference resolution")
    p.add_argument("--device", type=str, default="cuda",
                   help="[cnn only] 'cuda' or 'cpu'")

    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            "image_data_annotated" if args.pipeline == "segmentation"
            else "image_data_annotated_cnn"
        )

    print(f"=== Auto-Annotation dispatcher: pipeline='{args.pipeline}' ===")

    if args.pipeline == "segmentation":
        import auto_annotate as backend
        backend_args = build_segmentation_args(args)
        if backend_args.verify:
            backend.run_verify()
        else:
            backend.run(backend_args)
    else:
        import cnn_auto_annotate as backend
        backend_args = build_cnn_args(args)
        if backend_args.verify:
            backend.run_verify(backend_args)
        else:
            backend.run(backend_args)


if __name__ == "__main__":
    main()
