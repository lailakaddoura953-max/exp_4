"""
YOLO Training Pipeline for the Hazard Detection System.

Wraps the Ultralytics YOLO library to train YOLOv12 on the Roboflow
dataset, with support for checkpointing, resuming, fine-tuning, and
TensorBoard-compatible metric logging.

Requirements covered:
- 15.1: Train YOLOv12 on the Roboflow dataset with data.yaml
- 15.2: Configurable hyperparameters (epochs, batch_size, lr, resolution, augmentation)
- 15.3: Save checkpoints at configurable interval, retain best by mAP@0.5
- 15.4: Resume from previously saved checkpoint (weights, optimizer, epoch)
- 15.5: Fine-tune pretrained checkpoint with supplemental/synthetic data
- 15.6: Log training metrics to TensorBoard format
- 15.7: Exit with non-zero status for invalid hyperparameters or missing data.yaml
"""

import sys
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazard_detection.diagnostics import get_logger

logger = get_logger("training_pipeline")


# ---------------------------------------------------------------------------
# Hyperparameter bounds (Requirement 15.2 / Property 26)
# ---------------------------------------------------------------------------

EPOCHS_MIN = 1
EPOCHS_MAX = 1000

BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 64

LR_MIN = 1e-6
LR_MAX = 1e-1

RESOLUTION_MIN = 320
RESOLUTION_MAX = 1280


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the YOLO training pipeline."""

    # Core hyperparameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    image_resolution: int = 640
    checkpoint_interval: int = 5          # Save checkpoint every N epochs

    # Data
    data_yaml: str = "roboflow data/data.yaml"
    output_dir: str = "runs/train"         # Root dir for Ultralytics run outputs
    project_name: str = "hazard_yolo"      # Sub-project folder under output_dir

    # Device
    device: str = "cuda"                   # "cuda", "cpu", or device index "0"

    # Pretrained weights (for initial training or fine-tuning base)
    pretrained_weights: str = "yolo12n.pt" # Ultralytics pretrained base model

    # Resume / fine-tune
    resume_checkpoint: Optional[str] = None   # Path to checkpoint to resume from

    # Augmentation toggles (passed to Ultralytics as overrides)
    augmentation: Dict[str, bool] = field(default_factory=lambda: {
        "rotation": True,
        "flip": True,
        "brightness": True,
        "contrast": True,
        "noise": True,
    })

    def __post_init__(self) -> None:
        """Validate all hyperparameters; sys.exit(1) on any violation."""
        errors: List[str] = []

        if not (EPOCHS_MIN <= self.epochs <= EPOCHS_MAX):
            errors.append(
                f"epochs={self.epochs} is out of range [{EPOCHS_MIN}, {EPOCHS_MAX}]"
            )
        if not (BATCH_SIZE_MIN <= self.batch_size <= BATCH_SIZE_MAX):
            errors.append(
                f"batch_size={self.batch_size} is out of range "
                f"[{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]"
            )
        if not (LR_MIN <= self.learning_rate <= LR_MAX):
            errors.append(
                f"learning_rate={self.learning_rate} is out of range "
                f"[{LR_MIN}, {LR_MAX}]"
            )
        if not (RESOLUTION_MIN <= self.image_resolution <= RESOLUTION_MAX):
            errors.append(
                f"image_resolution={self.image_resolution} is out of range "
                f"[{RESOLUTION_MIN}, {RESOLUTION_MAX}]"
            )
        if self.checkpoint_interval < 1:
            errors.append(
                f"checkpoint_interval={self.checkpoint_interval} must be >= 1"
            )

        if errors:
            for msg in errors:
                logger.error(f"Invalid training hyperparameter: {msg}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: validate data.yaml
# ---------------------------------------------------------------------------


def _validate_data_yaml(data_yaml: str) -> Path:
    """
    Validate that data_yaml points to a readable YAML file.

    Returns the resolved Path on success; calls sys.exit(1) on failure.
    """
    path = Path(data_yaml)
    if not path.exists():
        msg = f"data.yaml not found at path: {path.resolve()}"
        logger.error(msg)
        sys.exit(1)
    if not path.is_file():
        msg = f"data.yaml path is not a file: {path.resolve()}"
        logger.error(msg)
        sys.exit(1)
    # Quick YAML parse check
    try:
        import yaml  # lazy import so the module stays importable without pyyaml at class-def time
        with open(path, "r", encoding="utf-8") as fh:
            contents = yaml.safe_load(fh)
        if not isinstance(contents, dict):
            msg = f"data.yaml does not contain a valid YAML mapping: {path.resolve()}"
            logger.error(msg)
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to parse data.yaml at {path.resolve()}: {exc}"
        logger.error(msg)
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# Helper: build Ultralytics augmentation kwargs
# ---------------------------------------------------------------------------


def _augmentation_kwargs(augmentation: Dict[str, bool]) -> Dict[str, Any]:
    """
    Translate augmentation toggles to Ultralytics train() keyword arguments.

    Ultralytics uses float values (0.0 = off, positive = on) for most
    augmentation parameters.
    """
    kwargs: Dict[str, Any] = {}

    # degrees: rotation range in degrees (0 disables)
    if "rotation" in augmentation:
        kwargs["degrees"] = 10.0 if augmentation["rotation"] else 0.0

    # fliplr: probability of horizontal flip
    if "flip" in augmentation:
        kwargs["fliplr"] = 0.5 if augmentation["flip"] else 0.0

    # hsv_v: brightness-related HSV value augmentation
    if "brightness" in augmentation:
        kwargs["hsv_v"] = 0.4 if augmentation["brightness"] else 0.0

    # hsv_s: saturation/contrast-related augmentation
    if "contrast" in augmentation:
        kwargs["hsv_s"] = 0.7 if augmentation["contrast"] else 0.0

    # Noise is not natively supported as a toggle in Ultralytics v8+;
    # we leave it as a no-op but log it for traceability.
    if "noise" in augmentation and augmentation["noise"]:
        logger.info(
            "Noise augmentation toggle noted; Ultralytics handles noise internally."
        )

    return kwargs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class YOLOTrainingPipeline:
    """
    Training pipeline wrapper for YOLOv12 using the Ultralytics library.

    Handles training from scratch, resuming from checkpoints, and fine-tuning
    pretrained models with supplemental or synthetic data.

    Usage:
        config = TrainingConfig(epochs=50, batch_size=8, learning_rate=5e-4)
        pipeline = YOLOTrainingPipeline(config)
        pipeline.train("roboflow data/data.yaml")

    Fine-tuning:
        pipeline.fine_tune(
            pretrained_checkpoint="checkpoints/yolov12_best.pt",
            additional_data_yaml="supplemental_output/data.yaml",
        )
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialise the training pipeline.

        The TrainingConfig __post_init__ already performs hyperparameter
        validation and calls sys.exit(1) on any invalid value, so by the
        time we reach here the config is guaranteed valid.

        Args:
            config: Validated TrainingConfig instance.
        """
        self.config = config
        self._run_dir: Optional[Path] = None  # set after training completes

        logger.info(
            "YOLOTrainingPipeline initialised",
            extra={
                "extra_data": {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "image_resolution": config.image_resolution,
                    "checkpoint_interval": config.checkpoint_interval,
                    "device": config.device,
                    "pretrained_weights": config.pretrained_weights,
                    "resume_checkpoint": config.resume_checkpoint,
                }
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, data_yaml: str) -> None:
        """
        Train YOLOv12 on the provided dataset from scratch (or resume).

        If config.resume_checkpoint is set the run resumes from that file,
        restoring weights, optimizer state, and epoch count automatically
        via the Ultralytics resume mechanism.

        Checkpoints are saved every config.checkpoint_interval epochs via
        Ultralytics' save_period parameter.  The best checkpoint (highest
        mAP@0.5) is retained automatically by Ultralytics as best.pt.

        Training metrics (loss, mAP@0.5, precision, recall) are written in
        TensorBoard format to the run directory by Ultralytics.

        Args:
            data_yaml: Path to the YOLO data.yaml configuration file.

        Raises:
            SystemExit: With code 1 on invalid data_yaml or any fatal error.
        """
        data_path = _validate_data_yaml(data_yaml)
        logger.info(f"Starting training on dataset: {data_path}")

        try:
            from ultralytics import YOLO  # deferred import
        except ImportError as exc:
            logger.error(f"Ultralytics library not available: {exc}")
            sys.exit(1)

        # Select base weights: resume path takes priority
        if self.config.resume_checkpoint:
            resume_path = Path(self.config.resume_checkpoint)
            if not resume_path.exists():
                logger.error(
                    f"Resume checkpoint not found: {resume_path.resolve()}"
                )
                sys.exit(1)
            weights = str(resume_path)
            resume = True
            logger.info(
                f"Resuming training from checkpoint: {resume_path}",
                extra={"extra_data": {"checkpoint": str(resume_path)}},
            )
        else:
            weights = self.config.pretrained_weights
            resume = False

        model = YOLO(weights)

        aug_kwargs = _augmentation_kwargs(self.config.augmentation)

        start_time = time.time()
        logger.info(
            "Training started",
            extra={
                "extra_data": {
                    "weights": weights,
                    "data_yaml": str(data_path),
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "image_resolution": self.config.image_resolution,
                    "save_period": self.config.checkpoint_interval,
                    "resume": resume,
                    "augmentation_kwargs": aug_kwargs,
                }
            },
        )

        try:
            results = model.train(
                data=str(data_path),
                epochs=self.config.epochs,
                imgsz=self.config.image_resolution,
                batch=self.config.batch_size,
                lr0=self.config.learning_rate,
                device=self.config.device,
                project=self.config.output_dir,
                name=self.config.project_name,
                save_period=self.config.checkpoint_interval,
                resume=resume,
                exist_ok=True,
                plots=True,
                workers=0,               # required on Windows to avoid multiprocessing spawn error
                **aug_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Training failed with exception: {exc}\n{traceback.format_exc()}"
            )
            sys.exit(1)

        elapsed = time.time() - start_time
        self._run_dir = Path(self.config.output_dir) / self.config.project_name
        self._log_training_completion(results, elapsed)

    def fine_tune(
        self,
        pretrained_checkpoint: str,
        additional_data_yaml: str,
    ) -> None:
        """
        Fine-tune a pretrained checkpoint with supplemental or synthetic data.

        Loads the pretrained weights and continues training on the combined
        dataset described in additional_data_yaml.  The frozen-backbone
        approach is NOT used here — the full model is unfrozen so that
        supplemental class patterns can propagate throughout all layers.

        Args:
            pretrained_checkpoint: Path to pretrained .pt weights file.
            additional_data_yaml:  Path to a YOLO data.yaml for the
                                   additional/supplemental dataset.

        Raises:
            SystemExit: With code 1 on missing checkpoint, invalid data_yaml,
                        or any fatal training error.
        """
        checkpoint_path = Path(pretrained_checkpoint)
        if not checkpoint_path.exists():
            logger.error(
                f"Pretrained checkpoint not found: {checkpoint_path.resolve()}"
            )
            sys.exit(1)

        data_path = _validate_data_yaml(additional_data_yaml)

        logger.info(
            "Starting fine-tuning",
            extra={
                "extra_data": {
                    "pretrained_checkpoint": str(checkpoint_path),
                    "additional_data_yaml": str(data_path),
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                }
            },
        )

        try:
            from ultralytics import YOLO  # deferred import
        except ImportError as exc:
            logger.error(f"Ultralytics library not available: {exc}")
            sys.exit(1)

        model = YOLO(str(checkpoint_path))

        aug_kwargs = _augmentation_kwargs(self.config.augmentation)

        # Use a lower learning rate for fine-tuning to avoid destroying pretrained features
        fine_tune_lr = min(self.config.learning_rate, 1e-4)

        start_time = time.time()

        try:
            results = model.train(
                data=str(data_path),
                epochs=self.config.epochs,
                imgsz=self.config.image_resolution,
                batch=self.config.batch_size,
                lr0=fine_tune_lr,
                device=self.config.device,
                project=self.config.output_dir,
                name=f"{self.config.project_name}_finetune",
                save_period=self.config.checkpoint_interval,
                resume=False,
                exist_ok=True,
                plots=True,
                workers=0,               # required on Windows
                **aug_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Fine-tuning failed with exception: {exc}\n{traceback.format_exc()}"
            )
            sys.exit(1)

        elapsed = time.time() - start_time
        self._run_dir = (
            Path(self.config.output_dir) / f"{self.config.project_name}_finetune"
        )
        self._log_training_completion(results, elapsed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_training_completion(self, results: Any, elapsed_seconds: float) -> None:
        """
        Log a structured summary after training completes.

        Extracts final epoch metrics from the Ultralytics results object
        and writes them to the structured logger.

        Args:
            results: Ultralytics training results object.
            elapsed_seconds: Total wall-clock time for the training run.
        """
        run_dir = self._run_dir or Path(self.config.output_dir) / self.config.project_name
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"

        # Extract final metrics if available
        metrics_summary: Dict[str, Any] = {}
        try:
            if results is not None and hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics_summary = {
                    "box_loss": rd.get("train/box_loss"),
                    "cls_loss": rd.get("train/cls_loss"),
                    "dfl_loss": rd.get("train/dfl_loss"),
                    "mAP50": rd.get("metrics/mAP50(B)"),
                    "mAP50_95": rd.get("metrics/mAP50-95(B)"),
                    "precision": rd.get("metrics/precision(B)"),
                    "recall": rd.get("metrics/recall(B)"),
                }
        except Exception:  # noqa: BLE001
            pass  # metrics extraction is best-effort

        logger.info(
            f"Training complete in {elapsed_seconds:.1f}s",
            extra={
                "extra_data": {
                    "elapsed_seconds": round(elapsed_seconds, 1),
                    "run_dir": str(run_dir),
                    "best_checkpoint": str(best_pt) if best_pt.exists() else None,
                    "last_checkpoint": str(last_pt) if last_pt.exists() else None,
                    "final_metrics": metrics_summary,
                }
            },
        )

        if best_pt.exists():
            logger.info(f"Best checkpoint (highest mAP@0.5) saved to: {best_pt}")
        else:
            logger.warning(
                f"Expected best.pt not found at {best_pt}. "
                "Check run directory for checkpoint files."
            )

    def _log_epoch_memory(self, epoch: int) -> None:
        """
        Log current GPU/CPU memory usage at the given epoch.

        This is called internally if memory monitoring is desired outside
        of the Ultralytics training loop.  The main training loop in
        Ultralytics handles its own epoch-level logging; this helper is
        available for external monitoring integrations.

        Args:
            epoch: Current epoch number (1-based).
        """
        memory_info: Dict[str, Any] = {"epoch": epoch}

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                memory_info["gpu_allocated_mb"] = round(allocated, 1)
                memory_info["gpu_reserved_mb"] = round(reserved, 1)
        except Exception:  # noqa: BLE001
            pass

        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / (1024 ** 2)
            memory_info["cpu_rss_mb"] = round(rss_mb, 1)
        except Exception:  # noqa: BLE001
            pass

        logger.info(
            f"Epoch {epoch} memory usage",
            extra={"extra_data": memory_info},
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> Optional[Path]:
        """Return the Ultralytics run directory after training, or None."""
        return self._run_dir

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Return the path to best.pt if training has completed."""
        if self._run_dir is None:
            return None
        candidate = self._run_dir / "weights" / "best.pt"
        return candidate if candidate.exists() else None

    @property
    def last_checkpoint_path(self) -> Optional[Path]:
        """Return the path to last.pt if training has completed."""
        if self._run_dir is None:
            return None
        candidate = self._run_dir / "weights" / "last.pt"
        return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    """Parse command-line arguments for the training pipeline script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv12 training pipeline for the Hazard Detection System"
    )
    parser.add_argument(
        "--data-yaml",
        default="roboflow data/data.yaml",
        help="Path to the YOLO data.yaml file (default: 'roboflow data/data.yaml')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=f"Number of training epochs [{EPOCHS_MIN}-{EPOCHS_MAX}] (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=f"Batch size [{BATCH_SIZE_MIN}-{BATCH_SIZE_MAX}] (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help=f"Initial learning rate [{LR_MIN}-{LR_MAX}] (default: 1e-3)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help=f"Input image resolution (square) [{RESOLUTION_MIN}-{RESOLUTION_MAX}] (default: 640)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device: 'cuda', 'cpu', or device index (default: cuda)",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="yolo12n.pt",
        help="Ultralytics base weights for initial training (default: yolo12n.pt)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--fine-tune-checkpoint",
        default=None,
        help="Pretrained checkpoint path for fine-tuning mode",
    )
    parser.add_argument(
        "--fine-tune-data-yaml",
        default=None,
        help="data.yaml for supplemental/synthetic data when fine-tuning",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/train",
        help="Root directory for run outputs (default: runs/train)",
    )
    parser.add_argument(
        "--project-name",
        default="hazard_yolo",
        help="Sub-project name under output-dir (default: hazard_yolo)",
    )
    # Augmentation toggles
    for aug in ["rotation", "flip", "brightness", "contrast", "noise"]:
        parser.add_argument(
            f"--no-{aug}",
            action="store_true",
            default=False,
            help=f"Disable {aug} augmentation",
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    augmentation = {
        "rotation": not args.no_rotation,
        "flip": not args.no_flip,
        "brightness": not args.no_brightness,
        "contrast": not args.no_contrast,
        "noise": not args.no_noise,
    }

    # TrainingConfig validates hyperparameters and exits(1) on violation
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_resolution=args.resolution,
        checkpoint_interval=args.checkpoint_interval,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        project_name=args.project_name,
        device=args.device,
        pretrained_weights=args.pretrained_weights,
        resume_checkpoint=args.resume_checkpoint,
        augmentation=augmentation,
    )

    pipeline = YOLOTrainingPipeline(config)

    if args.fine_tune_checkpoint:
        if not args.fine_tune_data_yaml:
            logger.error(
                "--fine-tune-data-yaml is required when --fine-tune-checkpoint is specified"
            )
            sys.exit(1)
        pipeline.fine_tune(
            pretrained_checkpoint=args.fine_tune_checkpoint,
            additional_data_yaml=args.fine_tune_data_yaml,
        )
    else:
        pipeline.train(data_yaml=args.data_yaml)
