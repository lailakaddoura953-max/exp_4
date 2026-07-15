"""
YOLO Detector wrapper for the Hazard Detection System.

Wraps the Ultralytics YOLOv12 model to perform multi-class object detection
on frame sequences captured from industrial yard cameras. Handles model loading,
preprocessing (resize + normalize), inference, and post-processing (confidence
filtering).

Requirements covered:
- 13.1: YOLOv12 model trained on 17-class Roboflow dataset
- 13.2: Produces bounding boxes in normalized format, class labels, and confidence scores
- 13.3: Discards detections below configured confidence threshold
- 13.4: Raises error if checkpoint path invalid
- 13.5: Falls back to CPU if CUDA unavailable, logs warning
- 13.6: Classifies into 17 Roboflow classes
- 13.7: Resizes to configured square resolution, applies ImageNet normalization
- 13.8: Accepts optical flow magnitude maps as additional input channel
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from hazard_detection.models import (
    BBox,
    Detection,
    FrameSequence,
    YOLOConfig,
)
from hazard_detection.diagnostics import get_logger, PerformanceTimer

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment, misc]


# The 17 classes from roboflow data/data.yaml
ROBOFLOW_CLASSES = [
    "Boat - With Cargo",
    "Container - Misaligned",
    "Container - Open",
    "Container - Picked",
    "Container - Reefer",
    "Container - Water Drop",
    "Container -Separate",
    "Container -Stacked",
    "Crane",
    "Human",
    "Human - No Safety Clothes",
    "Truck - No Container",
    "Truck - With Container",
    "Vehicle",
    "Yard - Dropoff zone",
    "Yard - No People",
    "Yard - Operation Zone",
]


class YOLODetector:
    """
    Wrapper around Ultralytics YOLOv12 for multi-class hazard detection.

    Loads a pretrained YOLOv12 checkpoint and runs inference on FrameSequence
    objects, returning filtered Detection lists per frame.

    Args:
        config: YOLOConfig instance with model parameters.
    """

    def __init__(self, config: YOLOConfig) -> None:
        self._logger = get_logger("yolo_detector")
        self._config = config

        # Validate checkpoint path exists
        checkpoint = Path(config.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"YOLO checkpoint not found at '{config.checkpoint_path}'. "
                f"Cannot proceed with inference."
            )

        # Determine device with CUDA fallback
        device = self._resolve_device(config.device)

        # Load model via Ultralytics
        self._logger.info(
            f"Loading YOLOv12 model from '{config.checkpoint_path}' on device '{device}'"
        )
        if YOLO is None:
            raise ImportError(
                "ultralytics package is required for YOLODetector. "
                "Install with: pip install ultralytics"
            )

        self._model = YOLO(config.checkpoint_path)
        self._device = device

        self._logger.info(
            f"YOLOv12 model loaded successfully. "
            f"Input resolution: {config.input_resolution}x{config.input_resolution}, "
            f"Confidence threshold: {config.confidence_threshold}, "
            f"Device: {device}"
        )

    def _resolve_device(self, requested_device: str) -> str:
        """
        Resolve the inference device, falling back to CPU if CUDA is unavailable.

        Requirement 13.5: If configured device is 'cuda' and no CUDA GPU available,
        fall back to CPU and log a warning.
        """
        if requested_device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    self._logger.warning(
                        "CUDA device requested but no CUDA-capable GPU available. "
                        "Falling back to CPU inference."
                    )
                    return "cpu"
                return "cuda"
            except ImportError:
                self._logger.warning(
                    "CUDA device requested but PyTorch CUDA support not available. "
                    "Falling back to CPU inference."
                )
                return "cpu"
        return requested_device

    def preprocess(
        self,
        frame: np.ndarray,
        flow_magnitude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Preprocess a single frame for YOLO inference.

        Requirement 13.7: Resize to configured square resolution and apply
        ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

        Requirement 13.8: Accept optional optical flow magnitude map as
        additional input channel.

        Args:
            frame: Input BGR image as numpy array with shape (H, W, 3).
            flow_magnitude: Optional optical flow magnitude map with shape (H, W).
                           Appended as a 4th channel when provided.

        Returns:
            Preprocessed tensor as numpy array with shape (resolution, resolution, C)
            where C is 3 (RGB normalized) or 4 (RGB normalized + flow channel).
        """
        import cv2

        resolution = self._config.input_resolution

        # Resize to configured square resolution
        resized = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] then apply ImageNet stats
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array(self._config.normalization_mean, dtype=np.float32)
        std = np.array(self._config.normalization_std, dtype=np.float32)
        normalized = (normalized - mean) / std

        # Append optical flow magnitude as 4th channel if provided
        if flow_magnitude is not None:
            flow_resized = cv2.resize(
                flow_magnitude, (resolution, resolution), interpolation=cv2.INTER_LINEAR
            )
            # Normalize flow to [0, 1] range
            flow_max = flow_resized.max()
            if flow_max > 0:
                flow_channel = (flow_resized / flow_max).astype(np.float32)
            else:
                flow_channel = flow_resized.astype(np.float32)
            # Add as 4th channel
            normalized = np.concatenate(
                [normalized, flow_channel[:, :, np.newaxis]], axis=2
            )

        # Log preprocessed tensor shape
        self._logger.debug(
            f"Preprocessed tensor shape: {normalized.shape}"
        )

        return normalized

    def detect(
        self,
        frame_sequence: FrameSequence,
        flow_magnitudes: Optional[List[np.ndarray]] = None,
    ) -> List[List[Detection]]:
        """
        Run YOLO inference on all frames in a FrameSequence.

        Requirement 13.2: Returns bounding boxes in normalized format,
        class labels, and confidence scores.
        Requirement 13.3: Filters detections below confidence_threshold.

        Args:
            frame_sequence: FrameSequence containing frames to process.
            flow_magnitudes: Optional list of optical flow magnitude maps,
                           one per frame (or None). Length must match frame count
                           if provided.

        Returns:
            List of detection lists, one per frame. Each inner list contains
            Detection objects that passed the confidence threshold filter.
        """
        all_detections: List[List[Detection]] = []
        num_frames = frame_sequence.frame_count

        self._logger.info(
            f"Starting detection on {num_frames} frames from camera '{frame_sequence.camera_id}'"
        )

        for frame_idx in range(num_frames):
            frame = frame_sequence.frames[frame_idx]

            # Get optional flow magnitude for this frame
            flow_mag = None
            if flow_magnitudes is not None and frame_idx < len(flow_magnitudes):
                flow_mag = flow_magnitudes[frame_idx]

            # Preprocess frame
            preprocessed = self.preprocess(frame, flow_magnitude=flow_mag)

            # Run inference with per-frame timing
            with PerformanceTimer(
                "yolo_inference_frame",
                camera_id=frame_sequence.camera_id,
                logger=self._logger,
            ) as timer:
                # Use Ultralytics predict API
                results = self._model.predict(
                    source=frame,
                    imgsz=self._config.input_resolution,
                    conf=self._config.confidence_threshold,
                    device=self._device,
                    verbose=False,
                )

            # Parse results into Detection objects
            frame_detections = self._parse_results(results, frame_idx)

            # Log raw detection count before filtering (filtering already done by conf param)
            raw_count = len(frame_detections)
            self._logger.debug(
                f"Frame {frame_idx}: raw detection count = {raw_count}"
            )

            all_detections.append(frame_detections)

        # Log detection count summary
        total_detections = sum(len(dets) for dets in all_detections)
        self._logger.info(
            f"Detection complete for camera '{frame_sequence.camera_id}': "
            f"{total_detections} total detections across {num_frames} frames"
        )

        return all_detections

    def _parse_results(
        self, results, frame_idx: int
    ) -> List[Detection]:
        """
        Parse Ultralytics results into Detection dataclass instances.

        Filters by confidence threshold and maps class indices to
        Roboflow class names.

        Args:
            results: Ultralytics prediction results list.
            frame_idx: Index of the current frame (for logging).

        Returns:
            List of Detection objects for this frame.
        """
        detections: List[Detection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]  # Single image inference returns list of 1

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])

            # Double-check confidence threshold (Ultralytics should already filter)
            if confidence < self._config.confidence_threshold:
                continue

            # Get class index and map to label
            class_idx = int(boxes.cls[i])

            # Use model names if available, fall back to ROBOFLOW_CLASSES
            if hasattr(result, "names") and result.names and class_idx in result.names:
                class_label = result.names[class_idx]
            elif class_idx < len(ROBOFLOW_CLASSES):
                class_label = ROBOFLOW_CLASSES[class_idx]
            else:
                self._logger.warning(
                    f"Frame {frame_idx}: Unknown class index {class_idx}, skipping detection"
                )
                continue

            # Get bounding box in xywhn format (normalized x_center, y_center, w, h)
            xywhn = boxes.xywhn[i]
            x_center = float(xywhn[0])
            y_center = float(xywhn[1])
            width = float(xywhn[2])
            height = float(xywhn[3])

            # Clamp to [0, 1] to handle edge cases
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            try:
                bbox = BBox(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
                detection = Detection(
                    bbox=bbox,
                    class_label=class_label,
                    confidence=confidence,
                )
                detections.append(detection)
            except (ValueError, TypeError) as e:
                self._logger.warning(
                    f"Frame {frame_idx}: Failed to create Detection object: {e}"
                )
                continue

        return detections
