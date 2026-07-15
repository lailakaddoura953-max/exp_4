"""
Inference Engine for the Yard Hazard Inference Dashboard.

Wraps the existing ``YOLODetector`` and the deterministic ``RuleEngine``
(``dashboard.rules``) to provide a single-method interface for per-image
hazard classification.

Requirements covered:
- 1.1: Accept decoded image + camera_id; return List[HazardResult]
- 1.2: Return [] when no detections exceed confidence threshold
- 1.3: Attach camera_id to every HazardResult
- 1.4: Catch all YOLODetector exceptions; log + return []
- 1.5: Reuse YOLODetector from src/hazard_detection/yolo_detector.py
- 1.6: Raise ValueError at construction if config is invalid
- 16.2: Clamp detection confidence to [0.0, 1.0] before classify_all
- 16.5: One HazardResult per filtered detection (one-to-one)
- 17.1: Configurable via InferenceEngineConfig
- 17.3: ValueError when confidence_threshold out-of-range or checkpoint empty
- 17.4: CUDA fallback — log own warning for traceability
"""

import logging
import time
from typing import List

import numpy as np

from hazard_detection.models import FrameSequence, YOLOConfig
from hazard_detection.yolo_detector import YOLODetector
from dashboard.models import HazardResult, InferenceEngineConfig
from dashboard.rules import classify_all

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Single-frame hazard inference engine.

    Accepts a raw image (NumPy BGR array) and a camera identifier, runs
    YOLO detection, applies the priority-ordered rule set from
    ``dashboard.rules``, and returns one ``HazardResult`` per detection
    that meets or exceeds the configured confidence threshold.

    Args:
        config: ``InferenceEngineConfig`` specifying checkpoint path,
                confidence threshold, device, and rule thresholds.

    Raises:
        ValueError: If ``config.confidence_threshold`` is outside [0.0, 1.0]
                    or ``config.checkpoint_path`` is empty/None.
                    (Propagated from ``InferenceEngineConfig.__post_init__``.)
        FileNotFoundError: If the checkpoint file does not exist on disk.

    Requirements: 1.5, 1.6, 17.1, 17.3, 17.4
    """

    def __init__(self, config: InferenceEngineConfig) -> None:
        # Validation is enforced by InferenceEngineConfig.__post_init__; any
        # ValueError raised there propagates transparently to the caller
        # (Requirement 1.6, 17.3).
        self._config = config

        # Log startup information at INFO level (no sensitive checkpoint path
        # content — only metadata; Requirement 17.5 guidance).
        logger.info(
            "InferenceEngine initialising — "
            "checkpoint_path=%r, device=%r, confidence_threshold=%s",
            config.checkpoint_path,
            config.device,
            config.confidence_threshold,
        )

        # Build a YOLOConfig from InferenceEngineConfig fields.
        # YOLODetector._resolve_device handles the CUDA→CPU fallback internally;
        # we also emit our own warning here for dashboard-layer traceability
        # (Requirement 17.4).
        if config.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        "InferenceEngine: device='cuda' requested but no CUDA GPU "
                        "is available. Falling back to CPU inference."
                    )
            except ImportError:
                logger.warning(
                    "InferenceEngine: device='cuda' requested but PyTorch CUDA "
                    "support is not installed. Falling back to CPU inference."
                )

        yolo_config = YOLOConfig(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
        )

        self._detector = YOLODetector(yolo_config)

        logger.info(
            "InferenceEngine ready — detector loaded on device=%r",
            self._detector._device,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray, camera_id: str) -> List[HazardResult]:
        """
        Run single-frame hazard inference on one image.

        Steps
        -----
        1. Wrap ``image`` in a single-frame ``FrameSequence``.
        2. Call ``YOLODetector.detect()``; index ``results[0]`` for this frame.
        3. Filter detections whose confidence is below
           ``config.confidence_threshold``.
        4. Clamp each remaining detection's confidence to [0.0, 1.0].
        5. Pass filtered detections to ``classify_all()`` from ``dashboard.rules``.
        6. Return the resulting ``List[HazardResult]``.

        Any exception raised by ``YOLODetector.detect()`` is caught, logged
        with ``camera_id`` and ``image.shape``, and causes the method to
        return ``[]`` rather than propagating to the caller (Requirement 1.4).

        Args:
            image:     Decoded BGR image as a NumPy array (H × W × 3).
            camera_id: Camera identifier string; attached to every result.

        Returns:
            List of ``HazardResult`` records — one per detection at or above
            the confidence threshold, in the order returned by the detector.
            Returns ``[]`` on detector failure.

        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 16.2, 16.5
        """
        # Step 1 — wrap in a single-frame FrameSequence
        frame_sequence = FrameSequence(
            frames=[image],
            camera_id=camera_id,
            timestamps=[time.time()],
        )

        # Step 2 — invoke YOLODetector; catch ALL exceptions (Requirement 1.4)
        try:
            raw_results = self._detector.detect(frame_sequence)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "InferenceEngine: YOLODetector.detect() failed for "
                "camera_id=%r, image.shape=%s — %s: %s",
                camera_id,
                image.shape,
                type(exc).__name__,
                exc,
            )
            return []

        # Step 3 — extract single-frame detections (outer list indexed by frame)
        frame_detections = raw_results[0] if raw_results else []

        # Step 4 — filter below confidence threshold and clamp to [0.0, 1.0]
        threshold = self._config.confidence_threshold
        filtered = []
        for det in frame_detections:
            if det.confidence < threshold:
                continue
            # Clamp confidence in-place: Detection is a dataclass without
            # frozen=True, so mutation is safe here (Requirement 16.2).
            det.confidence = max(0.0, min(1.0, det.confidence))
            filtered.append(det)

        # Step 5 — apply rule engine
        results = classify_all(filtered, self._config, camera_id)

        # Step 6 — return results (camera_id already attached by classify_all)
        return results
