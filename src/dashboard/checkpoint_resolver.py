"""
Checkpoint Resolver for the Yard Hazard Inference Dashboard v2.

Resolves which YOLO checkpoint to load at dashboard startup.

Priority:
  1. config/hazard_detection.yaml → yolo.checkpoint_path (if set and file exists)
  2. Most recent best.pt under runs/train/*/weights/ (by mtime)
  3. None (model_loaded=False; inference returns HTTP 500)

Requirements covered: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_PATTERN = "runs/train/*/weights/best.pt"


class CheckpointResolver:
    """
    Resolves the best available YOLO checkpoint path at startup.

    Args:
        config_path: Explicit checkpoint path from hazard_detection.yaml
            (yolo.checkpoint_path). Takes precedence if the file exists.
        discovery_pattern: Glob pattern for auto-discovery of best.pt files.
            Default: "runs/train/*/weights/best.pt"
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        discovery_pattern: str = DEFAULT_DISCOVERY_PATTERN,
    ) -> None:
        self._config_path = config_path
        self._discovery_pattern = discovery_pattern
        self._resolved: Optional[Path] = None
        self._source: str = "none"

        self._resolve()

    def _resolve(self) -> None:
        # Priority 1: explicit config path (if file exists)
        if self._config_path:
            candidate = Path(self._config_path)
            if candidate.is_file():
                self._resolved = candidate
                self._source = "config"
                logger.info(
                    "CheckpointResolver: using config-specified checkpoint: %s",
                    candidate,
                )
                return
            else:
                logger.info(
                    "CheckpointResolver: config checkpoint_path '%s' does not exist; "
                    "attempting auto-discovery.",
                    self._config_path,
                )

        # Priority 2: auto-discover most recent best.pt
        candidates = glob.glob(self._discovery_pattern)
        if candidates:
            # Sort by modification time, most recent first
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            self._resolved = Path(candidates[0])
            self._source = "auto-discovered"
            logger.info(
                "CheckpointResolver: auto-discovered checkpoint: %s (mtime newest of %d candidates)",
                self._resolved,
                len(candidates),
            )
            return

        # Priority 3: nothing found
        self._resolved = None
        self._source = "none"
        logger.warning(
            "CheckpointResolver: no checkpoint found via config or auto-discovery "
            "(pattern: '%s'). model_loaded will be False.",
            self._discovery_pattern,
        )

    def resolve(self) -> Optional[Path]:
        """Return the resolved checkpoint path, or None if nothing was found."""
        return self._resolved

    @property
    def source(self) -> str:
        """How the checkpoint was resolved: 'config', 'auto-discovered', or 'none'."""
        return self._source
