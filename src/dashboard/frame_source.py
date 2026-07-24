"""
Frame Source Manager for the Yard Hazard Inference Dashboard v2.

Provides images for dashboard inference by auto-cycling through
image_data_with_synth/ on a configurable interval. Falls back to
roboflow data/ when the primary source isn't available.

Designed to be modular and swappable: replacing this with a real
RTSP camera adapter later requires changing only this module, not
the inference engine, rule engine, or UI.

Requirements covered: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# CHANGE LATER WHEN SUPERVISOR REVIEWS — hourly cycling is a placeholder;
# reduce interval for real-time demonstration once live cameras are integrated.
DEFAULT_CYCLE_INTERVAL_SECONDS = 3600  # 1 hour
DEMO_CYCLE_INTERVAL_SECONDS = 600     # 10 minutes

IMAGE_EXTENSIONS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")

SYNTH_BUCKETS = [
    "augmented_hazards",
    "normal_operations/augmented_normal",
    "normal_operations/auto_accepted",
]

_DISCLAIMER_TEXT = (
    "This image has been synthetically injected with hazards or augmented. "
    "It is not live footage."
)


@dataclass
class FrameInfo:
    """Metadata about a single frame served by FrameSourceManager."""

    image: np.ndarray       # decoded BGR frame
    source_path: Path       # absolute path to the image file
    folder_name: str        # parent location folder (e.g. 'berth_401')
    map_location: int       # resolved map pin number (1-16)
    bucket: str             # 'augmented_hazards' / 'augmented_normal' / 'auto_accepted'
    is_synthetic: bool      # True when from image_data_with_synth/


def load_map_config(config_path: Path) -> Dict:
    """
    Load config/dashboard_map.json. Returns the parsed dict, or a minimal
    default if the file is missing/unreadable.
    """
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                "Failed to load dashboard_map.json (%s); using default mapping.", e
            )
    else:
        logger.warning(
            "config/dashboard_map.json not found at %s; using default mapping.",
            config_path,
        )

    # Hard-coded fallback with the 6 confirmed folders
    return {
        "folder_to_location": {
            "berth_401": 10,
            "berth_405_and_knuckle": 16,
            "E2_C_PTZ_Pedestal": 8,
            "E10_East_PTZ_Wall_st": 8,
            "TEL_1-35": 9,
            "TEL_144": 9,
        },
        "default_location_range": [1, 16],
    }


def _resolve_location(folder_name: str, folder_to_location: Dict[str, int], loc_range: List[int]) -> int:
    """
    Resolve a folder name to a map location number.
    Uses the explicit mapping if present; otherwise deterministic hash.
    """
    if folder_name in folder_to_location:
        return folder_to_location[folder_name]
    # Deterministic random: same folder always gets the same pin
    lo, hi = loc_range[0], loc_range[1]
    return (hash(folder_name) % (hi - lo + 1)) + lo


def _discover_synth_pairs(synth_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Discover (image_path, folder_name, bucket) triples across all three
    buckets under synth_dir. Recursive, deduplicates by resolved path.
    """
    seen: set = set()
    pairs: List[Tuple[Path, str, str]] = []

    for bucket_rel in SYNTH_BUCKETS:
        bucket_dir = synth_dir / bucket_rel
        if not bucket_dir.exists():
            continue

        # Derive short bucket name for FrameInfo
        bucket_name = bucket_rel.split("/")[-1] if "/" in bucket_rel else bucket_rel

        for ext in IMAGE_EXTENSIONS:
            for img_path in bucket_dir.rglob(f"*{ext}"):
                resolved = img_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)

                # Determine folder_name: first directory component under
                # the bucket dir (the location folder).
                try:
                    rel = img_path.relative_to(bucket_dir)
                    folder_name = rel.parts[0] if len(rel.parts) > 1 else ""
                except (ValueError, IndexError):
                    folder_name = ""

                pairs.append((img_path, folder_name, bucket_name))

    return sorted(pairs, key=lambda t: str(t[0]))


def _discover_fallback_images(fallback_dir: Path) -> List[Path]:
    """Discover JPEG/PNG images under roboflow data/test + train."""
    images: List[Path] = []
    for subdir in ["test/images", "train/images"]:
        d = fallback_dir / subdir
        if not d.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            images.extend(d.glob(f"*{ext}"))
    return sorted(set(images))


class FrameSourceManager:
    """
    Provides images for dashboard inference. Cycles through
    image_data_with_synth/ automatically on a configurable interval.

    # CHANGE LATER WHEN SUPERVISOR REVIEWS — hourly cycling is a placeholder;
    # reduce interval for real-time demonstration once live cameras are integrated.
    """

    def __init__(
        self,
        synth_dir: Path,
        fallback_dir: Path,
        map_config: Dict,
        cycle_interval_seconds: int = DEFAULT_CYCLE_INTERVAL_SECONDS,
        seed: int = 42,
    ) -> None:
        self._synth_dir = synth_dir
        self._fallback_dir = fallback_dir
        self._cycle_interval = cycle_interval_seconds
        self._seed = seed

        self._folder_to_location: Dict[str, int] = map_config.get("folder_to_location", {})
        self._loc_range: List[int] = map_config.get("default_location_range", [1, 16])

        # Discover frames
        self._using_synth = False
        self._synth_pairs: List[Tuple[Path, str, str]] = []
        self._fallback_images: List[Path] = []

        if synth_dir.exists():
            self._synth_pairs = _discover_synth_pairs(synth_dir)
            if self._synth_pairs:
                self._using_synth = True
                logger.info(
                    "FrameSourceManager: using image_data_with_synth/ (%d images across %s)",
                    len(self._synth_pairs),
                    SYNTH_BUCKETS,
                )
            else:
                logger.warning(
                    "FrameSourceManager: image_data_with_synth/ exists but contains "
                    "no image files; falling back to roboflow data/."
                )
        else:
            logger.warning(
                "FrameSourceManager: '%s' not found; falling back to roboflow data/. "
                "image_data_with_synth/ may live on a separate device.",
                synth_dir,
            )

        if not self._using_synth:
            self._fallback_images = _discover_fallback_images(fallback_dir)
            if self._fallback_images:
                logger.info(
                    "FrameSourceManager: fallback to roboflow data/ (%d images).",
                    len(self._fallback_images),
                )
            else:
                logger.warning(
                    "FrameSourceManager: no images found in fallback either. "
                    "Auto-cycle will produce nothing; manual upload still works."
                )

        # Build shuffled queue
        self._rng = random.Random(seed)
        self._queue: List[int] = list(range(self._frame_count()))
        self._rng.shuffle(self._queue)
        self._queue_index = 0

        # Cycle timing
        self._last_advance_time = time.time()
        self._current_index = self._queue[0] if self._queue else -1

    def _frame_count(self) -> int:
        if self._using_synth:
            return len(self._synth_pairs)
        return len(self._fallback_images)

    def _advance_if_due(self) -> None:
        """Advance to next frame if cycle_interval has elapsed."""
        now = time.time()
        if now - self._last_advance_time >= self._cycle_interval:
            self._queue_index += 1
            if self._queue_index >= len(self._queue):
                # Reshuffle and restart
                self._rng.shuffle(self._queue)
                self._queue_index = 0
            self._current_index = self._queue[self._queue_index] if self._queue else -1
            self._last_advance_time = now

    def get_current_frame(self) -> Optional[FrameInfo]:
        """
        Return the currently active frame (advances every cycle_interval).
        Returns None if no frames are available.
        """
        self._advance_if_due()
        if self._current_index < 0:
            return None
        return self._load_frame(self._current_index)

    def get_random_frame(self) -> Optional[FrameInfo]:
        """Return a random frame (for /api/test-image)."""
        total = self._frame_count()
        if total == 0:
            return None
        idx = self._rng.randint(0, total - 1)
        return self._load_frame(idx)

    def is_using_synth(self) -> bool:
        """True if primary source (image_data_with_synth/) is active."""
        return self._using_synth

    @property
    def source_disclaimer(self) -> str:
        """Disclaimer text when using synthetic data, empty for live/fallback."""
        if self._using_synth:
            return _DISCLAIMER_TEXT
        return ""

    def advance_now(self) -> None:
        """Force-advance to the next frame (useful for testing)."""
        self._last_advance_time = 0  # force _advance_if_due to trigger
        self._advance_if_due()

    def _load_frame(self, index: int) -> Optional[FrameInfo]:
        """Load and decode a frame by queue index."""
        if self._using_synth:
            if index >= len(self._synth_pairs):
                return None
            img_path, folder_name, bucket = self._synth_pairs[index]
            map_location = _resolve_location(folder_name, self._folder_to_location, self._loc_range)
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("FrameSourceManager: failed to decode %s", img_path)
                return None
            return FrameInfo(
                image=image,
                source_path=img_path,
                folder_name=folder_name,
                map_location=map_location,
                bucket=bucket,
                is_synthetic=True,
            )
        else:
            if index >= len(self._fallback_images):
                return None
            img_path = self._fallback_images[index]
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            # Fallback: no location folder info available
            map_location = (hash(str(img_path)) % 16) + 1
            return FrameInfo(
                image=image,
                source_path=img_path,
                folder_name="",
                map_location=map_location,
                bucket="roboflow_fallback",
                is_synthetic=False,
            )
