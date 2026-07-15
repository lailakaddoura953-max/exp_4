"""
Camera stub for the Yard Hazard Inference Dashboard.

Provides a configurable ``camera_id`` string and an optional test image so
that the inference pipeline can be exercised without a real camera.  Designed
so that future camera integration replaces only this module without touching
the InferenceEngine interface.

Requirements covered:
- 9.1: get_camera_id() -> str returns configurable camera identifier
- 9.2: get_test_image() -> Optional[np.ndarray] returns a decoded test image
       or None when no path is configured / file is unreadable
- 9.3: (interface concern — InferenceEngine accepts camera_id as a plain
        string; no dependency on this class at all)
- 9.4: Interface contract comment block below
- 9.5: Log warning + return None on read failure; silent None when no path set
- 15.1: get_test_image() supports directory mode (random selection)
- 15.2: get_test_image() returns None silently when no path is configured
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# REAL CAMERA INTERFACE CONTRACT
# =============================================================================
# Any module that replaces this stub MUST implement the following interface
# so that app.py and the rest of the dashboard can use it as a drop-in:
#
#   class RealCamera:
#       def get_camera_id(self) -> str:
#           """Return the unique identifier string for this camera source."""
#           ...
#
#       def get_frame(self) -> np.ndarray:
#           """
#           Capture and return the current camera frame as a decoded NumPy
#           array in BGR format (OpenCV convention), shape (H, W, 3).
#           Raises RuntimeError if the camera is disconnected or capture fails.
#           """
#           ...
#
#       def is_connected(self) -> bool:
#           """
#           Return True if the camera is reachable and ready to capture frames,
#           False otherwise.  Must not raise exceptions.
#           """
#           ...
#
# To swap in a real camera:
#   1. Implement the three methods above.
#   2. Replace CameraStub with your implementation in app.py's singleton block.
#   3. No other files need to change.
# =============================================================================

# Supported image file extensions for directory-mode random selection.
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


class CameraStub:
    """
    Stub camera source for development and demo use.

    Provides a configurable ``camera_id`` string and an optional path to a
    test image (file or directory).  When ``test_image_path`` points to a
    directory, ``get_test_image()`` picks one JPEG/PNG at random each call,
    matching the behaviour required by ``GET /api/test-image``.

    Parameters
    ----------
    camera_id : str
        Identifier string returned by ``get_camera_id()``.
        Default: ``"cam_stub_01"``.
    test_image_path : str or None
        Path to a single image file, or to a directory containing JPEG/PNG
        images.  ``None`` disables test-image serving (``get_test_image()``
        returns ``None`` silently).

    Requirements: 9.1, 9.2, 9.4, 9.5, 15.1, 15.2
    """

    def __init__(
        self,
        camera_id: str = "cam_stub_01",
        test_image_path: Optional[str] = None,
    ) -> None:
        self._camera_id = camera_id
        self._test_image_path = test_image_path

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def get_camera_id(self) -> str:
        """
        Return the configured camera identifier string.

        Requirement 9.1
        """
        return self._camera_id

    def get_test_image(self) -> Optional[np.ndarray]:
        """
        Return a test image as a decoded NumPy array (BGR, OpenCV convention).

        Behaviour
        ---------
        - If ``test_image_path`` is ``None``: return ``None`` silently.
        - If ``test_image_path`` is a regular file: read it with ``cv2.imread``
          and return the result.
        - If ``test_image_path`` is a directory: collect all JPEG/PNG files
          recursively, pick one at random with ``random.choice()``, and return
          the decoded image.
        - If the path is set but the file/directory cannot be read (missing,
          unreadable format, no images in directory, or ``cv2.imread`` returns
          ``None``): log a WARNING with the path and return ``None`` without
          raising an exception.

        Requirements: 9.2, 9.5, 15.1, 15.2
        """
        # No path configured — silent None (Req 9.5, 15.2)
        if self._test_image_path is None:
            return None

        path = self._test_image_path

        if os.path.isfile(path):
            return self._read_image(path)

        if os.path.isdir(path):
            return self._read_random_from_dir(path)

        # Path is set but doesn't exist as a file or directory.
        logger.warning(
            "CameraStub.get_test_image: path does not exist or is not "
            "accessible: %s",
            path,
        )
        return None

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _read_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Read a single image file via cv2.imread.

        Returns the decoded ndarray, or None (with a warning) on failure.
        """
        image = cv2.imread(file_path)
        if image is None:
            logger.warning(
                "CameraStub.get_test_image: cv2.imread returned None for "
                "file: %s",
                file_path,
            )
            return None
        return image

    def _collect_images(self, dir_path: str) -> list[str]:
        """
        Return a sorted list of JPEG/PNG file paths inside *dir_path*.

        Uses ``glob`` to scan one directory level; uses ``os.walk`` as a
        fallback is not needed since glob with ** handles recursion, but we
        keep it simple and non-recursive to avoid scanning huge trees.
        """
        found: list[str] = []
        for ext in _IMAGE_EXTENSIONS:
            # glob is case-sensitive on Linux; on Windows it isn't, but we
            # add both lower and upper patterns to be safe.
            found.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
            found.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
        # Deduplicate (Windows glob can return duplicates for mixed-case) and sort.
        return sorted(set(found))

    def _read_random_from_dir(self, dir_path: str) -> Optional[np.ndarray]:
        """
        Pick a random JPEG/PNG from *dir_path* and return its decoded image.

        Returns None (with a warning) when the directory contains no eligible
        images or when the selected file cannot be decoded.
        """
        candidates = self._collect_images(dir_path)
        if not candidates:
            logger.warning(
                "CameraStub.get_test_image: no JPEG/PNG images found in "
                "directory: %s",
                dir_path,
            )
            return None

        chosen = random.choice(candidates)
        return self._read_image(chosen)
