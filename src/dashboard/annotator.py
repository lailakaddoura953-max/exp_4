"""
Annotator — draw bounding boxes and labels on inference results.

Implements ``annotate(image, results) -> Optional[str]``:
  - Converts normalised YOLO bbox coordinates to pixel space.
  - Draws red boxes for is_hazard=True, green for is_hazard=False.
  - Labels each box with "{class_label} {confidence:.0%}".
  - For hazard detections, overlays hazard_reason in smaller text beneath
    the class label.
  - Returns the annotated image as a base64-encoded PNG string.
  - Returns None only on hard failure (invalid image input).
  - Returns the partial annotated image on recoverable per-detection errors.

Requirements: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import base64
import logging
from typing import List, Optional

import cv2
import numpy as np

from dashboard.models import HazardResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

# Box colours — BGR format (OpenCV convention)
_COLOUR_HAZARD = (0, 0, 255)      # Red   — is_hazard=True   (Req 11.1)
_COLOUR_SAFE = (0, 255, 0)        # Green — is_hazard=False  (Req 11.1)

_BOX_THICKNESS = 2                # pixels

# Primary label font settings
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.5
_FONT_SCALE_REASON = 0.38         # Slightly smaller for hazard_reason
_FONT_THICKNESS = 1

# Vertical line height for text rows (pixels)
_LINE_HEIGHT = 18


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate(
    image: np.ndarray,
    results: List[HazardResult],
) -> Optional[str]:
    """
    Draw bounding boxes and labels on a copy of *image* for each result.

    Parameters
    ----------
    image:
        Decoded image as a NumPy array (2D greyscale or 3D BGR/BGRA).
        The original array is never mutated.
    results:
        List of HazardResult objects produced by the InferenceEngine.

    Returns
    -------
    str
        Base64-encoded PNG of the annotated image.
    None
        Only on hard failure: image is None, empty, or has an invalid shape
        (not 2-D or 3-D), or cv2.imencode fails completely.

    On recoverable per-detection error the failing detection is skipped and
    the partial annotated image is still returned (Req 11.4).
    """
    # ------------------------------------------------------------------
    # Hard-failure guard — invalid input (Req 11.4)
    # ------------------------------------------------------------------
    if image is None:
        logger.error("annotate: received None image — returning None")
        return None

    if not isinstance(image, np.ndarray) or image.size == 0:
        logger.error("annotate: received empty or non-array image — returning None")
        return None

    if image.ndim not in (2, 3):
        logger.error(
            "annotate: image has invalid number of dimensions (%d) — returning None",
            image.ndim,
        )
        return None

    # ------------------------------------------------------------------
    # Work on a copy — never mutate the original (task note)
    # ------------------------------------------------------------------
    annotated = image.copy()

    # Ensure the image is BGR (3-channel) so cv2 drawing works correctly.
    # Greyscale (2-D) → convert to BGR; BGRA (4-channel) → strip alpha.
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    elif annotated.shape[2] == 4:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)

    h, w = annotated.shape[:2]

    # ------------------------------------------------------------------
    # Draw each detection — recoverable errors skip the failing box
    # ------------------------------------------------------------------
    for result in results:
        try:
            _draw_detection(annotated, result, h, w)
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            logger.warning(
                "annotate: skipping detection '%s' due to error: %s",
                result.class_label,
                exc,
            )
            # Continue drawing remaining detections (Req 11.4)

    # ------------------------------------------------------------------
    # Encode to PNG and base64 (Req 11.3)
    # ------------------------------------------------------------------
    try:
        success, buffer = cv2.imencode(".png", annotated)
        if not success or buffer is None:
            logger.error("annotate: cv2.imencode failed — returning None")
            return None
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.error("annotate: base64 encoding failed: %s — returning None", exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_detection(
    img: np.ndarray,
    result: HazardResult,
    h: int,
    w: int,
) -> None:
    """
    Draw a single bounding box + labels onto *img* (in-place).

    Bbox conversion (normalised YOLO → pixel coords):
        x1 = int((x_center - width/2)  * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2)  * w)
        y2 = int((y_center + height/2) * h)

    Parameters correspond to a single HazardResult; *h* and *w* are the
    image pixel dimensions already extracted by the caller.
    """
    bbox = result.bbox
    colour = _COLOUR_HAZARD if result.is_hazard else _COLOUR_SAFE

    # --- Convert normalised coords to pixel space -----------------------
    x1 = int((bbox.x_center - bbox.width / 2) * w)
    y1 = int((bbox.y_center - bbox.height / 2) * h)
    x2 = int((bbox.x_center + bbox.width / 2) * w)
    y2 = int((bbox.y_center + bbox.height / 2) * h)

    # Clamp to image bounds to avoid OpenCV complaints
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    # --- Draw bounding box (Req 11.1) -----------------------------------
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, _BOX_THICKNESS)

    # --- Build label text (Req 11.2) ------------------------------------
    # "{class_label} {confidence:.0%}"  e.g. "Container - Misaligned 87%"
    label = f"{result.class_label} {result.confidence:.0%}"

    # Text is placed just above the top-left corner of the box.
    # If the box is near the top, move text inside.
    label_y = y1 - 4
    if label_y < _LINE_HEIGHT:
        label_y = y1 + _LINE_HEIGHT

    # --- Draw label background rectangle for readability ----------------
    (label_w, label_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE_LABEL, _FONT_THICKNESS
    )
    cv2.rectangle(
        img,
        (x1, label_y - label_h - baseline),
        (x1 + label_w, label_y + baseline),
        colour,
        cv2.FILLED,
    )

    # --- Draw label text ------------------------------------------------
    # Use white text so it's readable on both red and green backgrounds.
    cv2.putText(
        img,
        label,
        (x1, label_y),
        _FONT,
        _FONT_SCALE_LABEL,
        (255, 255, 255),  # white
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )

    # --- Draw hazard_reason beneath label when is_hazard=True (Req 11.2) -
    if result.is_hazard and result.hazard_reason:
        reason_y = label_y + _LINE_HEIGHT
        (reason_w, reason_h), reason_baseline = cv2.getTextSize(
            result.hazard_reason, _FONT, _FONT_SCALE_REASON, _FONT_THICKNESS
        )
        # Background for reason text
        cv2.rectangle(
            img,
            (x1, reason_y - reason_h - reason_baseline),
            (x1 + reason_w, reason_y + reason_baseline),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            result.hazard_reason,
            (x1, reason_y),
            _FONT,
            _FONT_SCALE_REASON,
            (255, 255, 255),  # white
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
