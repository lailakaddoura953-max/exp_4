"""
Rule Engine for the Yard Hazard Inference Dashboard.

Pure, stateless helper functions used by ``classify_all`` to evaluate each
YOLO detection and produce a ``HazardResult``.  All functions here are side-
effect free — they receive plain data and return plain data.

Requirements covered:
- 2.1, 2.2, 2.3  : Unconditional hazard classes (Misaligned, Water Drop)
- 3.1, 3.2, 3.3  : Container-Open loading suppression via IoU
- 3.4             : IoU computed in normalised centre-format consistent with
                    ContainerAnalyzer._compute_iou
- 4.1, 4.2, 4.3, 4.4 : Container-Picked crane supervision and person proximity
- 5.1, 5.2        : PPE violation (Human - No Safety Clothes)
- 6.1, 6.2, 6.3  : Human stub with crane override; documented extension point
- 7.1, 7.2, 7.3  : Flipped container check (all container classes)
- 7.4             : is_flipped returns False when bbox.width == 0
- 8.1, 8.2, 8.3  : Non-hazard and zone classes → is_hazard=False
"""

from typing import List

import numpy as np

from hazard_detection.models import BBox, Detection
from dashboard.models import HazardResult, InferenceEngineConfig


# ---------------------------------------------------------------------------
# Class-set constants used by the rule chain
# ---------------------------------------------------------------------------

# All container classes subject to the flipped-container check (Requirement 7.1)
CONTAINER_CLASSES = frozenset({
    "Container - Stacked",
    "Container - Reefer",
    "Container - Separate",
    "Container - Open",
    "Container - Picked",
    "Container - Misaligned",
    "Container - Water Drop",
})

# Non-hazard classes that always produce is_hazard=False (unless flipped)
# Requirement 8.1
NON_HAZARD_CLASSES = frozenset({
    "Container - Separate",
    "Container - Stacked",
    "Container - Reefer",
    "Vehicle",
    "Truck - No Container",
    "Truck - With Container",
    "Boat - With Cargo",
    "Crane",
})

# Zone classes that are always context-only providers, never hazards
# Requirement 8.2, 8.3
ZONE_CLASSES = frozenset({
    "Yard - No People",
    "Yard - Operation Zone",
    "Yard - Dropoff zone",
})


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """
    Compute Intersection-over-Union between two normalised centre-format bboxes.

    Replicates ``ContainerAnalyzer._compute_iou`` as a standalone pure function.

    Args:
        box_a: First bounding box in normalised YOLO centre format.
        box_b: Second bounding box in normalised YOLO centre format.

    Returns:
        IoU value clamped to [0.0, 1.0]; 0.0 when union area is zero.

    Requirements: 3.4
    """
    # Convert centre-format to corner format: (x1, y1, x2, y2)
    a_x1 = box_a.x_center - box_a.width / 2.0
    a_y1 = box_a.y_center - box_a.height / 2.0
    a_x2 = box_a.x_center + box_a.width / 2.0
    a_y2 = box_a.y_center + box_a.height / 2.0

    b_x1 = box_b.x_center - box_b.width / 2.0
    b_y1 = box_b.y_center - box_b.height / 2.0
    b_x2 = box_b.x_center + box_b.width / 2.0
    b_y2 = box_b.y_center + box_b.height / 2.0

    # Intersection rectangle
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    iou = inter_area / union_area
    return float(np.clip(iou, 0.0, 1.0))


def is_flipped(bbox: BBox, threshold: float) -> bool:
    """
    Return True iff the bounding box has a height/width ratio above ``threshold``.

    A normally-oriented shipping container is wider than it is tall; a flipped
    container has height > width beyond the configured ratio.

    Returns False for degenerate bboxes where ``bbox.width == 0`` (Req 7.4).

    Args:
        bbox:      Bounding box to evaluate.
        threshold: Aspect-ratio threshold (height / width) above which the
                   container is considered flipped.

    Returns:
        True if ``bbox.width > 0`` and ``bbox.height / bbox.width > threshold``.

    Requirements: 7.1, 7.4
    """
    if bbox.width == 0:
        # Degenerate bbox — cannot compute ratio; treat as inconclusive (Req 7.4)
        return False
    return bbox.height / bbox.width > threshold


def person_below_crane(person_bbox: BBox, crane_bbox: BBox) -> bool:
    """
    Return True iff the person's vertical centre is at or below the crane's.

    In normalised image coordinates y increases downward, so a person whose
    ``y_center`` is greater-than-or-equal-to the crane's ``y_center`` is in
    the lower half of the crane bounding box — the danger zone beneath a lift.

    Args:
        person_bbox: Bounding box of a ``Human`` or ``Human - No Safety Clothes``
                     detection.
        crane_bbox:  Bounding box of a ``Crane`` detection.

    Returns:
        True iff ``person_bbox.y_center >= crane_bbox.y_center``.

    Requirements: 4.3
    """
    return person_bbox.y_center >= crane_bbox.y_center


# ---------------------------------------------------------------------------
# classify_all — entry point for rule application
# ---------------------------------------------------------------------------

def classify_all(
    detections: List[Detection],
    config: InferenceEngineConfig,
    camera_id: str,
) -> List[HazardResult]:
    """
    Classify every detection in a single frame using the priority-ordered rule chain.

    Processes detections in two passes:

    **Pass 1** — build context lookups from all detections in the frame:
        - ``crane_boxes``   : BBox list for all Crane detections
        - ``picked_boxes``  : BBox list for all Container-Picked detections
        - ``human_bboxes``  : Detection list for Human + Human-No-Safety-Clothes

    **Pass 2** — apply the following priority-ordered rules to each detection
    (once a rule fires for a detection, no further rules are evaluated for it):

        Rule 0  — Flipped container (all container classes; Req 7.1–7.4)
        Rule 1  — Unconditional hazards: Misaligned, Water Drop (Req 2.1–2.3)
        Rule 2  — Container-Open loading suppression via IoU (Req 3.1–3.4)
        Rule 3  — Container-Picked: crane/person proximity (Req 4.1–4.4)
        Rule 4  — Human - No Safety Clothes → ppe_violation (Req 5.1–5.2)
        Rule 5  — Human: crane override → human_below_crane;
                  stub fallback → human_detected_stub (Req 6.1–6.3)
        Rule 6  — Zone classes → is_hazard=False (Req 8.2–8.3)
        Rule 7  — Non-hazard classes → is_hazard=False (Req 8.1)
        Rule 8  — Unknown class defensive fallback → is_hazard=False

    Args:
        detections: List of ``Detection`` objects already filtered by confidence
                    threshold.  One ``HazardResult`` is produced per detection.
        config:     ``InferenceEngineConfig`` supplying threshold values
                    (``flipped_aspect_ratio_threshold``, ``iou_threshold``).
        camera_id:  Camera identifier propagated onto every ``HazardResult``.

    Returns:
        A list of ``HazardResult`` objects — one per detection, in the same
        order as the input list.

    Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4,
                  5.1, 5.2, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 8.1, 8.2, 8.3
    """
    # ------------------------------------------------------------------
    # Pass 1 — build context lookups
    # ------------------------------------------------------------------

    # All Crane bounding boxes present in this frame
    crane_boxes: List[BBox] = [
        d.bbox for d in detections if d.class_label == "Crane"
    ]

    # All Container - Picked bounding boxes (used in Rule 2 IoU check)
    picked_boxes: List[BBox] = [
        d.bbox for d in detections if d.class_label == "Container - Picked"
    ]

    # All human detections — both variants used for Rule 3 and Rule 5
    # proximity checks (Requirement 4.3 and 6.4)
    human_bboxes: List[Detection] = [
        d for d in detections
        if d.class_label in {"Human", "Human - No Safety Clothes"}
    ]

    # ------------------------------------------------------------------
    # Pass 2 — per-detection rule chain
    # ------------------------------------------------------------------

    results: List[HazardResult] = []

    for det in detections:
        label = det.class_label
        is_hazard: bool
        hazard_reason: str

        # ------------------------------------------------------------------
        # Rule 0 — Flipped container check
        # Applies to ALL container classes listed in Requirement 7.1.
        # Runs BEFORE Rule 1 so that a flipped misaligned/water-drop container
        # is reported as "flipped_container" (the more structural concern),
        # while the class_label still surfaces the misaligned/water-drop info.
        # Requirements: 7.1, 7.2, 7.3, 7.4
        # ------------------------------------------------------------------
        if label in CONTAINER_CLASSES and is_flipped(
            det.bbox, config.flipped_aspect_ratio_threshold
        ):
            is_hazard = True
            hazard_reason = "flipped_container"

        # ------------------------------------------------------------------
        # Rule 1 — Unconditional hazards
        # No spatial overlap or proximity checks apply here.
        # Requirements: 2.1, 2.2, 2.3
        # ------------------------------------------------------------------
        elif label == "Container - Misaligned":
            is_hazard = True
            hazard_reason = "misaligned_container"

        elif label == "Container - Water Drop":
            is_hazard = True
            hazard_reason = "water_drop_container"

        # ------------------------------------------------------------------
        # Rule 2 — Container - Open (loading operation suppression)
        # Suppress if any Crane or Container-Picked bbox overlaps this bbox
        # with IoU >= iou_threshold (active loading operation context).
        # Requirements: 3.1, 3.2, 3.3, 3.4
        # ------------------------------------------------------------------
        elif label == "Container - Open":
            suppressed = any(
                compute_iou(det.bbox, ctx_box) >= config.iou_threshold
                for ctx_box in (crane_boxes + picked_boxes)
            )
            if suppressed:
                # Active loading operation detected — suppress the hazard
                is_hazard = False
                hazard_reason = ""
            else:
                is_hazard = True
                hazard_reason = "open_container_unsecured"

        # ------------------------------------------------------------------
        # Rule 3 — Container - Picked (crane supervision + person proximity)
        # Requirements: 4.1, 4.2, 4.3, 4.4
        # ------------------------------------------------------------------
        elif label == "Container - Picked":
            if not crane_boxes:
                # No crane detected — unsupervised lift
                is_hazard = True
                hazard_reason = "picked_no_crane"
            elif any(
                person_below_crane(h.bbox, crane_bbox)
                for h in human_bboxes
                for crane_bbox in crane_boxes
            ):
                # Person positioned in the lower half of a crane bbox
                is_hazard = True
                hazard_reason = "picked_person_below_crane"
            else:
                # Crane present; no person below — safe supervision
                is_hazard = False
                hazard_reason = ""

        # ------------------------------------------------------------------
        # Rule 4 — Human - No Safety Clothes (PPE violation)
        # Unconditional — not suppressible by zone or any other rule.
        # Requirements: 5.1, 5.2
        # ------------------------------------------------------------------
        elif label == "Human - No Safety Clothes":
            is_hazard = True
            hazard_reason = "ppe_violation"

        # ------------------------------------------------------------------
        # Rule 5 — Human (crane override + stub fallback)
        #
        # STUB BEHAVIOUR (current implementation):
        #   Every Human detection is flagged as is_hazard=True with
        #   hazard_reason="human_detected_stub" unless a crane override fires.
        #
        # CRANE OVERRIDE:
        #   If the human's bounding-box centre falls within the lower half of
        #   any Crane bounding box (y_center of person >= y_center of crane),
        #   the override takes precedence and hazard_reason="human_below_crane"
        #   is emitted instead of the stub.  This override is permanent even
        #   after zone maps are activated (Requirement 6.4).
        #
        # FUTURE ZONE MAP RULE (extension point — NOT active):
        #   When zone maps are configured and loaded, replace the stub fallback
        #   block below with:
        #     - is_hazard=True,  reason="human_in_no_people_zone"
        #         if the detection centre falls inside a "Yard - No People" zone.
        #     - is_hazard=True,  reason="human_in_operation_zone" (conditional)
        #         if the detection centre falls inside a "Yard - Operation Zone".
        #     - is_hazard=False, hazard_reason=""
        #         if the detection centre falls inside a "Yard - Dropoff zone".
        #   Precondition for activating: zone_maps must be loaded into config
        #   and the camera_id must have a corresponding zone map entry.
        #   Until that precondition is met, this stub fires for ALL Human detections.
        #
        # Requirements: 6.1, 6.2, 6.3, 6.4
        # ------------------------------------------------------------------
        elif label == "Human":
            if any(
                person_below_crane(det.bbox, crane_bbox)
                for crane_bbox in crane_boxes
            ):
                # Crane override — person is beneath an active crane lift
                is_hazard = True
                hazard_reason = "human_below_crane"
            else:
                # STUB fallback — flag all humans until zone maps are available
                # TODO (zone-map-extension): Replace this stub with a real zone
                #   map lookup once zone_maps are loaded and camera_id has an
                #   entry.  See Rule 5 comment block above for the full future
                #   rule specification and activation preconditions.
                is_hazard = True
                hazard_reason = "human_detected_stub"

        # ------------------------------------------------------------------
        # Rule 6 — Zone classes (always context-only, never hazards)
        # Requirements: 8.2, 8.3
        # ------------------------------------------------------------------
        elif label in ZONE_CLASSES:
            is_hazard = False
            hazard_reason = ""

        # ------------------------------------------------------------------
        # Rule 7 — Non-hazard classes
        # These classes are never hazards unless the flipped rule fired first
        # (Rule 0 already handled that case above).
        # Requirements: 8.1
        # ------------------------------------------------------------------
        elif label in NON_HAZARD_CLASSES:
            is_hazard = False
            hazard_reason = ""

        # ------------------------------------------------------------------
        # Rule 8 — Unknown class defensive fallback
        # Any class label not covered by Rules 0–7 is treated as non-hazard.
        # This prevents an unrecognised label from propagating an exception.
        # ------------------------------------------------------------------
        else:
            is_hazard = False
            hazard_reason = ""

        results.append(
            HazardResult(
                class_label=label,
                confidence=det.confidence,
                bbox=det.bbox,
                is_hazard=is_hazard,
                hazard_reason=hazard_reason,
                camera_id=camera_id,
            )
        )

    return results
