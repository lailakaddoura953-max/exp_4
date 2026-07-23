"""
Class taxonomy for the Camera-Location-Aware Hazard Rules engine.

Defines the single, shared source of truth for the YOLO detection class
list used across training, dataset-generation, and inference scripts, per
requirements.md Requirement 12.

FULL_CLASS_NAMES is the original 17-class Roboflow taxonomy (unchanged,
kept for reference and for scripts/config that still operate on
already-existing 17-class-indexed data, per Requirement 12.6).

REDUCED_CLASS_SET is the 12-class list HSSE's confirmed rules actually
reference, dropping five classes that are not needed for the current rule
set (Requirement 12.1, 12.2).

IMPORTANT — this requires a NEW model training run from scratch, not a
fine-tune of an existing 17-class checkpoint. Dropping classes changes
class-index numbering: a YOLO model's output layer size and
index-to-name mapping are fixed at training time (Requirement 12.3). Any
checkpoint trained on FULL_CLASS_NAMES's 17-class indexing is NOT
compatible with REDUCED_CLASS_SET's 12-class indexing, and vice versa.
"""

from typing import Dict, List

# ============================================================================
# Original 17-class Roboflow taxonomy (unchanged)
# ============================================================================

FULL_CLASS_NAMES: List[str] = [
    "Boat - With Cargo",         # 0
    "Container - Misaligned",    # 1
    "Container - Open",          # 2
    "Container - Picked",        # 3
    "Container - Reefer",        # 4
    "Container - Water Drop",    # 5
    "Container -Separate",       # 6
    "Container -Stacked",        # 7
    "Crane",                     # 8
    "Human",                     # 9
    "Human - No Safety Clothes",  # 10
    "Truck - No Container",      # 11
    "Truck - With Container",    # 12
    "Vehicle",                   # 13
    "Yard - Dropoff zone",       # 14
    "Yard - No People",          # 15
    "Yard - Operation Zone",     # 16
]

# Class indices (into FULL_CLASS_NAMES) dropped from the Reduced_Class_Set,
# because HSSE's confirmed rules never reference them (Requirement 12.1).
DROPPED_CLASS_INDICES: List[int] = [0, 4, 5, 6, 14]

# ============================================================================
# Reduced_Class_Set (Requirement 12.2)
#
# Retains FULL_CLASS_NAMES's original relative ordering, just with the
# dropped indices removed — so REDUCED_CLASS_SET[i] always corresponds to
# the same real-world class regardless of how DROPPED_CLASS_INDICES is
# defined, rather than being hand-typed out of sync with FULL_CLASS_NAMES.
# ============================================================================

REDUCED_CLASS_SET: List[str] = [
    name for i, name in enumerate(FULL_CLASS_NAMES) if i not in DROPPED_CLASS_INDICES
]

assert len(REDUCED_CLASS_SET) == 12, (
    f"REDUCED_CLASS_SET must have exactly 12 classes, got {len(REDUCED_CLASS_SET)}. "
    f"Check DROPPED_CLASS_INDICES against FULL_CLASS_NAMES."
)

# ============================================================================
# Audit: known locations elsewhere in the codebase with class-index or
# class-name assumptions tied to the original 17-class taxonomy
# (requirements.md Requirement 12.5, Task 9.2). None of these are changed
# by this module — they are listed here so Phase B's retraining task
# (Requirement 14.3) can address them explicitly before any cutover to a
# Reduced_Class_Set-trained checkpoint.
#
# 1. src/hazard_detection/container_analyzer.py, _process_detection():
#    the `container_classes` set gating the flipped-container check
#    hard-codes "Container - Separate", "Container - Reefer", and
#    "Container - Water Drop" — three of the five DROPPED_CLASS_INDICES
#    classes. Once retrained on REDUCED_CLASS_SET, the YOLO model will
#    never emit these class labels again, so these three entries become
#    permanently-dead branches (not a crash risk — string equality
#    against a class label that never occurs just never matches — but
#    dead code that should be cleaned up once the 12-class checkpoint is
#    the production model).
#
# 2. src/hazard_detection/zone_map.py, YOLO_CLASS_TO_ZONE_TYPE /
#    VALID_ZONE_TYPES: maps "Yard - Dropoff zone" (a dropped class) to
#    the "dropoff" zone type. Same situation as #1 — not a crash risk,
#    but the "dropoff" zone type will never be initialized from live
#    YOLO detections once the model is retrained without that class
#    (ZoneMap.initialize_from_detections() simply won't receive it).
#    Any zone config file that manually defines a "dropoff" zone_type
#    for a camera would still work (zone_type is independent of the
#    detector's class list), so this is a training-time gap, not a
#    ZoneMap defect.
#
# 3. Neither container_analyzer.py nor human_detector.py hard-codes a
#    class INDEX or a class COUNT (e.g. no "if len(classes) == 17" or
#    "class_id == 4" logic was found) — all class references found are
#    by class NAME (string equality), which is why the retraining
#    cutover risk here is "dead code that never matches" rather than an
#    index-mismatch crash. This is a materially different (safer) risk
#    profile than, e.g., a hard-coded output-layer size assumption would
#    be — but config/hazard_detection.yaml's `checkpoint_path` still
#    must be re-pointed deliberately (Requirement 14.3, 14.4), since a
#    17-class checkpoint and a 12-class checkpoint are not
#    interchangeable at the YOLODetector/model level regardless of how
#    downstream code references class names.
# ============================================================================

# Mapping from the ORIGINAL 17-class index to the NEW 0-based Reduced_Class_Set
# index, for remapping already-existing labels without re-annotating
# (Requirement 12.4, 12.6). Dropped classes are absent from this mapping —
# callers MUST check membership before remapping a label's class_id.
FULL_TO_REDUCED_INDEX: Dict[int, int] = {
    old_index: REDUCED_CLASS_SET.index(name)
    for old_index, name in enumerate(FULL_CLASS_NAMES)
    if old_index not in DROPPED_CLASS_INDICES
}
