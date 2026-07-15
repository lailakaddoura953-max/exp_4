"""
Human Detector for the Hazard Detection System.

Interprets YOLO detections for human classes and cross-references the
Zone Map to determine zone violations and PPE violations.

Binary hazard classification:
- is_hazard=True: confirmed detection (>=2 consecutive frames above threshold
  in no-people zone, or any PPE violation above threshold)
- is_hazard=False: transient detection (1 frame only) or below-threshold
  detections — logged only, no alert dispatch

Requirements covered:
- 2.1: Use YOLO outputs for "Human" and "Human - No Safety Clothes" classes
- 2.2: Emit zone_violation when person in no-people zone
- 2.3: Include confidence score, use as filter threshold
- 2.4: Do NOT emit zone_violation for operation or dropoff zones
- 2.5: Log below-threshold detections without emitting HazardEvent
- 2.6: Single frame detection = transient (log only, not a hazard)
- 2.7: >=2 consecutive frames = confirmed (is_hazard=True)
- 2.8: PPE violation for "Human - No Safety Clothes" regardless of zone
- 2.9: Detections outside all zones treated as no-people zone
"""

from typing import Dict, List, Optional, Tuple

from hazard_detection.diagnostics import DiagnosticDumper, get_logger
from hazard_detection.models import (
    BBox,
    Detection,
    DiagnosticMetadata,
    HazardEvent,
    HumanDetectorConfig,
)
from hazard_detection.zone_map import ZoneMap

logger = get_logger("human_detector")

# Human detection class labels from the Roboflow 17-class taxonomy
HUMAN_CLASSES = {"Human", "Human - No Safety Clothes"}
PPE_VIOLATION_CLASS = "Human - No Safety Clothes"


class HumanDetector:
    """
    Interprets YOLO detections for human classes and cross-references zone map.

    Filters detections for 'Human' and 'Human - No Safety Clothes' classes,
    cross-references bounding box centers against the ZoneMap, and emits
    HazardEvents for zone violations and PPE violations.

    Temporal logic:
    - 1 frame = "transient" (is_hazard=False, logged only)
    - >=2 consecutive frames = "confirmed" (is_hazard=True)

    PPE violations are an exception: they emit as confirmed hazards on any
    single frame above threshold.
    """

    def __init__(
        self,
        zone_map: ZoneMap,
        config: HumanDetectorConfig,
        diagnostic_dumper: Optional[DiagnosticDumper] = None,
    ):
        """
        Args:
            zone_map: Spatial zone definitions per camera for zone lookups.
            config: HumanDetectorConfig with confidence_threshold (default 0.5).
            diagnostic_dumper: Optional dumper for saving per-frame detection state.
        """
        self._zone_map = zone_map
        self._config = config
        self._dumper = diagnostic_dumper

        logger.info(
            f"HumanDetector initialized with confidence_threshold="
            f"{self._config.confidence_threshold}",
            extra={"component": "human_detector"},
        )

    def analyze(
        self, detections_per_frame: List[List[Detection]], camera_id: str
    ) -> List[HazardEvent]:
        """
        Analyze human detections across a frame sequence.

        Binary decision logic:
        - Filters for 'Human' and 'Human - No Safety Clothes' classes
        - Cross-references bbox center against zone_map
        - HAZARD (is_hazard=True): confidence >= threshold AND detected in
          >=2 consecutive frames in no-people zone
        - HAZARD (is_hazard=True): 'Human - No Safety Clothes' with confidence
          >= threshold in ANY single frame (PPE exception)
        - NOT HAZARD (is_hazard=False): confidence below threshold OR detected
          in fewer than 2 consecutive frames in no-people zone
        - Treats detections outside any zone as no-people zone

        Args:
            detections_per_frame: List of detection lists, one per frame in the
                                  frame sequence.
            camera_id: Camera identifier for zone map lookups.

        Returns:
            List of HazardEvent objects for all detected hazards (both confirmed
            and transient are returned; is_hazard flag distinguishes them).
        """
        logger.info(
            f"Analyzing {len(detections_per_frame)} frames for camera '{camera_id}'",
            extra={"component": "human_detector", "camera_id": camera_id},
        )

        hazard_events: List[HazardEvent] = []

        # Step 1: Extract and classify human detections per frame
        per_frame_human_detections = self._extract_human_detections(
            detections_per_frame, camera_id
        )

        # Step 2: Process PPE violations (any single frame above threshold)
        ppe_events = self._process_ppe_violations(
            per_frame_human_detections, camera_id
        )
        hazard_events.extend(ppe_events)

        # Step 3: Process zone violations with temporal logic
        zone_events = self._process_zone_violations(
            per_frame_human_detections, camera_id
        )
        hazard_events.extend(zone_events)

        # Diagnostic dump
        if self._dumper:
            self._dumper.dump(
                stage="human_detector_analysis",
                state={
                    "camera_id": camera_id,
                    "total_frames": len(detections_per_frame),
                    "human_detections_per_frame": [
                        len(frame_dets) for frame_dets in per_frame_human_detections
                    ],
                    "ppe_events_count": len(ppe_events),
                    "zone_events_count": len(zone_events),
                    "total_hazard_events": len(hazard_events),
                },
                camera_id=camera_id,
            )

        logger.info(
            f"Human detection analysis complete for camera '{camera_id}': "
            f"{len(hazard_events)} hazard events "
            f"({len(ppe_events)} PPE, {len(zone_events)} zone)",
            extra={"component": "human_detector", "camera_id": camera_id},
        )

        return hazard_events

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _extract_human_detections(
        self, detections_per_frame: List[List[Detection]], camera_id: str
    ) -> List[List[Dict]]:
        """
        Extract and enrich human detections from all frames.

        For each detection of class "Human" or "Human - No Safety Clothes",
        look up the zone type and annotate accordingly.

        Returns:
            List of lists (one per frame) of enriched detection dicts containing:
            - detection: the original Detection object
            - zone_type: the zone type from the zone map
            - frame_index: the frame index in the sequence
            - above_threshold: whether confidence meets threshold
        """
        per_frame_results: List[List[Dict]] = []

        for frame_idx, frame_detections in enumerate(detections_per_frame):
            frame_humans: List[Dict] = []

            for detection in frame_detections:
                if detection.class_label not in HUMAN_CLASSES:
                    continue

                # Get bbox center for zone lookup
                center = detection.bbox.center
                zone_type = self._zone_map.get_zone_type(camera_id, center)

                above_threshold = (
                    detection.confidence >= self._config.confidence_threshold
                )

                enriched = {
                    "detection": detection,
                    "zone_type": zone_type,
                    "frame_index": frame_idx,
                    "above_threshold": above_threshold,
                }
                frame_humans.append(enriched)

                # Structured logging for each detection
                temporal_state = "pending"  # Will be resolved later
                logger.debug(
                    f"Frame {frame_idx}: {detection.class_label} detected at "
                    f"({center[0]:.4f}, {center[1]:.4f}), "
                    f"zone='{zone_type}', confidence={detection.confidence:.3f}, "
                    f"above_threshold={above_threshold}",
                    extra={
                        "component": "human_detector",
                        "camera_id": camera_id,
                    },
                )

                # Log below-threshold detections (Requirement 2.5)
                if not above_threshold:
                    logger.info(
                        f"Below-threshold detection: {detection.class_label} "
                        f"confidence={detection.confidence:.3f} < "
                        f"{self._config.confidence_threshold} "
                        f"(frame {frame_idx}, zone '{zone_type}'). Logged only.",
                        extra={
                            "component": "human_detector",
                            "camera_id": camera_id,
                        },
                    )

            per_frame_results.append(frame_humans)

        return per_frame_results

    def _process_ppe_violations(
        self, per_frame_human_detections: List[List[Dict]], camera_id: str
    ) -> List[HazardEvent]:
        """
        Process PPE violations: 'Human - No Safety Clothes' emits regardless of zone.

        PPE violations are confirmed hazards on any single frame above threshold.
        They do NOT require temporal confirmation (>=2 frames).

        Requirement 2.8: PPE violation regardless of zone type.

        Returns:
            List of HazardEvent objects for PPE violations.
        """
        events: List[HazardEvent] = []
        # Track already-emitted PPE violations to avoid duplicates within same analysis
        # We emit one PPE event per unique detection location pattern
        ppe_emitted_frames: set = set()

        for frame_idx, frame_dets in enumerate(per_frame_human_detections):
            for det_info in frame_dets:
                detection = det_info["detection"]

                # Only PPE violation class
                if detection.class_label != PPE_VIOLATION_CLASS:
                    continue

                # Must be above threshold
                if not det_info["above_threshold"]:
                    continue

                # Avoid duplicate PPE events for the same frame
                if frame_idx in ppe_emitted_frames:
                    continue

                ppe_emitted_frames.add(frame_idx)

                # Count how many frames this PPE detection appears in
                frames_with_ppe = self._count_ppe_frames(
                    per_frame_human_detections, detection.bbox
                )

                event = HazardEvent(
                    event_id=HazardEvent.generate_event_id(),
                    hazard_type="ppe_violation",
                    camera_id=camera_id,
                    timestamp=HazardEvent.generate_timestamp(),
                    is_hazard=True,  # PPE violations are always confirmed
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    metadata=DiagnosticMetadata(
                        frame_index=frame_idx,
                        detection_class=detection.class_label,
                        frames_detected=frames_with_ppe,
                    ),
                )

                logger.info(
                    f"PPE violation detected: confidence={detection.confidence:.3f}, "
                    f"zone='{det_info['zone_type']}', frame={frame_idx}, "
                    f"is_hazard=True",
                    extra={
                        "component": "human_detector",
                        "camera_id": camera_id,
                    },
                )

                events.append(event)
                # Only emit one PPE event per analysis (for the first above-threshold frame)
                break

        return events

    def _process_zone_violations(
        self, per_frame_human_detections: List[List[Dict]], camera_id: str
    ) -> List[HazardEvent]:
        """
        Process zone violations with temporal confirmation logic.

        Temporal logic:
        - 1 frame in no-people zone = "transient" (is_hazard=False, logged)
        - >=2 consecutive frames in no-people zone = "confirmed" (is_hazard=True)

        Detections in operation or dropoff zones do NOT emit zone_violation.
        Detections outside all zones are treated as no-people zone.

        Requirements 2.2, 2.4, 2.6, 2.7, 2.9.

        Returns:
            List of HazardEvent objects for zone violations.
        """
        events: List[HazardEvent] = []

        # Find consecutive sequences of above-threshold human detections
        # in no-people zones
        consecutive_count = 0
        best_detection_info: Optional[Dict] = None
        first_frame_idx = 0

        for frame_idx, frame_dets in enumerate(per_frame_human_detections):
            # Check if this frame has an above-threshold human in a no-people zone
            frame_has_violation = False
            frame_best_det: Optional[Dict] = None

            for det_info in frame_dets:
                detection = det_info["detection"]
                zone_type = det_info["zone_type"]

                # Skip below-threshold detections
                if not det_info["above_threshold"]:
                    continue

                # Only no-people zone triggers zone violations
                # (Requirement 2.4: operation and dropoff zones do NOT emit)
                if zone_type != "no_people":
                    continue

                frame_has_violation = True
                # Keep the highest-confidence detection for this frame
                if (
                    frame_best_det is None
                    or detection.confidence > frame_best_det["detection"].confidence
                ):
                    frame_best_det = det_info

            if frame_has_violation and frame_best_det is not None:
                if consecutive_count == 0:
                    first_frame_idx = frame_idx
                consecutive_count += 1
                # Track the best detection across the consecutive sequence
                if (
                    best_detection_info is None
                    or frame_best_det["detection"].confidence
                    > best_detection_info["detection"].confidence
                ):
                    best_detection_info = frame_best_det
            else:
                # Sequence broken — emit event for previous sequence if any
                if consecutive_count > 0 and best_detection_info is not None:
                    event = self._create_zone_violation_event(
                        best_detection_info, camera_id, consecutive_count, first_frame_idx
                    )
                    events.append(event)

                # Reset tracking
                consecutive_count = 0
                best_detection_info = None

        # Handle trailing sequence (sequence continues to end of frames)
        if consecutive_count > 0 and best_detection_info is not None:
            event = self._create_zone_violation_event(
                best_detection_info, camera_id, consecutive_count, first_frame_idx
            )
            events.append(event)

        return events

    def _create_zone_violation_event(
        self,
        det_info: Dict,
        camera_id: str,
        consecutive_count: int,
        first_frame_idx: int,
    ) -> HazardEvent:
        """
        Create a zone_violation HazardEvent based on temporal classification.

        Args:
            det_info: Enriched detection dict with the best detection info.
            camera_id: Camera identifier.
            consecutive_count: Number of consecutive frames the detection appeared.
            first_frame_idx: Index of the first frame in the consecutive sequence.

        Returns:
            HazardEvent with is_hazard set based on temporal confirmation.
        """
        detection = det_info["detection"]

        # Temporal classification:
        # 1 frame = transient (is_hazard=False)
        # >=2 consecutive = confirmed (is_hazard=True)
        is_confirmed = consecutive_count >= 2
        temporal_state = "confirmed" if is_confirmed else "transient"

        event = HazardEvent(
            event_id=HazardEvent.generate_event_id(),
            hazard_type="zone_violation",
            camera_id=camera_id,
            timestamp=HazardEvent.generate_timestamp(),
            is_hazard=is_confirmed,
            confidence=detection.confidence,
            bbox=detection.bbox,
            metadata=DiagnosticMetadata(
                frame_index=first_frame_idx,
                detection_class=detection.class_label,
                frames_detected=consecutive_count,
            ),
        )

        logger.info(
            f"Zone violation: temporal_state='{temporal_state}', "
            f"consecutive_frames={consecutive_count}, "
            f"confidence={detection.confidence:.3f}, "
            f"is_hazard={is_confirmed}, "
            f"first_frame={first_frame_idx}",
            extra={
                "component": "human_detector",
                "camera_id": camera_id,
            },
        )

        return event

    def _count_ppe_frames(
        self, per_frame_human_detections: List[List[Dict]], target_bbox: BBox
    ) -> int:
        """
        Count frames containing a PPE violation detection near the target bbox.

        Uses a simple center-distance proximity check to identify likely
        same-person detections across frames.

        Args:
            per_frame_human_detections: All enriched detection data per frame.
            target_bbox: BBox of the reference detection.

        Returns:
            Number of frames containing a matching PPE detection.
        """
        count = 0
        target_center = target_bbox.center
        proximity_threshold = 0.15  # Normalized distance for matching

        for frame_dets in per_frame_human_detections:
            for det_info in frame_dets:
                detection = det_info["detection"]
                if detection.class_label != PPE_VIOLATION_CLASS:
                    continue
                if not det_info["above_threshold"]:
                    continue

                # Simple center-distance proximity check
                det_center = detection.bbox.center
                distance = (
                    (det_center[0] - target_center[0]) ** 2
                    + (det_center[1] - target_center[1]) ** 2
                ) ** 0.5

                if distance < proximity_threshold:
                    count += 1
                    break  # One match per frame is enough

        return count
