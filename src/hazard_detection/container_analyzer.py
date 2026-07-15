"""
Container Analyzer for the Hazard Detection System.

Interprets YOLO detections for container hazard states including misalignment,
open doors, flipped orientation, and dangling containers.

Binary hazard classification:
- is_hazard=True: confirmed detection (>=2 frames above threshold, not suppressed)
- is_hazard=False: unconfirmed (<2 frames), below threshold, or suppressed

Requirements covered:
- 3.1-3.6: Container Misalignment Detection
- 4.1-4.5: Container Door State Detection
- 5.1-5.7: Container Orientation Hazard Detection
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from hazard_detection.models import (
    BBox,
    Detection,
    DiagnosticMetadata,
    FrameSequence,
    HazardEvent,
    ContainerAnalyzerConfig,
)
from hazard_detection.diagnostics import (
    get_logger,
    PerformanceTimer,
    DiagnosticDumper,
)
from cv.flow_analyzer import OpticalFlowAnalyzer


logger = get_logger("container_analyzer")


@dataclass
class _TemporalState:
    """Tracks per-detection temporal confirmation state."""
    hazard_type: str
    frame_indices: List[int] = field(default_factory=list)
    confidence: float = 0.0
    bbox: Optional[BBox] = None
    detection_class: str = ""
    flow_consistency_score: Optional[float] = None

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def is_confirmed(self) -> bool:
        """Confirmed if present in >=2 frames."""
        return self.frame_count >= 2


class ContainerAnalyzer:
    """
    Interprets YOLO detections for container hazard states.

    Detects:
    - Misalignment: "Container - Misaligned" with IoU-based disambiguation
    - Door open: "Container - Open" with loading operation suppression
    - Flipped: bbox height/width ratio exceeds threshold
    - Dangling: "Container - Picked" without adequate Crane overlap

    All hazards require temporal confirmation (>=2 frames) for is_hazard=True.
    """

    def __init__(
        self,
        flow_analyzer: OpticalFlowAnalyzer,
        config: ContainerAnalyzerConfig,
        dumper: Optional[DiagnosticDumper] = None,
    ):
        """
        Args:
            flow_analyzer: OpticalFlowAnalyzer for flow_consistency_score
            config: Container analyzer configuration thresholds
            dumper: Optional diagnostic dumper for intermediate state snapshots
        """
        self._flow_analyzer = flow_analyzer
        self._config = config
        self._dumper = dumper
        logger.info(
            "ContainerAnalyzer initialized",
            extra={
                "component": "container_analyzer",
                "extra_data": {
                    "confidence_threshold": config.confidence_threshold,
                    "flipped_aspect_ratio_threshold": config.flipped_aspect_ratio_threshold,
                    "safe_overlap_threshold": config.safe_overlap_threshold,
                    "ground_level_threshold": config.ground_level_threshold,
                    "motion_threshold": config.motion_threshold,
                    "iou_threshold": config.iou_threshold,
                },
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        detections_per_frame: List[List[Detection]],
        frames: FrameSequence,
    ) -> List[HazardEvent]:
        """
        Analyze container detections across a frame sequence.

        Args:
            detections_per_frame: List of detection lists, one per frame.
            frames: The FrameSequence providing raw frames for flow analysis.

        Returns:
            List of HazardEvent objects (is_hazard=True for confirmed hazards,
            is_hazard=False for unconfirmed/suppressed detections).
        """
        with PerformanceTimer("container_analysis", camera_id=frames.camera_id):
            events: List[HazardEvent] = []

            # Compute flow scores across frame pairs
            flow_scores = self._compute_flow_scores(frames)

            # Accumulate temporal state across frames
            misalignment_states: Dict[str, _TemporalState] = {}
            door_open_states: Dict[str, _TemporalState] = {}
            flipped_states: Dict[str, _TemporalState] = {}
            dangling_states: Dict[str, _TemporalState] = {}

            for frame_idx, frame_detections in enumerate(detections_per_frame):
                self._process_frame(
                    frame_idx=frame_idx,
                    detections=frame_detections,
                    all_detections=detections_per_frame,
                    flow_scores=flow_scores,
                    misalignment_states=misalignment_states,
                    door_open_states=door_open_states,
                    flipped_states=flipped_states,
                    dangling_states=dangling_states,
                )

            # Dump intermediate temporal state
            if self._dumper:
                self._dumper.dump(
                    stage="temporal_confirmation",
                    state={
                        "misalignment_count": len(misalignment_states),
                        "door_open_count": len(door_open_states),
                        "flipped_count": len(flipped_states),
                        "dangling_count": len(dangling_states),
                    },
                    camera_id=frames.camera_id,
                )

            # Emit events from accumulated states
            events.extend(
                self._emit_events(misalignment_states, frames.camera_id)
            )
            events.extend(
                self._emit_events(door_open_states, frames.camera_id)
            )
            events.extend(
                self._emit_events(flipped_states, frames.camera_id)
            )
            events.extend(
                self._emit_events(dangling_states, frames.camera_id)
            )

            logger.info(
                f"Container analysis complete: {len(events)} events "
                f"({sum(1 for e in events if e.is_hazard)} hazards)",
                extra={
                    "component": "container_analyzer",
                    "camera_id": frames.camera_id,
                },
            )
            return events

    # ------------------------------------------------------------------
    # Geometric Calculations
    # ------------------------------------------------------------------

    def _compute_iou(self, box_a: BBox, box_b: BBox) -> float:
        """
        Compute Intersection-over-Union between two bounding boxes.

        Converts center-format boxes to corner format, computes intersection
        area, and divides by union area.

        Returns:
            IoU value in [0.0, 1.0].
        """
        # Convert to corner format: (x1, y1, x2, y2)
        a_x1 = box_a.x_center - box_a.width / 2.0
        a_y1 = box_a.y_center - box_a.height / 2.0
        a_x2 = box_a.x_center + box_a.width / 2.0
        a_y2 = box_a.y_center + box_a.height / 2.0

        b_x1 = box_b.x_center - box_b.width / 2.0
        b_y1 = box_b.y_center - box_b.height / 2.0
        b_x2 = box_b.x_center + box_b.width / 2.0
        b_y2 = box_b.y_center + box_b.height / 2.0

        # Intersection
        inter_x1 = max(a_x1, b_x1)
        inter_y1 = max(a_y1, b_y1)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        area_a = box_a.width * box_a.height
        area_b = box_b.width * box_b.height
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        iou = inter_area / union_area
        return float(np.clip(iou, 0.0, 1.0))

    def _compute_ioa(self, box_inner: BBox, box_outer: BBox) -> float:
        """
        Compute Intersection-over-Area (area of box_inner) for crane overlap.

        Measures what fraction of the inner box is covered by the outer box.

        Returns:
            IoA value in [0.0, 1.0].
        """
        # Convert to corner format
        a_x1 = box_inner.x_center - box_inner.width / 2.0
        a_y1 = box_inner.y_center - box_inner.height / 2.0
        a_x2 = box_inner.x_center + box_inner.width / 2.0
        a_y2 = box_inner.y_center + box_inner.height / 2.0

        b_x1 = box_outer.x_center - box_outer.width / 2.0
        b_y1 = box_outer.y_center - box_outer.height / 2.0
        b_x2 = box_outer.x_center + box_outer.width / 2.0
        b_y2 = box_outer.y_center + box_outer.height / 2.0

        # Intersection
        inter_x1 = max(a_x1, b_x1)
        inter_y1 = max(a_y1, b_y1)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Area of inner box
        area_inner = box_inner.width * box_inner.height
        if area_inner <= 0.0:
            return 0.0

        ioa = inter_area / area_inner
        return float(np.clip(ioa, 0.0, 1.0))

    def _is_flipped(self, bbox: BBox) -> bool:
        """
        Check if a container's height/width ratio exceeds the flipped threshold.

        A normally-oriented shipping container is wider than it is tall.
        A flipped container has height > width beyond the configured ratio.

        Returns:
            True if the container appears flipped based on aspect ratio.
        """
        if bbox.width <= 0.0:
            return False
        ratio = bbox.height / bbox.width
        is_flipped = ratio > self._config.flipped_aspect_ratio_threshold

        logger.debug(
            f"Flipped check: h/w ratio={ratio:.3f}, "
            f"threshold={self._config.flipped_aspect_ratio_threshold}, "
            f"is_flipped={is_flipped}",
            extra={"component": "container_analyzer"},
        )
        return is_flipped

    def _is_dangling(
        self, picked_bbox: BBox, crane_bboxes: List[BBox]
    ) -> bool:
        """
        Check if a picked container has insufficient crane overlap
        and high vertical position (indicating dangerous dangling state).

        Conditions for dangling:
        (a) No Crane in frame, OR
        (b) IoA with nearest Crane < safe_overlap_threshold AND
            vertical midpoint above ground_level_threshold.

        Returns:
            True if the container is in a dangling state.
        """
        # (a) No crane in frame
        if not crane_bboxes:
            logger.debug(
                "Dangling check: no crane detected in frame",
                extra={"component": "container_analyzer"},
            )
            return True

        # (b) Find nearest crane by IoA and check overlap + vertical position
        max_ioa = 0.0
        for crane_bbox in crane_bboxes:
            ioa = self._compute_ioa(picked_bbox, crane_bbox)
            max_ioa = max(max_ioa, ioa)

        # Container midpoint is above ground level (lower y_center = higher)
        is_above_ground = picked_bbox.y_center < self._config.ground_level_threshold
        insufficient_overlap = max_ioa < self._config.safe_overlap_threshold

        is_dangling = insufficient_overlap and is_above_ground

        logger.debug(
            f"Dangling check: max_ioa={max_ioa:.3f}, "
            f"safe_overlap_threshold={self._config.safe_overlap_threshold}, "
            f"y_center={picked_bbox.y_center:.3f}, "
            f"ground_level_threshold={self._config.ground_level_threshold}, "
            f"is_dangling={is_dangling}",
            extra={"component": "container_analyzer"},
        )
        return is_dangling

    # ------------------------------------------------------------------
    # Flow Analysis Integration
    # ------------------------------------------------------------------

    def _compute_flow_scores(
        self, frames: FrameSequence
    ) -> List[Optional[float]]:
        """
        Compute flow consistency scores between consecutive frame pairs.

        Returns a list of scores (one per consecutive pair), or empty if
        fewer than 2 frames are available.
        """
        scores: List[Optional[float]] = []
        if frames.frame_count < 2:
            return scores

        for i in range(frames.frame_count - 1):
            try:
                flow_result = self._flow_analyzer.compute_flow(
                    frames.frames[i], frames.frames[i + 1]
                )
                score = self._flow_analyzer.get_flow_consistency_score(
                    flow_result
                )
                scores.append(score)
            except (ValueError, Exception) as e:
                logger.warning(
                    f"Flow computation failed for frame pair {i}-{i+1}: {e}",
                    extra={"component": "container_analyzer"},
                )
                scores.append(None)

        logger.debug(
            f"Flow scores computed: {scores}",
            extra={"component": "container_analyzer"},
        )
        return scores

    # ------------------------------------------------------------------
    # Per-Frame Processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame_idx: int,
        detections: List[Detection],
        all_detections: List[List[Detection]],
        flow_scores: List[Optional[float]],
        misalignment_states: Dict[str, _TemporalState],
        door_open_states: Dict[str, _TemporalState],
        flipped_states: Dict[str, _TemporalState],
        dangling_states: Dict[str, _TemporalState],
    ) -> None:
        """Process a single frame's detections and update temporal states."""

        # Gather crane bboxes in this frame for dangling check
        crane_bboxes = [
            d.bbox for d in detections if d.class_label == "Crane"
        ]

        # Gather stacked detections for disambiguation
        stacked_detections = [
            d for d in detections if d.class_label == "Container - Stacked"
        ]

        # Gather picked/crane detections for door suppression
        picked_detections = [
            d for d in detections if d.class_label == "Container - Picked"
        ]

        # Get flow score for this frame (based on pair index)
        flow_score = None
        if frame_idx < len(flow_scores):
            flow_score = flow_scores[frame_idx]

        for det in detections:
            self._process_detection(
                det=det,
                frame_idx=frame_idx,
                crane_bboxes=crane_bboxes,
                stacked_detections=stacked_detections,
                picked_detections=picked_detections,
                flow_score=flow_score,
                misalignment_states=misalignment_states,
                door_open_states=door_open_states,
                flipped_states=flipped_states,
                dangling_states=dangling_states,
            )

    def _process_detection(
        self,
        det: Detection,
        frame_idx: int,
        crane_bboxes: List[BBox],
        stacked_detections: List[Detection],
        picked_detections: List[Detection],
        flow_score: Optional[float],
        misalignment_states: Dict[str, _TemporalState],
        door_open_states: Dict[str, _TemporalState],
        flipped_states: Dict[str, _TemporalState],
        dangling_states: Dict[str, _TemporalState],
    ) -> None:
        """Route a single detection to the appropriate hazard handler."""

        if det.class_label == "Container - Misaligned":
            self._handle_misalignment(
                det, frame_idx, stacked_detections, flow_score,
                misalignment_states,
            )
        elif det.class_label == "Container - Open":
            self._handle_door_open(
                det, frame_idx, picked_detections, crane_bboxes,
                door_open_states,
            )
        elif det.class_label == "Container - Picked":
            self._handle_dangling(
                det, frame_idx, crane_bboxes, dangling_states,
            )

        # Check flipped for any container-class detection
        container_classes = {
            "Container - Misaligned", "Container - Open",
            "Container - Picked", "Container - Stacked",
            "Container - Separate", "Container - Reefer",
            "Container - Water Drop",
        }
        if det.class_label in container_classes:
            self._handle_flipped(det, frame_idx, flipped_states)

    # ------------------------------------------------------------------
    # Hazard-Specific Handlers
    # ------------------------------------------------------------------

    def _handle_misalignment(
        self,
        det: Detection,
        frame_idx: int,
        stacked_detections: List[Detection],
        flow_score: Optional[float],
        states: Dict[str, _TemporalState],
    ) -> None:
        """
        Handle misalignment detection with IoU-based disambiguation.

        Requirement 3.5: When both "Container - Misaligned" and
        "Container - Stacked" are predicted for overlapping bboxes
        (IoU >= 0.5), use the class with higher confidence.
        """
        if det.confidence < self._config.confidence_threshold:
            logger.debug(
                f"Misalignment below threshold: conf={det.confidence:.3f}",
                extra={"component": "container_analyzer"},
            )
            return

        # Disambiguate against stacked detections
        for stacked in stacked_detections:
            iou = self._compute_iou(det.bbox, stacked.bbox)
            if iou >= self._config.iou_threshold:
                # Overlapping: use higher confidence class
                if stacked.confidence > det.confidence:
                    logger.debug(
                        f"Misalignment suppressed by stacked: "
                        f"misaligned_conf={det.confidence:.3f}, "
                        f"stacked_conf={stacked.confidence:.3f}, "
                        f"iou={iou:.3f}",
                        extra={"component": "container_analyzer"},
                    )
                    return  # Stacked wins, skip this detection

        # Track temporal state using bbox center as key
        key = f"misaligned_{det.bbox.x_center:.3f}_{det.bbox.y_center:.3f}"
        if key not in states:
            states[key] = _TemporalState(
                hazard_type="container_misalignment",
                detection_class=det.class_label,
            )
        state = states[key]
        state.frame_indices.append(frame_idx)
        state.confidence = max(state.confidence, det.confidence)
        state.bbox = det.bbox
        state.flow_consistency_score = flow_score

        logger.debug(
            f"Misalignment tracked: key={key}, frames={state.frame_count}, "
            f"conf={state.confidence:.3f}, flow_score={flow_score}",
            extra={"component": "container_analyzer"},
        )

    def _handle_door_open(
        self,
        det: Detection,
        frame_idx: int,
        picked_detections: List[Detection],
        crane_bboxes: List[BBox],
        states: Dict[str, _TemporalState],
    ) -> None:
        """
        Handle door open detection with loading operation suppression.

        Requirement 4.4-4.5: If "Container - Open" overlaps with
        "Container - Picked" or "Crane" (IoU >= 0.5), suppress the
        door open event (loading operation in progress).
        """
        if det.confidence < self._config.confidence_threshold:
            logger.debug(
                f"Door open below threshold: conf={det.confidence:.3f}",
                extra={"component": "container_analyzer"},
            )
            return

        # Check for loading operation suppression
        # Overlap with "Container - Picked"
        for picked in picked_detections:
            iou = self._compute_iou(det.bbox, picked.bbox)
            if iou >= self._config.iou_threshold:
                logger.debug(
                    f"Door open suppressed (loading): "
                    f"overlap with Picked, iou={iou:.3f}",
                    extra={"component": "container_analyzer"},
                )
                return  # Suppressed by loading operation

        # Overlap with "Crane"
        for crane_bbox in crane_bboxes:
            iou = self._compute_iou(det.bbox, crane_bbox)
            if iou >= self._config.iou_threshold:
                logger.debug(
                    f"Door open suppressed (loading): "
                    f"overlap with Crane, iou={iou:.3f}",
                    extra={"component": "container_analyzer"},
                )
                return  # Suppressed by loading operation

        # Track temporal state
        key = f"door_open_{det.bbox.x_center:.3f}_{det.bbox.y_center:.3f}"
        if key not in states:
            states[key] = _TemporalState(
                hazard_type="container_door_open",
                detection_class=det.class_label,
            )
        state = states[key]
        state.frame_indices.append(frame_idx)
        state.confidence = max(state.confidence, det.confidence)
        state.bbox = det.bbox

        logger.debug(
            f"Door open tracked: key={key}, frames={state.frame_count}, "
            f"conf={state.confidence:.3f}",
            extra={"component": "container_analyzer"},
        )

    def _handle_flipped(
        self,
        det: Detection,
        frame_idx: int,
        states: Dict[str, _TemporalState],
    ) -> None:
        """
        Handle flipped container detection based on aspect ratio.

        Requirement 5.2: Height/width ratio exceeds configured threshold
        indicates a flipped container.
        """
        if det.confidence < self._config.confidence_threshold:
            return

        if not self._is_flipped(det.bbox):
            return

        # Track temporal state
        key = f"flipped_{det.bbox.x_center:.3f}_{det.bbox.y_center:.3f}"
        if key not in states:
            states[key] = _TemporalState(
                hazard_type="container_flipped",
                detection_class=det.class_label,
            )
        state = states[key]
        state.frame_indices.append(frame_idx)
        state.confidence = max(state.confidence, det.confidence)
        state.bbox = det.bbox

        logger.debug(
            f"Flipped tracked: key={key}, frames={state.frame_count}, "
            f"aspect_ratio={det.bbox.aspect_ratio:.3f}",
            extra={"component": "container_analyzer"},
        )

    def _handle_dangling(
        self,
        det: Detection,
        frame_idx: int,
        crane_bboxes: List[BBox],
        states: Dict[str, _TemporalState],
    ) -> None:
        """
        Handle dangling container detection.

        Requirement 5.3-5.4: "Container - Picked" with either no Crane
        or insufficient crane overlap AND high vertical position.
        """
        if det.confidence < self._config.confidence_threshold:
            return

        if not self._is_dangling(det.bbox, crane_bboxes):
            return

        # Track temporal state
        key = f"dangling_{det.bbox.x_center:.3f}_{det.bbox.y_center:.3f}"
        if key not in states:
            states[key] = _TemporalState(
                hazard_type="container_dangling",
                detection_class=det.class_label,
            )
        state = states[key]
        state.frame_indices.append(frame_idx)
        state.confidence = max(state.confidence, det.confidence)
        state.bbox = det.bbox

        logger.debug(
            f"Dangling tracked: key={key}, frames={state.frame_count}, "
            f"conf={state.confidence:.3f}",
            extra={"component": "container_analyzer"},
        )

    # ------------------------------------------------------------------
    # Event Emission
    # ------------------------------------------------------------------

    def _emit_events(
        self,
        states: Dict[str, _TemporalState],
        camera_id: str,
    ) -> List[HazardEvent]:
        """
        Convert temporal states into HazardEvent objects.

        is_hazard=True if confirmed (>=2 frames), False otherwise.
        """
        events: List[HazardEvent] = []

        for key, state in states.items():
            if state.bbox is None:
                continue

            is_hazard = state.is_confirmed

            metadata = DiagnosticMetadata(
                frame_index=state.frame_indices[0] if state.frame_indices else 0,
                detection_class=state.detection_class,
                frames_detected=state.frame_count,
                flow_consistency_score=state.flow_consistency_score,
            )

            event = HazardEvent(
                event_id=HazardEvent.generate_event_id(),
                hazard_type=state.hazard_type,
                camera_id=camera_id,
                timestamp=HazardEvent.generate_timestamp(),
                is_hazard=is_hazard,
                confidence=state.confidence,
                bbox=state.bbox,
                metadata=metadata,
            )

            if is_hazard:
                logger.info(
                    f"HAZARD confirmed: {state.hazard_type}, "
                    f"conf={state.confidence:.3f}, "
                    f"frames={state.frame_count}",
                    extra={
                        "component": "container_analyzer",
                        "camera_id": camera_id,
                    },
                )
            else:
                logger.debug(
                    f"Detection logged (not confirmed): {state.hazard_type}, "
                    f"conf={state.confidence:.3f}, "
                    f"frames={state.frame_count}",
                    extra={
                        "component": "container_analyzer",
                        "camera_id": camera_id,
                    },
                )

            events.append(event)

        return events
