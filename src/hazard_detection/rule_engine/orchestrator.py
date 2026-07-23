"""
Orchestrator for the Camera-Location-Aware Hazard Rules engine.

Ties everything together end-to-end for one camera's detection set: resolves
the camera name to a location type, loads the matching rule set, runs every
applicable check from check_rules_from_object_label.py, assembles
QualifiedHazardEvents, and writes the audit log. This file answers "for this
camera and this batch of detections, what actually happened?" — per the
module-structure split requested for this project (requirements.md
Requirement 9.2).

Design note on HumanDetector reuse (requirements.md Requirement 9.3): the
existing ContainerAnalyzer is reused directly here, since check_container()
is a pure gate on its already-computed HazardEvents (design.md). HumanDetector,
however, makes its own hazard decision (zone_violation) based on ZoneMap's
no_people/operation/dropoff zone types — a different concept from this
engine's Location_Rule_Set human_presence_policy. check_human()'s signature
(detection, rule_set) confirms the design intent: human/vehicle temporal
confirmation is the orchestrator's own responsibility here, generically
implemented (mirroring ContainerAnalyzer's confirmation pattern: >=2
consecutive frames for policy/zone-type hazards, immediate confirmation for
PPE violations), rather than delegated to HumanDetector's zone-map-specific
implementation.

Design note on decoupling from ContainerAnalyzer's import chain: this
module depends on rule_engine.interfaces.ContainerAnalyzerProtocol (a
structural Protocol), not hazard_detection.container_analyzer.ContainerAnalyzer
directly. ContainerAnalyzer transitively imports cv.flow_analyzer, which
imports a src.models.core.FlowResult module that does not exist anywhere
in this codebase — an unrelated, pre-existing break in that dependency
chain. Depending on the Protocol instead means this module (and its tests)
can be imported and exercised independently of that break, per
requirements.md Requirement 9.4 ("testable in isolation... accepting mock
Detection lists as input"). The real ContainerAnalyzer still satisfies the
Protocol and can be passed in wherever its own import chain is fixed.

Requirements covered: 7.1-7.7, 9.1-9.6
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hazard_detection.diagnostics import get_logger
from hazard_detection.models import BBox, Detection, FrameSequence, HazardEvent
from hazard_detection.rule_engine.audit_logger import (
    AuditLogger,
    OUTCOME_CHECK_DISABLED,
    OUTCOME_HAZARD_EMITTED,
    OUTCOME_NO_HAZARD,
    OUTCOME_POLICY_UNKNOWN,
    OUTCOME_SUPPRESSED,
)
from hazard_detection.rule_engine.camera_location_resolver import CameraLocationResolver
from hazard_detection.rule_engine.interfaces import ContainerAnalyzerProtocol
from hazard_detection.rule_engine.check_rules_from_object_label import (
    HUMAN_CLASSES,
    VEHICLE_CLASSES,
    check_container,
    check_human,
    check_tel_occupancy,
    check_tel_spot,
    check_vehicle,
)
from hazard_detection.rule_engine.rules import LocationRuleLoader, LocationRuleSet

logger = get_logger("rule_engine.orchestrator")

# Default confidence threshold and temporal confirmation requirement,
# matching the rest of the pipeline (hazard-detection-system spec).
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MIN_FRAMES_FOR_CONFIRMATION = 2

# HSSE could not provide a specific quantifiable unsafe-distance value
# ("pretty tight, not sure how to provide a meaningful data point") —
# this default is a documented placeholder, not a validated safety
# threshold (requirements.md Requirement 5.2).
DEFAULT_VEHICLE_UNSAFE_Y_THRESHOLD = 0.8

TEL_LOCATION_TYPES = {"TEL", "TELs"}


@dataclass
class QualifiedHazardEvent:
    """
    A HazardEvent enriched with the resolved Camera_Location_Type and the
    matched rule that triggered it (requirements.md Requirement 3.6).
    """

    event: HazardEvent
    camera_location_type: str
    matched_rule: str


@dataclass
class _TemporalTrack:
    """Tracks per-detection temporal confirmation state across frames."""

    detection_class: str
    frame_indices: List[int] = field(default_factory=list)
    confidence: float = 0.0
    bbox: Optional[BBox] = None

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    def is_confirmed(self, min_frames: int) -> bool:
        return self.frame_count >= min_frames


class HazardRuleOrchestrator:
    """
    End-to-end per-camera rule evaluation.

    Wires together CameraLocationResolver, LocationRuleLoader,
    check_rules_from_object_label.py, the existing ContainerAnalyzer, and
    the AuditLogger.
    """

    def __init__(
        self,
        resolver: CameraLocationResolver,
        rule_loader: LocationRuleLoader,
        container_analyzer: ContainerAnalyzerProtocol,
        audit_logger: AuditLogger,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        min_frames_for_confirmation: int = DEFAULT_MIN_FRAMES_FOR_CONFIRMATION,
        vehicle_unsafe_y_threshold: float = DEFAULT_VEHICLE_UNSAFE_Y_THRESHOLD,
        trucker_spot_polygons: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    ):
        """
        Args:
            resolver: CameraLocationResolver for camera-name parsing.
            rule_loader: LocationRuleLoader providing rule sets and the
                camera_name_overrides / camera_id_to_name mappings.
            container_analyzer: The existing ContainerAnalyzer (or any
                object satisfying ContainerAnalyzerProtocol), reused
                directly for visual container hazard analysis.
            audit_logger: AuditLogger for the JSON-lines audit trail.
            confidence_threshold: Minimum confidence for human/vehicle
                detections to be considered at all (Requirement 3.5, 5.4).
            min_frames_for_confirmation: Consecutive-frame count required
                to confirm a zone/vehicle/TEL-spot hazard (Requirement 3.5,
                5.4, 6.2). PPE violations are confirmed on any single
                qualifying frame, matching existing HumanDetector behavior.
            vehicle_unsafe_y_threshold: Tunable placeholder threshold for
                vehicle proximity checks (Requirement 5.2) — NOT a
                validated safety distance.
            trucker_spot_polygons: Optional mapping of camera_name ->
                trucker spot polygon vertices, for TEL/TELs cameras
                (Requirement 6.1, 6.5). Cameras absent from this mapping
                are treated fail-safe by check_tel_spot() (Requirement 6.4).
        """
        self._resolver = resolver
        self._rule_loader = rule_loader
        self._container_analyzer = container_analyzer
        self._audit_logger = audit_logger
        self._confidence_threshold = confidence_threshold
        self._min_frames_for_confirmation = min_frames_for_confirmation
        self._vehicle_unsafe_y_threshold = vehicle_unsafe_y_threshold
        self._trucker_spot_polygons = trucker_spot_polygons or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        camera_name: str,
        detections_per_frame: List[List[Detection]],
        frames: FrameSequence,
    ) -> List[QualifiedHazardEvent]:
        """
        Evaluate one camera's full detection set against its resolved
        location rules, end to end.

        Args:
            camera_name: The full Ocularis Camera_Name for this camera.
            detections_per_frame: List of detection lists, one per frame.
            frames: The FrameSequence (passed through to ContainerAnalyzer).

        Returns:
            List of QualifiedHazardEvent (both confirmed hazards and
            transient/logged-only events, matching existing pipeline
            convention — the caller/AlertDispatcher distinguishes via
            event.is_hazard).
        """
        location_type = self._resolver.resolve_with_override(
            camera_name, self._rule_loader.camera_name_overrides
        )
        rule_set = self._rule_loader.get_rule_set(location_type)

        qualified_events: List[QualifiedHazardEvent] = []

        qualified_events.extend(
            self._evaluate_human_and_ppe(camera_name, location_type, rule_set, detections_per_frame)
        )

        if rule_set.trucker_spot_check_enabled and location_type in TEL_LOCATION_TYPES:
            qualified_events.extend(
                self._evaluate_tel_spot(camera_name, location_type, rule_set, detections_per_frame)
            )
            qualified_events.extend(
                self._evaluate_tel_occupancy(camera_name, location_type, rule_set, detections_per_frame)
            )

        qualified_events.extend(
            self._evaluate_containers(camera_name, location_type, rule_set, detections_per_frame, frames)
        )

        qualified_events.extend(
            self._evaluate_vehicles(camera_name, location_type, rule_set, detections_per_frame)
        )

        return qualified_events

    # ------------------------------------------------------------------
    # Human / PPE evaluation
    # ------------------------------------------------------------------

    def _evaluate_human_and_ppe(
        self,
        camera_name: str,
        location_type: str,
        rule_set: LocationRuleSet,
        detections_per_frame: List[List[Detection]],
    ) -> List[QualifiedHazardEvent]:
        if rule_set.human_presence_policy == "unknown":
            for frame_idx, frame_detections in enumerate(detections_per_frame):
                for det in frame_detections:
                    if det.class_label in HUMAN_CLASSES:
                        self._log_audit(
                            camera_name, location_type, det, frame_idx,
                            rule_name="human_presence_policy_unknown",
                            outcome=OUTCOME_POLICY_UNKNOWN,
                        )
            return []

        events: List[QualifiedHazardEvent] = []

        # PPE violations confirm immediately (any single qualifying frame),
        # matching existing HumanDetector behavior (Requirement 3.2).
        ppe_emitted_frames: set = set()
        # Zone-violation-type outcomes require temporal confirmation.
        zone_tracks: Dict[str, _TemporalTrack] = {}

        for frame_idx, frame_detections in enumerate(detections_per_frame):
            for det in frame_detections:
                if det.class_label not in HUMAN_CLASSES:
                    continue

                if det.confidence < self._confidence_threshold:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="below_confidence_threshold",
                        outcome=OUTCOME_NO_HAZARD,
                    )
                    continue

                result = check_human(det, rule_set)
                if result is None:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="human_presence_compliant",
                        outcome=OUTCOME_NO_HAZARD,
                    )
                    continue

                hazard_type, matched_rule = result

                if hazard_type == "ppe_violation":
                    if frame_idx in ppe_emitted_frames:
                        continue
                    ppe_emitted_frames.add(frame_idx)
                    event = self._build_event(
                        hazard_type, camera_name, det, frame_idx, is_hazard=True, frames_detected=1,
                    )
                    events.append(QualifiedHazardEvent(event, location_type, matched_rule))
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name=matched_rule, outcome=OUTCOME_HAZARD_EMITTED,
                    )
                else:
                    key = self._temporal_key(det)
                    track = zone_tracks.setdefault(
                        key, _TemporalTrack(detection_class=det.class_label)
                    )
                    track.frame_indices.append(frame_idx)
                    track.confidence = max(track.confidence, det.confidence)
                    track.bbox = det.bbox

                    is_hazard = track.is_confirmed(self._min_frames_for_confirmation)
                    event = self._build_event(
                        hazard_type, camera_name, det, frame_idx,
                        is_hazard=is_hazard, frames_detected=track.frame_count,
                    )
                    events.append(QualifiedHazardEvent(event, location_type, matched_rule))
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name=matched_rule,
                        outcome=OUTCOME_HAZARD_EMITTED if is_hazard else OUTCOME_NO_HAZARD,
                    )

        return events

    # ------------------------------------------------------------------
    # TEL trucker spot evaluation
    # ------------------------------------------------------------------

    def _evaluate_tel_spot(
        self,
        camera_name: str,
        location_type: str,
        rule_set: LocationRuleSet,
        detections_per_frame: List[List[Detection]],
    ) -> List[QualifiedHazardEvent]:
        polygon = self._trucker_spot_polygons.get(camera_name)
        events: List[QualifiedHazardEvent] = []
        tracks: Dict[str, _TemporalTrack] = {}

        for frame_idx, frame_detections in enumerate(detections_per_frame):
            for det in frame_detections:
                if det.class_label not in HUMAN_CLASSES:
                    continue
                if det.confidence < self._confidence_threshold:
                    continue

                result = check_tel_spot(det, polygon)
                if result is None:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="tel_trucker_spot_inside",
                        outcome=OUTCOME_NO_HAZARD,
                    )
                    continue

                hazard_type, matched_rule = result
                key = self._temporal_key(det)
                track = tracks.setdefault(key, _TemporalTrack(detection_class=det.class_label))
                track.frame_indices.append(frame_idx)
                track.confidence = max(track.confidence, det.confidence)
                track.bbox = det.bbox

                is_hazard = track.is_confirmed(self._min_frames_for_confirmation)
                event = self._build_event(
                    hazard_type, camera_name, det, frame_idx,
                    is_hazard=is_hazard, frames_detected=track.frame_count,
                )
                events.append(QualifiedHazardEvent(event, location_type, matched_rule))
                self._log_audit(
                    camera_name, location_type, det, frame_idx,
                    rule_name=matched_rule,
                    outcome=OUTCOME_HAZARD_EMITTED if is_hazard else OUTCOME_NO_HAZARD,
                )

        return events

    # ------------------------------------------------------------------
    # TEL occupancy evaluation
    # ------------------------------------------------------------------

    def _evaluate_tel_occupancy(
        self,
        camera_name: str,
        location_type: str,
        rule_set: LocationRuleSet,
        detections_per_frame: List[List[Detection]],
    ) -> List[QualifiedHazardEvent]:
        events: List[QualifiedHazardEvent] = []

        for frame_idx, frame_detections in enumerate(detections_per_frame):
            confirmed_humans = [
                det for det in frame_detections
                if det.class_label in HUMAN_CLASSES and det.confidence >= self._confidence_threshold
            ]
            if not confirmed_humans:
                continue

            result = check_tel_occupancy(confirmed_humans, rule_set)
            if result is None:
                continue

            hazard_type, matched_rule = result
            # Represent the occupancy violation with the highest-confidence
            # human detection in this frame as the representative bbox.
            representative = max(confirmed_humans, key=lambda d: d.confidence)
            event = self._build_event(
                hazard_type, camera_name, representative, frame_idx,
                is_hazard=True, frames_detected=1,
            )
            events.append(QualifiedHazardEvent(event, location_type, matched_rule))
            self._log_audit(
                camera_name, location_type, representative, frame_idx,
                rule_name=matched_rule, outcome=OUTCOME_HAZARD_EMITTED,
            )

        return events

    # ------------------------------------------------------------------
    # Container evaluation
    # ------------------------------------------------------------------

    def _evaluate_containers(
        self,
        camera_name: str,
        location_type: str,
        rule_set: LocationRuleSet,
        detections_per_frame: List[List[Detection]],
        frames: FrameSequence,
    ) -> List[QualifiedHazardEvent]:
        analyzer_events = self._container_analyzer.analyze(detections_per_frame, frames)

        events: List[QualifiedHazardEvent] = []
        for analyzer_event in analyzer_events:
            # Use the metadata's detection_class as a stand-in Detection for
            # check_container()'s interface (it only inspects analyzer_result).
            pseudo_detection = Detection(
                bbox=analyzer_event.bbox,
                class_label=analyzer_event.metadata.detection_class,
                confidence=analyzer_event.confidence,
            )
            result = check_container(pseudo_detection, rule_set, analyzer_event)

            if result is None:
                check_name = self._container_check_name_for_hazard_type(analyzer_event.hazard_type)
                if check_name and check_name in rule_set.container_checks_suppressed:
                    outcome = OUTCOME_SUPPRESSED
                else:
                    outcome = OUTCOME_CHECK_DISABLED
                self._log_audit(
                    camera_name, location_type, pseudo_detection,
                    analyzer_event.metadata.frame_index,
                    rule_name=f"container_{check_name}_check" if check_name else "container_check",
                    outcome=outcome,
                )
                continue

            hazard_type, matched_rule = result
            events.append(QualifiedHazardEvent(analyzer_event, location_type, matched_rule))
            self._log_audit(
                camera_name, location_type, pseudo_detection,
                analyzer_event.metadata.frame_index,
                rule_name=matched_rule,
                outcome=OUTCOME_HAZARD_EMITTED if analyzer_event.is_hazard else OUTCOME_NO_HAZARD,
            )

        return events

    @staticmethod
    def _container_check_name_for_hazard_type(hazard_type: str) -> Optional[str]:
        mapping = {
            "container_misalignment": "misalignment",
            "container_door_open": "open_doors",
            "container_flipped": "flipped",
            "container_dangling": "dangling",
        }
        return mapping.get(hazard_type)

    # ------------------------------------------------------------------
    # Vehicle evaluation
    # ------------------------------------------------------------------

    def _evaluate_vehicles(
        self,
        camera_name: str,
        location_type: str,
        rule_set: LocationRuleSet,
        detections_per_frame: List[List[Detection]],
    ) -> List[QualifiedHazardEvent]:
        events: List[QualifiedHazardEvent] = []
        tracks: Dict[str, _TemporalTrack] = {}

        for frame_idx, frame_detections in enumerate(detections_per_frame):
            for det in frame_detections:
                if det.class_label not in VEHICLE_CLASSES:
                    continue

                if not rule_set.vehicle_checks_enabled:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="vehicle_checks_disabled",
                        outcome=OUTCOME_CHECK_DISABLED,
                    )
                    continue

                if det.confidence < self._confidence_threshold:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="below_confidence_threshold",
                        outcome=OUTCOME_NO_HAZARD,
                    )
                    continue

                result = check_vehicle(det, rule_set, self._vehicle_unsafe_y_threshold)
                if result is None:
                    self._log_audit(
                        camera_name, location_type, det, frame_idx,
                        rule_name="vehicle_proximity_check",
                        outcome=OUTCOME_NO_HAZARD,
                    )
                    continue

                hazard_type, matched_rule = result
                key = self._temporal_key(det)
                track = tracks.setdefault(key, _TemporalTrack(detection_class=det.class_label))
                track.frame_indices.append(frame_idx)
                track.confidence = max(track.confidence, det.confidence)
                track.bbox = det.bbox

                is_hazard = track.is_confirmed(self._min_frames_for_confirmation)
                event = self._build_event(
                    hazard_type, camera_name, det, frame_idx,
                    is_hazard=is_hazard, frames_detected=track.frame_count,
                )
                events.append(QualifiedHazardEvent(event, location_type, matched_rule))
                self._log_audit(
                    camera_name, location_type, det, frame_idx,
                    rule_name=matched_rule,
                    outcome=OUTCOME_HAZARD_EMITTED if is_hazard else OUTCOME_NO_HAZARD,
                )

        return events

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_key(det: Detection) -> str:
        """
        Build a coarse proximity key for grouping the same real-world
        detection across frames, matching ContainerAnalyzer's convention.
        """
        return f"{det.class_label}_{det.bbox.x_center:.2f}_{det.bbox.y_center:.2f}"

    @staticmethod
    def _build_event(
        hazard_type: str,
        camera_name: str,
        det: Detection,
        frame_idx: int,
        is_hazard: bool,
        frames_detected: int,
    ) -> HazardEvent:
        from hazard_detection.models import DiagnosticMetadata

        return HazardEvent(
            event_id=HazardEvent.generate_event_id(),
            hazard_type=hazard_type,
            camera_id=camera_name,
            timestamp=HazardEvent.generate_timestamp(),
            is_hazard=is_hazard,
            confidence=det.confidence,
            bbox=det.bbox,
            metadata=DiagnosticMetadata(
                frame_index=frame_idx,
                detection_class=det.class_label,
                frames_detected=frames_detected,
            ),
        )

    def _log_audit(
        self,
        camera_name: str,
        location_type: str,
        det: Detection,
        frame_idx: int,
        rule_name: str,
        outcome: str,
    ) -> None:
        self._audit_logger.log_evaluation(
            camera_name=camera_name,
            location_type=location_type,
            detection_class=det.class_label,
            confidence=det.confidence,
            bbox=det.bbox,
            rule_name=rule_name,
            outcome=outcome,
            frame_index=frame_idx,
        )
