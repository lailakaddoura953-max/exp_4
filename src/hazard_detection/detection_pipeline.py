"""
Detection Pipeline Orchestrator for the Hazard Detection System.

Coordinates all pipeline components (FrameSampler, YOLODetector, HumanDetector,
ContainerAnalyzer, AlertDispatcher, CameraSwitcher) to process cameras
sequentially in a configurable order. Each camera goes through the full
pipeline: sample → detect → analyze → dispatch.

Features:
- Sequential camera processing from YAML-configured order
- Per-camera timeout enforcement (default 30s), skips on timeout
- Exception handling: logs error (module name + camera_id), skips to next camera
- Continuous cycling until shutdown via threading.Event
- Performance timing breakdown per stage
- Pipeline state snapshots between stages for debugging
- Diagnostic summary at end of each camera cycle

Requirements covered:
- 6.1: Process cameras one at a time in configurable sequence
- 6.2: Invoke Frame_Sampler → YOLO_Detector → Human_Detector/Container_Analyzer → Alert_Dispatcher
- 6.3: Per-camera timeout (default 30s) before transitioning to next camera
- 6.4: Log start and end time for each camera's processing cycle
- 6.5: On exception, log error with module name and camera_id, skip to next camera
- 6.6: Restart from first camera after completing sequence, cycle continuously
- 6.7: Terminate on timeout, log warning with camera_id and elapsed time
"""

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from hazard_detection.alert_dispatcher import AlertDispatcher
from hazard_detection.camera_switcher import CameraSwitcher
from hazard_detection.diagnostics import (
    DiagnosticDumper,
    PerformanceTimer,
    PipelineTracer,
    get_logger,
)
from hazard_detection.human_detector import HumanDetector
from hazard_detection.models import HazardEvent, PipelineConfig
from hazard_detection.rule_engine.interfaces import (
    ContainerAnalyzerProtocol,
    FrameSamplerProtocol,
)
from hazard_detection.rule_engine.orchestrator import HazardRuleOrchestrator
from hazard_detection.yolo_detector import YOLODetector

# NOTE: ContainerAnalyzer and FrameSampler are intentionally NOT imported
# directly here (only their structural interfaces are). ContainerAnalyzer
# transitively imports cv.flow_analyzer -> src.models.core.FlowResult, and
# FrameSampler transitively imports acquisition.frame_acquisition ->
# src.models.core.SynchronizedFrameBatch. Neither src.models.core module
# exists anywhere in this codebase -- a pre-existing, unrelated break in
# both dependency chains. Depending on the Protocols means DetectionPipeline
# can still be imported, constructed, and unit tested (with mocks/stubs in
# place of either component) independently of that break. Callers with a
# working import chain for the real classes can still pass them in; both
# satisfy their respective Protocol.

logger = get_logger("detection_pipeline")


class DetectionPipeline:
    """
    Orchestrates the full hazard detection pipeline across multiple cameras.

    Coordinates FrameSampler → YOLODetector → HumanDetector + ContainerAnalyzer
    → AlertDispatcher for each camera in sequence, cycling continuously until
    a shutdown event is signaled.

    Args:
        config: PipelineConfig with camera_sequence and per_camera_timeout_seconds.
        frame_sampler: FrameSampler instance (or any object satisfying
            FrameSamplerProtocol) for frame acquisition.
        yolo_detector: YOLODetector instance for object detection.
        human_detector: HumanDetector instance for zone/PPE violation detection.
        container_analyzer: ContainerAnalyzer instance (or any object
            satisfying ContainerAnalyzerProtocol) for container hazard detection.
        alert_dispatcher: AlertDispatcher instance for alert routing.
        camera_switcher: CameraSwitcher instance for camera transitions.
        shutdown_event: threading.Event used to signal graceful shutdown.
        tracer: Optional PipelineTracer for execution tracing.
        dumper: Optional DiagnosticDumper for intermediate state snapshots.
        hazard_rule_orchestrator: Optional HazardRuleOrchestrator from the
            camera-location-hazard-rules spec. When provided, it REPLACES
            the direct human_detector.analyze() / container_analyzer.analyze()
            calls for stages 4-5 with a single orchestrator.evaluate() call
            that applies location-aware rules (see rule_engine/orchestrator.py).
            When None (the default), the pipeline behaves exactly as before —
            human_detector and container_analyzer are invoked directly and
            independently, preserving existing exception-isolation behavior
            between the two stages. This is an additive, opt-in integration:
            deployments that haven't configured location rules are unaffected.
        get_camera_name: Optional callable mapping camera_id -> the full
            Ocularis Camera_Name string, used only when hazard_rule_orchestrator
            is provided (the orchestrator resolves location type from the
            camera name, not the short camera_id). Defaults to the identity
            function (camera_id used as-is) if not provided.
    """

    def __init__(
        self,
        config: PipelineConfig,
        frame_sampler: FrameSamplerProtocol,
        yolo_detector: YOLODetector,
        human_detector: HumanDetector,
        container_analyzer: ContainerAnalyzerProtocol,
        alert_dispatcher: AlertDispatcher,
        camera_switcher: CameraSwitcher,
        shutdown_event: Optional[threading.Event] = None,
        tracer: Optional[PipelineTracer] = None,
        dumper: Optional[DiagnosticDumper] = None,
        hazard_rule_orchestrator: Optional[HazardRuleOrchestrator] = None,
        get_camera_name: Optional[Callable[[str], str]] = None,
    ):
        self._config = config
        self._frame_sampler = frame_sampler
        self._yolo_detector = yolo_detector
        self._human_detector = human_detector
        self._container_analyzer = container_analyzer
        self._alert_dispatcher = alert_dispatcher
        self._camera_switcher = camera_switcher
        self._shutdown_event = shutdown_event or threading.Event()
        self._tracer = tracer
        self._dumper = dumper
        self._hazard_rule_orchestrator = hazard_rule_orchestrator
        self._get_camera_name = get_camera_name or (lambda camera_id: camera_id)

        # Pipeline statistics
        self._cycle_count: int = 0
        self._total_detections: int = 0
        self._total_hazards: int = 0

        logger.info(
            "DetectionPipeline initialized",
            extra={
                "component": "detection_pipeline",
                "extra_data": {
                    "camera_sequence": config.camera_sequence,
                    "per_camera_timeout_seconds": config.per_camera_timeout_seconds,
                    "camera_count": len(config.camera_sequence),
                },
            },
        )

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._config

    @property
    def cycle_count(self) -> int:
        """Return the number of completed full camera cycles."""
        return self._cycle_count

    @property
    def shutdown_event(self) -> threading.Event:
        """Return the shutdown event."""
        return self._shutdown_event

    def run(self) -> None:
        """
        Main pipeline loop: cycle through cameras continuously until shutdown.

        Processes each camera in the configured sequence, then restarts from
        the first camera. Continues until the shutdown_event is set.

        Requirement 6.6: Restart from first camera after completing sequence.
        """
        logger.info(
            "Detection pipeline starting continuous operation",
            extra={"component": "detection_pipeline"},
        )

        while not self._shutdown_event.is_set():
            cycle_start = time.perf_counter()

            # Process all cameras in sequence
            cycle_detections = 0
            cycle_hazards = 0
            camera_timings: Dict[str, float] = {}

            for camera_id in self._config.camera_sequence:
                if self._shutdown_event.is_set():
                    logger.info(
                        "Shutdown signaled during camera cycle, exiting",
                        extra={"component": "detection_pipeline"},
                    )
                    break

                # Process single camera with timeout enforcement
                camera_start = time.perf_counter()
                events = self._process_camera_with_timeout(camera_id)
                camera_elapsed = (time.perf_counter() - camera_start) * 1000.0
                camera_timings[camera_id] = camera_elapsed

                # Accumulate stats
                if events:
                    cycle_detections += len(events)
                    cycle_hazards += sum(1 for e in events if e.is_hazard)

            # Cycle complete — emit diagnostic summary
            cycle_elapsed = (time.perf_counter() - cycle_start) * 1000.0
            self._cycle_count += 1
            self._total_detections += cycle_detections
            self._total_hazards += cycle_hazards

            self._emit_cycle_summary(
                cycle_detections, cycle_hazards, cycle_elapsed, camera_timings
            )

        logger.info(
            "Detection pipeline shut down gracefully",
            extra={
                "component": "detection_pipeline",
                "extra_data": {
                    "total_cycles": self._cycle_count,
                    "total_detections": self._total_detections,
                    "total_hazards": self._total_hazards,
                },
            },
        )

    def process_camera(self, camera_id: str) -> List[HazardEvent]:
        """
        Process a single camera through the full pipeline.

        Pipeline stages: sample → detect → analyze (human + container) → dispatch.

        Logs start/end times, captures performance timing per stage, and dumps
        intermediate state between stages for debugging.

        Requirement 6.2: Full pipeline invocation order.
        Requirement 6.4: Log start and end time for each camera cycle.
        Requirement 6.5: On exception, log error with module name and camera_id.

        Args:
            camera_id: The camera identifier to process.

        Returns:
            List of HazardEvent objects produced during processing.
        """
        all_events: List[HazardEvent] = []
        stage_timings: Dict[str, float] = {}

        # Log start time (Req 6.4)
        start_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        logger.info(
            f"Camera processing started: '{camera_id}'",
            extra={
                "component": "detection_pipeline",
                "camera_id": camera_id,
                "extra_data": {"start_timestamp": start_timestamp},
            },
        )

        # Start pipeline trace
        if self._tracer:
            self._tracer.start_trace(camera_id=camera_id)

        # --- Stage 1: Camera Transition ---
        try:
            if self._tracer:
                self._tracer.enter_module(
                    "camera_switcher", input_info={"target_camera_id": camera_id}
                )

            with PerformanceTimer("camera_transition", camera_id=camera_id) as timer:
                transition_ok = self._camera_switcher.transition(camera_id)

            stage_timings["camera_transition"] = timer.elapsed_ms

            if self._tracer:
                self._tracer.exit_module(
                    "camera_switcher",
                    output_info={"success": transition_ok},
                )

            if not transition_ok:
                logger.warning(
                    f"Camera transition failed for '{camera_id}', skipping",
                    extra={
                        "component": "detection_pipeline",
                        "camera_id": camera_id,
                    },
                )
                if self._tracer:
                    self._tracer.end_trace()
                return all_events

        except Exception as exc:
            self._handle_stage_exception("camera_switcher", camera_id, exc)
            if self._tracer:
                self._tracer.exit_module(
                    "camera_switcher", output_info={"error": str(exc)}
                )
                self._tracer.end_trace()
            return all_events

        # --- Stage 2: Frame Sampling ---
        frame_sequence = None
        try:
            if self._tracer:
                self._tracer.enter_module(
                    "frame_sampler", input_info={"camera_id": camera_id}
                )

            with PerformanceTimer("frame_sampling", camera_id=camera_id) as timer:
                frame_sequence = self._frame_sampler.sample(camera_id)

            stage_timings["frame_sampling"] = timer.elapsed_ms

            if self._tracer:
                self._tracer.exit_module(
                    "frame_sampler",
                    output_info={
                        "frame_count": frame_sequence.frame_count
                        if frame_sequence
                        else 0,
                        "success": frame_sequence is not None,
                    },
                )

            # Dump state after sampling
            if self._dumper and frame_sequence:
                self._dumper.dump(
                    stage="post_sampling",
                    state={
                        "frame_count": frame_sequence.frame_count,
                        "timestamps": frame_sequence.timestamps,
                    },
                    camera_id=camera_id,
                )

            if frame_sequence is None:
                logger.warning(
                    f"Frame sampling returned None for camera '{camera_id}', skipping",
                    extra={
                        "component": "detection_pipeline",
                        "camera_id": camera_id,
                    },
                )
                if self._tracer:
                    self._tracer.end_trace()
                return all_events

        except Exception as exc:
            self._handle_stage_exception("frame_sampler", camera_id, exc)
            if self._tracer:
                self._tracer.exit_module(
                    "frame_sampler", output_info={"error": str(exc)}
                )
                self._tracer.end_trace()
            return all_events

        # --- Stage 3: YOLO Detection (Inference) ---
        detections_per_frame = None
        try:
            if self._tracer:
                self._tracer.enter_module(
                    "yolo_detector",
                    input_info={"frame_count": frame_sequence.frame_count},
                )

            with PerformanceTimer("yolo_inference", camera_id=camera_id) as timer:
                detections_per_frame = self._yolo_detector.detect(frame_sequence)

            stage_timings["yolo_inference"] = timer.elapsed_ms

            total_dets = sum(len(d) for d in detections_per_frame)
            if self._tracer:
                self._tracer.exit_module(
                    "yolo_detector",
                    output_info={
                        "total_detections": total_dets,
                        "detections_per_frame": [len(d) for d in detections_per_frame],
                    },
                )

            # Dump state after detection
            if self._dumper:
                self._dumper.dump(
                    stage="post_detection",
                    state={
                        "total_detections": total_dets,
                        "detections_per_frame": [len(d) for d in detections_per_frame],
                    },
                    camera_id=camera_id,
                )

        except Exception as exc:
            self._handle_stage_exception("yolo_detector", camera_id, exc)
            if self._tracer:
                self._tracer.exit_module(
                    "yolo_detector", output_info={"error": str(exc)}
                )
                self._tracer.end_trace()
            # Release frame data before returning
            self._frame_sampler.release()
            return all_events

        if self._hazard_rule_orchestrator is not None:
            # --- Stages 4-5 (combined): Location-Aware Rule Evaluation ---
            # Replaces the direct human_detector.analyze() / container_analyzer.analyze()
            # calls below with a single orchestrator.evaluate() call, per the
            # camera-location-hazard-rules spec (Requirement 9.1). Opt-in only —
            # see the hazard_rule_orchestrator constructor docstring.
            try:
                camera_name = self._get_camera_name(camera_id)

                if self._tracer:
                    self._tracer.enter_module(
                        "hazard_rule_orchestrator",
                        input_info={
                            "frames_to_analyze": len(detections_per_frame),
                            "camera_name": camera_name,
                        },
                    )

                with PerformanceTimer("hazard_rule_evaluation", camera_id=camera_id) as timer:
                    qualified_events = self._hazard_rule_orchestrator.evaluate(
                        camera_name, detections_per_frame, frame_sequence
                    )

                stage_timings["hazard_rule_evaluation"] = timer.elapsed_ms
                rule_events = [qe.event for qe in qualified_events]

                if self._tracer:
                    self._tracer.exit_module(
                        "hazard_rule_orchestrator",
                        output_info={
                            "events_produced": len(rule_events),
                            "hazards": sum(1 for e in rule_events if e.is_hazard),
                        },
                    )

                all_events.extend(rule_events)

                if self._dumper:
                    self._dumper.dump(
                        stage="post_hazard_rule_evaluation",
                        state={
                            "rule_events_count": len(rule_events),
                            "rule_hazards": sum(1 for e in rule_events if e.is_hazard),
                        },
                        camera_id=camera_id,
                    )

            except Exception as exc:
                self._handle_stage_exception("hazard_rule_orchestrator", camera_id, exc)
                if self._tracer:
                    self._tracer.exit_module(
                        "hazard_rule_orchestrator", output_info={"error": str(exc)}
                    )

        else:
            # --- Stage 4: Human Detection Analysis ---
            try:
                if self._tracer:
                    self._tracer.enter_module(
                        "human_detector",
                        input_info={
                            "frames_to_analyze": len(detections_per_frame),
                        },
                    )

                with PerformanceTimer("human_analysis", camera_id=camera_id) as timer:
                    human_events = self._human_detector.analyze(
                        detections_per_frame, camera_id
                    )

                stage_timings["human_analysis"] = timer.elapsed_ms

                if self._tracer:
                    self._tracer.exit_module(
                        "human_detector",
                        output_info={
                            "events_produced": len(human_events),
                            "hazards": sum(1 for e in human_events if e.is_hazard),
                        },
                    )

                all_events.extend(human_events)

                # Dump state after human analysis
                if self._dumper:
                    self._dumper.dump(
                        stage="post_human_analysis",
                        state={
                            "human_events_count": len(human_events),
                            "human_hazards": sum(1 for e in human_events if e.is_hazard),
                        },
                        camera_id=camera_id,
                    )

            except Exception as exc:
                self._handle_stage_exception("human_detector", camera_id, exc)
                if self._tracer:
                    self._tracer.exit_module(
                        "human_detector", output_info={"error": str(exc)}
                    )

            # --- Stage 5: Container Analysis ---
            try:
                if self._tracer:
                    self._tracer.enter_module(
                        "container_analyzer",
                        input_info={
                            "frames_to_analyze": len(detections_per_frame),
                        },
                    )

                with PerformanceTimer("container_analysis", camera_id=camera_id) as timer:
                    container_events = self._container_analyzer.analyze(
                        detections_per_frame, frame_sequence
                    )

                stage_timings["container_analysis"] = timer.elapsed_ms

                if self._tracer:
                    self._tracer.exit_module(
                        "container_analyzer",
                        output_info={
                            "events_produced": len(container_events),
                            "hazards": sum(1 for e in container_events if e.is_hazard),
                        },
                    )

                all_events.extend(container_events)

                # Dump state after container analysis
                if self._dumper:
                    self._dumper.dump(
                        stage="post_container_analysis",
                        state={
                            "container_events_count": len(container_events),
                            "container_hazards": sum(
                                1 for e in container_events if e.is_hazard
                            ),
                        },
                        camera_id=camera_id,
                    )

            except Exception as exc:
                self._handle_stage_exception("container_analyzer", camera_id, exc)
                if self._tracer:
                    self._tracer.exit_module(
                        "container_analyzer", output_info={"error": str(exc)}
                    )

        # --- Stage 6: Alert Dispatch ---
        try:
            if self._tracer:
                self._tracer.enter_module(
                    "alert_dispatcher",
                    input_info={"events_to_dispatch": len(all_events)},
                )

            with PerformanceTimer("alert_dispatch", camera_id=camera_id) as timer:
                for event in all_events:
                    self._alert_dispatcher.dispatch(event)

            stage_timings["alert_dispatch"] = timer.elapsed_ms

            if self._tracer:
                self._tracer.exit_module(
                    "alert_dispatcher",
                    output_info={
                        "events_dispatched": len(all_events),
                        "hazards_dispatched": sum(
                            1 for e in all_events if e.is_hazard
                        ),
                    },
                )

        except Exception as exc:
            self._handle_stage_exception("alert_dispatcher", camera_id, exc)
            if self._tracer:
                self._tracer.exit_module(
                    "alert_dispatcher", output_info={"error": str(exc)}
                )

        # Release frame data after processing (Req 1.3)
        self._frame_sampler.release()

        # End trace and log end time
        if self._tracer:
            self._tracer.end_trace()

        end_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        total_time_ms = sum(stage_timings.values())

        logger.info(
            f"Camera processing complete: '{camera_id}' — "
            f"{len(all_events)} events ({sum(1 for e in all_events if e.is_hazard)} hazards), "
            f"total time: {total_time_ms:.1f}ms",
            extra={
                "component": "detection_pipeline",
                "camera_id": camera_id,
                "extra_data": {
                    "end_timestamp": end_timestamp,
                    "stage_timings_ms": stage_timings,
                    "total_events": len(all_events),
                    "total_hazards": sum(1 for e in all_events if e.is_hazard),
                },
            },
        )

        return all_events

    # -------------------------------------------------------------------------
    # Timeout Enforcement
    # -------------------------------------------------------------------------

    def _process_camera_with_timeout(
        self, camera_id: str
    ) -> List[HazardEvent]:
        """
        Process a camera with per-camera timeout enforcement.

        Uses a thread to run process_camera and enforces the configured
        timeout. If the timeout is exceeded, logs a warning and returns
        an empty event list.

        Requirement 6.3: Complete processing within per-camera timeout.
        Requirement 6.7: Terminate on timeout, log warning with camera_id
        and elapsed time.

        Args:
            camera_id: The camera to process.

        Returns:
            List of HazardEvent objects, or empty list on timeout.
        """
        result: List[HazardEvent] = []
        exception_holder: List[Optional[Exception]] = [None]

        def _worker():
            nonlocal result
            try:
                result = self.process_camera(camera_id)
            except Exception as exc:
                exception_holder[0] = exc

        worker_thread = threading.Thread(target=_worker, daemon=True)
        start_time = time.perf_counter()
        worker_thread.start()
        worker_thread.join(timeout=self._config.per_camera_timeout_seconds)

        elapsed_seconds = time.perf_counter() - start_time

        if worker_thread.is_alive():
            # Timeout exceeded (Req 6.7)
            elapsed_ms = elapsed_seconds * 1000.0
            logger.warning(
                f"Per-camera timeout exceeded for camera '{camera_id}': "
                f"elapsed={elapsed_ms:.1f}ms, "
                f"timeout={self._config.per_camera_timeout_seconds}s. "
                f"Skipping to next camera.",
                extra={
                    "component": "detection_pipeline",
                    "camera_id": camera_id,
                    "extra_data": {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "timeout_seconds": self._config.per_camera_timeout_seconds,
                    },
                },
            )
            return []

        if exception_holder[0] is not None:
            # Unexpected exception escaped process_camera
            logger.error(
                f"Unhandled exception processing camera '{camera_id}': "
                f"{exception_holder[0]}",
                extra={
                    "component": "detection_pipeline",
                    "camera_id": camera_id,
                },
                exc_info=exception_holder[0],
            )
            return []

        return result

    # -------------------------------------------------------------------------
    # Error Handling
    # -------------------------------------------------------------------------

    def _handle_stage_exception(
        self, module_name: str, camera_id: str, exc: Exception
    ) -> None:
        """
        Log an exception from a pipeline stage.

        Requirement 6.5: Log error with module name and camera identifier,
        then skip to next camera.

        Args:
            module_name: Name of the module that raised the exception.
            camera_id: Camera being processed when the error occurred.
            exc: The exception that was raised.
        """
        logger.error(
            f"Pipeline stage exception: module='{module_name}', "
            f"camera='{camera_id}', error='{exc}'",
            extra={
                "component": "detection_pipeline",
                "camera_id": camera_id,
                "extra_data": {
                    "module": module_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            },
            exc_info=True,
        )

    # -------------------------------------------------------------------------
    # Diagnostic Summary
    # -------------------------------------------------------------------------

    def _emit_cycle_summary(
        self,
        cycle_detections: int,
        cycle_hazards: int,
        cycle_elapsed_ms: float,
        camera_timings: Dict[str, float],
    ) -> None:
        """
        Emit a diagnostic summary at the end of each camera cycle.

        Includes total detections, hazards found, and timing breakdown.

        Args:
            cycle_detections: Total detection events in this cycle.
            cycle_hazards: Total confirmed hazards in this cycle.
            cycle_elapsed_ms: Total cycle elapsed time in milliseconds.
            camera_timings: Per-camera timing in milliseconds.
        """
        logger.info(
            f"Cycle {self._cycle_count} complete: "
            f"{cycle_detections} detections, {cycle_hazards} hazards, "
            f"elapsed={cycle_elapsed_ms:.1f}ms",
            extra={
                "component": "detection_pipeline",
                "extra_data": {
                    "cycle_number": self._cycle_count,
                    "total_detections": cycle_detections,
                    "total_hazards": cycle_hazards,
                    "cycle_elapsed_ms": round(cycle_elapsed_ms, 3),
                    "camera_timings_ms": camera_timings,
                    "cumulative_detections": self._total_detections,
                    "cumulative_hazards": self._total_hazards,
                },
            },
        )

        # Dump cycle summary as diagnostic snapshot
        if self._dumper:
            self._dumper.dump(
                stage="cycle_summary",
                state={
                    "cycle_number": self._cycle_count,
                    "total_detections": cycle_detections,
                    "total_hazards": cycle_hazards,
                    "cycle_elapsed_ms": round(cycle_elapsed_ms, 3),
                    "camera_timings_ms": camera_timings,
                },
            )

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return accumulated pipeline statistics."""
        return {
            "cycle_count": self._cycle_count,
            "total_detections": self._total_detections,
            "total_hazards": self._total_hazards,
            "camera_sequence": self._config.camera_sequence,
            "per_camera_timeout_seconds": self._config.per_camera_timeout_seconds,
        }
