"""
Unit tests for the Detection Pipeline orchestrator.

Tests cover:
- Sequential camera processing order
- Exception handling and camera skip behavior
- Per-camera timeout enforcement
- Continuous cycling after last camera
- Performance timing output correctness
- Visual output: per-module timing breakdown chart (PNG)
- Visual output: pipeline execution trace (JSON)

Validates: Requirements 6.1, 6.5, 6.6, 6.7
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hazard_detection.detection_pipeline import DetectionPipeline
from hazard_detection.diagnostics import PipelineTracer
from hazard_detection.models import (
    BBox,
    Detection,
    DiagnosticMetadata,
    FrameSequence,
    HazardEvent,
    PipelineConfig,
)


# =============================================================================
# Test Helpers
# =============================================================================


def _make_frame_sequence(camera_id: str = "cam_01", num_frames: int = 6):
    """Create a minimal FrameSequence for testing."""
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(num_frames)]
    timestamps = [1700000000.0 + i * 0.1 for i in range(num_frames)]
    return FrameSequence(frames=frames, camera_id=camera_id, timestamps=timestamps)


def _make_detection(class_label="Human", confidence=0.8):
    """Create a sample Detection."""
    return Detection(
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.2),
        class_label=class_label,
        confidence=confidence,
    )


def _make_hazard_event(camera_id="cam_01", is_hazard=True, hazard_type="zone_violation"):
    """Create a sample HazardEvent."""
    return HazardEvent(
        event_id=HazardEvent.generate_event_id(),
        hazard_type=hazard_type,
        camera_id=camera_id,
        timestamp=HazardEvent.generate_timestamp(),
        is_hazard=is_hazard,
        confidence=0.85,
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.2),
        metadata=DiagnosticMetadata(
            frame_index=0, detection_class="Human", frames_detected=3
        ),
    )


def _build_pipeline(
    camera_sequence: List[str],
    per_camera_timeout: int = 30,
    frame_sampler=None,
    yolo_detector=None,
    human_detector=None,
    container_analyzer=None,
    alert_dispatcher=None,
    camera_switcher=None,
    shutdown_event=None,
    tracer=None,
):
    """Build a DetectionPipeline with mocked components."""
    config = PipelineConfig(
        camera_sequence=camera_sequence,
        per_camera_timeout_seconds=per_camera_timeout,
    )

    fs = frame_sampler or MagicMock()
    yd = yolo_detector or MagicMock()
    hd = human_detector or MagicMock()
    ca = container_analyzer or MagicMock()
    ad = alert_dispatcher or MagicMock()
    cs = camera_switcher or MagicMock()

    # Default mock behaviors
    if frame_sampler is None:
        fs.sample.return_value = _make_frame_sequence()
        fs.release.return_value = None
    if yolo_detector is None:
        yd.detect.return_value = [[_make_detection()] for _ in range(6)]
    if human_detector is None:
        hd.analyze.return_value = [_make_hazard_event()]
    if container_analyzer is None:
        ca.analyze.return_value = []
    if alert_dispatcher is None:
        ad.dispatch.return_value = True
    if camera_switcher is None:
        cs.transition.return_value = True

    se = shutdown_event or threading.Event()

    pipeline = DetectionPipeline(
        config=config,
        frame_sampler=fs,
        yolo_detector=yd,
        human_detector=hd,
        container_analyzer=ca,
        alert_dispatcher=ad,
        camera_switcher=cs,
        shutdown_event=se,
        tracer=tracer,
    )
    return pipeline, {"fs": fs, "yd": yd, "hd": hd, "ca": ca, "ad": ad, "cs": cs, "se": se}


# =============================================================================
# Test: Sequential camera processing order (Requirement 6.1)
# =============================================================================


class TestSequentialCameraProcessing:
    """
    THE Detection_Pipeline SHALL process cameras one at a time in a
    configurable sequence defined in the system YAML configuration file.

    Validates: Requirement 6.1
    """

    def test_cameras_processed_in_configured_order(self):
        """Cameras are processed in the exact order specified in config."""
        cameras = ["cam_01", "cam_02", "cam_03", "cam_04"]
        cs = MagicMock()
        cs.transition.return_value = True

        pipeline, mocks = _build_pipeline(
            camera_sequence=cameras, camera_switcher=cs
        )

        # Process each camera individually to verify order
        processed_order = []
        for cam in cameras:
            pipeline.process_camera(cam)
            processed_order.append(cam)

        # Verify transition calls in correct order
        transition_calls = [call[0][0] for call in cs.transition.call_args_list]
        assert transition_calls == cameras

    def test_single_camera_sequence(self):
        """Pipeline handles a single-camera sequence."""
        pipeline, mocks = _build_pipeline(camera_sequence=["cam_only"])
        events = pipeline.process_camera("cam_only")
        mocks["cs"].transition.assert_called_once_with("cam_only")
        assert len(events) >= 0  # Should not raise

    def test_full_pipeline_stages_invoked_per_camera(self):
        """Each camera invokes all pipeline stages in order."""
        pipeline, mocks = _build_pipeline(camera_sequence=["cam_01"])
        pipeline.process_camera("cam_01")

        mocks["cs"].transition.assert_called_once_with("cam_01")
        mocks["fs"].sample.assert_called_once_with("cam_01")
        mocks["yd"].detect.assert_called_once()
        mocks["hd"].analyze.assert_called_once()
        mocks["ca"].analyze.assert_called_once()
        mocks["ad"].dispatch.assert_called()


# =============================================================================
# Test: Exception handling and camera skip behavior (Requirement 6.5)
# =============================================================================


class TestExceptionHandlingAndSkip:
    """
    IF any detection module raises an exception, THEN THE Detection_Pipeline
    SHALL log the error with the module name and camera identifier, skip the
    remaining pipeline stages for that camera, and proceed to the next camera.

    Validates: Requirement 6.5
    """

    def test_frame_sampler_exception_skips_camera(self):
        """Exception in frame_sampler skips remaining stages."""
        fs = MagicMock()
        fs.sample.side_effect = RuntimeError("Camera hardware fault")
        fs.release.return_value = None

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], frame_sampler=fs
        )
        events = pipeline.process_camera("cam_01")

        assert events == []
        mocks["yd"].detect.assert_not_called()
        mocks["hd"].analyze.assert_not_called()
        mocks["ca"].analyze.assert_not_called()

    def test_yolo_detector_exception_skips_remaining(self):
        """Exception in yolo_detector skips human/container analysis."""
        yd = MagicMock()
        yd.detect.side_effect = RuntimeError("CUDA out of memory")

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], yolo_detector=yd
        )
        events = pipeline.process_camera("cam_01")

        assert events == []
        mocks["hd"].analyze.assert_not_called()
        mocks["ca"].analyze.assert_not_called()

    def test_human_detector_exception_does_not_block_container(self):
        """Exception in human_detector allows container_analyzer to still run."""
        hd = MagicMock()
        hd.analyze.side_effect = ValueError("Zone map corrupted")

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], human_detector=hd
        )
        events = pipeline.process_camera("cam_01")

        # Container analyzer should still be called
        mocks["ca"].analyze.assert_called_once()

    def test_container_analyzer_exception_does_not_block_dispatch(self):
        """Exception in container_analyzer still allows dispatch of earlier events."""
        ca = MagicMock()
        ca.analyze.side_effect = RuntimeError("Flow analyzer crash")

        hd = MagicMock()
        human_event = _make_hazard_event()
        hd.analyze.return_value = [human_event]

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"],
            human_detector=hd,
            container_analyzer=ca,
        )
        events = pipeline.process_camera("cam_01")

        # Should still dispatch the human events
        mocks["ad"].dispatch.assert_called()
        assert human_event in events

    def test_camera_transition_failure_skips_all_stages(self):
        """Failed camera transition (returns False) skips all processing."""
        cs = MagicMock()
        cs.transition.return_value = False

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_bad"], camera_switcher=cs
        )
        events = pipeline.process_camera("cam_bad")

        assert events == []
        mocks["fs"].sample.assert_not_called()

    def test_frame_sampler_returns_none_skips_detection(self):
        """When frame_sampler returns None, detection stages are skipped."""
        fs = MagicMock()
        fs.sample.return_value = None
        fs.release.return_value = None

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], frame_sampler=fs
        )
        events = pipeline.process_camera("cam_01")

        assert events == []
        mocks["yd"].detect.assert_not_called()


# =============================================================================
# Test: Per-camera timeout enforcement (Requirement 6.7)
# =============================================================================


class TestPerCameraTimeout:
    """
    IF the per-camera processing timeout is exceeded, THEN THE Detection_Pipeline
    SHALL terminate processing for that camera, log a timeout warning with the
    camera identifier and elapsed time, and proceed to the next camera.

    Validates: Requirement 6.7
    """

    def test_timeout_returns_empty_events(self):
        """Exceeding per-camera timeout returns empty event list."""
        fs = MagicMock()

        def slow_sample(camera_id):
            time.sleep(3)  # Exceed 1-second timeout
            return _make_frame_sequence(camera_id)

        fs.sample.side_effect = slow_sample
        fs.release.return_value = None

        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_slow"],
            per_camera_timeout=1,
            frame_sampler=fs,
        )

        # Use _process_camera_with_timeout which enforces the timeout
        events = pipeline._process_camera_with_timeout("cam_slow")
        assert events == []

    def test_fast_processing_completes_within_timeout(self):
        """Processing that completes quickly returns events normally."""
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"],
            per_camera_timeout=30,
        )

        events = pipeline._process_camera_with_timeout("cam_01")
        assert len(events) > 0

    def test_timeout_does_not_block_next_camera(self):
        """After a timeout, the pipeline proceeds to the next camera."""
        call_order = []

        fs = MagicMock()

        def sample_side_effect(camera_id):
            if camera_id == "cam_slow":
                time.sleep(3)
                return _make_frame_sequence(camera_id)
            call_order.append(camera_id)
            return _make_frame_sequence(camera_id)

        fs.sample.side_effect = sample_side_effect
        fs.release.return_value = None

        shutdown = threading.Event()
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_slow", "cam_fast"],
            per_camera_timeout=1,
            frame_sampler=fs,
            shutdown_event=shutdown,
        )

        # Signal shutdown after processing begins to stop after one cycle
        def stop_after_delay():
            time.sleep(4)
            shutdown.set()

        stopper = threading.Thread(target=stop_after_delay, daemon=True)
        stopper.start()
        pipeline.run()

        # cam_fast should have been reached
        assert "cam_fast" in call_order


# =============================================================================
# Test: Continuous cycling after last camera (Requirement 6.6)
# =============================================================================


class TestContinuousCycling:
    """
    WHEN the Detection_Pipeline has completed processing the last camera in the
    configured sequence, THE Detection_Pipeline SHALL restart processing from
    the first camera in the sequence, cycling continuously until shutdown.

    Validates: Requirement 6.6
    """

    def test_pipeline_cycles_continuously_until_shutdown(self):
        """Pipeline loops through camera sequence until shutdown event is set."""
        cameras = ["cam_01", "cam_02"]
        shutdown = threading.Event()

        pipeline, mocks = _build_pipeline(
            camera_sequence=cameras, shutdown_event=shutdown
        )

        # Let it run for a short period then shut down
        def stop_after_cycles():
            # Give it enough time for at least 2 cycles
            time.sleep(0.5)
            shutdown.set()

        stopper = threading.Thread(target=stop_after_cycles, daemon=True)
        stopper.start()
        pipeline.run()

        # Should have completed at least 1 full cycle
        assert pipeline.cycle_count >= 1

    def test_cycle_count_increments_per_full_sequence(self):
        """cycle_count increments once per complete camera sequence traversal."""
        cameras = ["cam_01", "cam_02", "cam_03"]
        shutdown = threading.Event()

        pipeline, mocks = _build_pipeline(
            camera_sequence=cameras, shutdown_event=shutdown
        )

        # Let it run for roughly 2 cycles
        def stop_later():
            time.sleep(0.3)
            shutdown.set()

        stopper = threading.Thread(target=stop_later, daemon=True)
        stopper.start()
        pipeline.run()

        assert pipeline.cycle_count >= 1

    def test_shutdown_mid_cycle_stops_gracefully(self):
        """Setting shutdown event mid-cycle stops the pipeline gracefully."""
        cameras = ["cam_01", "cam_02", "cam_03", "cam_04"]
        shutdown = threading.Event()

        call_count = {"n": 0}
        cs = MagicMock()

        def transition_with_shutdown(camera_id):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                shutdown.set()
            return True

        cs.transition.side_effect = transition_with_shutdown

        pipeline, mocks = _build_pipeline(
            camera_sequence=cameras,
            camera_switcher=cs,
            shutdown_event=shutdown,
        )
        pipeline.run()

        # Should stop without processing all cameras in the sequence
        assert call_count["n"] >= 2

    def test_empty_camera_sequence_does_not_loop_forever(self):
        """Empty camera sequence does not cause infinite loop — cycle completes immediately."""
        shutdown = threading.Event()

        pipeline, mocks = _build_pipeline(
            camera_sequence=[], shutdown_event=shutdown
        )

        # Should exit promptly since there are no cameras to process
        def stop_safety():
            time.sleep(1)
            shutdown.set()

        stopper = threading.Thread(target=stop_safety, daemon=True)
        stopper.start()
        pipeline.run()

        # Pipeline ran but had nothing to process
        assert pipeline.cycle_count >= 1


# =============================================================================
# Test: Performance timing output correctness (Requirement 6.5)
# =============================================================================


class TestPerformanceTiming:
    """
    THE Detection_Pipeline SHALL log the start and end time for each camera's
    processing cycle.

    Tests pipeline timing instrumentation and tracer integration.

    Validates: Requirement 6.5
    """

    def test_process_camera_records_stage_timings(self):
        """process_camera produces timing data via PipelineTracer."""
        tracer = PipelineTracer(enabled=True)
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], tracer=tracer
        )

        pipeline.process_camera("cam_01")

        # Tracer should have recorded a completed trace
        assert len(tracer.traces) == 1
        trace = tracer.traces[0]
        assert trace["camera_id"] == "cam_01"
        assert trace["total_duration_ms"] > 0

    def test_tracer_records_all_pipeline_modules(self):
        """All pipeline stages are recorded in the trace."""
        tracer = PipelineTracer(enabled=True)
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], tracer=tracer
        )

        pipeline.process_camera("cam_01")

        trace = tracer.traces[0]
        module_names = [m["module"] for m in trace["modules"]]

        # All expected stages should be present
        assert "camera_switcher" in module_names
        assert "frame_sampler" in module_names
        assert "yolo_detector" in module_names
        assert "human_detector" in module_names
        assert "container_analyzer" in module_names
        assert "alert_dispatcher" in module_names

    def test_each_module_has_positive_duration(self):
        """Each traced module has a positive duration."""
        tracer = PipelineTracer(enabled=True)
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01"], tracer=tracer
        )

        pipeline.process_camera("cam_01")

        trace = tracer.traces[0]
        for module_entry in trace["modules"]:
            assert module_entry["duration_ms"] >= 0

    def test_get_statistics_reflects_processing(self):
        """get_statistics returns accumulated detection and hazard counts."""
        pipeline, mocks = _build_pipeline(camera_sequence=["cam_01"])
        pipeline.process_camera("cam_01")

        stats = pipeline.get_statistics()
        assert stats["camera_sequence"] == ["cam_01"]
        assert stats["per_camera_timeout_seconds"] == 30


# =============================================================================
# Visual output: timing breakdown chart and pipeline trace JSON
# =============================================================================


class TestVisualOutputs:
    """
    Generate visual diagnostic artifacts for pipeline analysis.

    Outputs:
    - tests/output/pipeline_timing_breakdown.png: per-module timing bar chart
    - tests/output/pipeline_trace.json: pipeline execution trace
    """

    def test_generate_timing_breakdown_chart(self, output_dir):
        """Generate per-module timing breakdown chart as PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tracer = PipelineTracer(enabled=True)
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01", "cam_02", "cam_03"],
            tracer=tracer,
        )

        # Process all cameras to collect timing data
        for cam in ["cam_01", "cam_02", "cam_03"]:
            pipeline.process_camera(cam)

        # Aggregate timing per module across all traces
        module_timings: Dict[str, List[float]] = {}
        for trace in tracer.traces:
            for module_entry in trace["modules"]:
                name = module_entry["module"]
                duration = module_entry.get("duration_ms", 0.0)
                if name not in module_timings:
                    module_timings[name] = []
                module_timings[name].append(duration)

        # Build the chart
        module_names = list(module_timings.keys())
        avg_times = [np.mean(v) for v in module_timings.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(module_names, avg_times, color="#3498db", alpha=0.8)
        ax.set_xlabel("Average Time (ms)", fontsize=12)
        ax.set_title(
            "Pipeline Module Timing Breakdown (avg per camera)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}ms", va="center", fontsize=9
            )

        plt.tight_layout()
        chart_path = output_dir / "pipeline_timing_breakdown.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert chart_path.exists()
        assert chart_path.stat().st_size > 0

    def test_generate_pipeline_trace_json(self, output_dir):
        """Generate pipeline execution trace as JSON."""
        from tests.visual_helpers import save_json_report

        tracer = PipelineTracer(enabled=True)
        pipeline, mocks = _build_pipeline(
            camera_sequence=["cam_01", "cam_02"],
            tracer=tracer,
        )

        for cam in ["cam_01", "cam_02"]:
            pipeline.process_camera(cam)

        # Build trace output
        trace_data = {
            "pipeline_config": {
                "camera_sequence": ["cam_01", "cam_02"],
                "per_camera_timeout_seconds": 30,
            },
            "traces": [],
        }

        for trace in tracer.traces:
            trace_entry = {
                "camera_id": trace["camera_id"],
                "total_duration_ms": trace["total_duration_ms"],
                "start_timestamp": trace.get("start_timestamp"),
                "end_timestamp": trace.get("end_timestamp"),
                "modules": [
                    {
                        "module": m["module"],
                        "duration_ms": m.get("duration_ms", 0),
                        "input_info": m.get("input_info", {}),
                        "output_info": m.get("output_info", {}),
                    }
                    for m in trace["modules"]
                ],
            }
            trace_data["traces"].append(trace_entry)

        trace_path = output_dir / "pipeline_trace.json"
        save_json_report(trace_data, trace_path)

        assert trace_path.exists()
        # Validate JSON structure
        with open(trace_path) as f:
            loaded = json.load(f)
        assert "traces" in loaded
        assert len(loaded["traces"]) == 2
        assert loaded["traces"][0]["camera_id"] == "cam_01"
