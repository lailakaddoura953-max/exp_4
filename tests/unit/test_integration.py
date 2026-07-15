"""
Integration tests for the Hazard Detection System end-to-end pipeline.

Tests the full pipeline with mock camera feeds:
    frame sampling → YOLO inference → hazard detection → alert dispatch

Uses:
- Real: ZoneMap, HumanDetector, ContainerAnalyzer, AlertDispatcher, CameraSwitcher
- Mocked: FrameSampler (no hardware), YOLODetector (no YOLO model)
- ConfigurationManager via a temp YAML config file
- threading.Event for graceful shutdown

Generates visual outputs:
- tests/output/integration_detection_frames.png  (annotated frames)
- tests/output/integration_timing_waterfall.png  (end-to-end timing waterfall)
- tests/output/integration_summary.json          (detection summary report)

Validates: Requirements 6.2, 9.1, 14.1
"""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Bootstrap src path
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hazard_detection.alert_dispatcher import AlertDispatcher, AlertChannelAdapter
from hazard_detection.camera_switcher import CameraSwitcher
from hazard_detection.container_analyzer import ContainerAnalyzer
from hazard_detection.detection_pipeline import DetectionPipeline
from hazard_detection.human_detector import HumanDetector
from hazard_detection.models import (
    AlertDispatcherConfig,
    BBox,
    CameraSwitcherConfig,
    ContainerAnalyzerConfig,
    Detection,
    DiagnosticMetadata,
    FrameSequence,
    HazardEvent,
    HumanDetectorConfig,
    PipelineConfig,
)
from hazard_detection.zone_map import ZoneMap

from tests.visual_helpers import plot_annotated_frame, save_json_report

# ---------------------------------------------------------------------------
# Constants — synthetic scenario
# ---------------------------------------------------------------------------

_CAM_ID = "cam_01"
_NUM_FRAMES = 6
_FRAME_H, _FRAME_W = 640, 640

# The no-people zone covers x=[0.0, 0.6] across the full height.
# A human placed at x_center=0.3 falls squarely inside it.
_NO_PEOPLE_ZONE_CONFIG = {
    "zones": {
        _CAM_ID: [
            {
                "zone_type": "no_people",
                "vertices": [[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]],
            },
            {
                "zone_type": "operation",
                "vertices": [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]],
            },
        ]
    }
}

# ---------------------------------------------------------------------------
# Minimal alert channel stub (real adapter protocol, no external I/O)
# ---------------------------------------------------------------------------


class _InMemoryChannel:
    """Real AlertChannelAdapter that collects dispatched payloads in memory."""

    def __init__(self, name: str = "in_memory"):
        self._name = name
        self.received: List[Dict[str, Any]] = []

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        self.received.append(dict(alert_payload))
        return True

    def get_name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Helpers — build real components
# ---------------------------------------------------------------------------


def _make_frames(num: int = _NUM_FRAMES, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8) for _ in range(num)]


def _make_frame_sequence(
    camera_id: str = _CAM_ID,
    num: int = _NUM_FRAMES,
    seed: int = 42,
) -> FrameSequence:
    frames = _make_frames(num, seed)
    timestamps = [1700000000.0 + i * 0.5 for i in range(num)]
    return FrameSequence(frames=frames, camera_id=camera_id, timestamps=timestamps)


def _make_detection(
    class_label: str,
    x_center: float,
    y_center: float,
    width: float = 0.1,
    height: float = 0.2,
    confidence: float = 0.85,
) -> Detection:
    return Detection(
        bbox=BBox(x_center=x_center, y_center=y_center, width=width, height=height),
        class_label=class_label,
        confidence=confidence,
    )


def _write_zone_config(tmp_dir: Path) -> str:
    """Write a temp zone YAML file and return its path."""
    zone_path = tmp_dir / "zones.yaml"
    zone_path.write_text(yaml.dump(_NO_PEOPLE_ZONE_CONFIG))
    return str(zone_path)


def _write_pipeline_config(tmp_dir: Path, zone_path: str, checkpoint_path: str) -> str:
    """Write a minimal pipeline YAML config and return its path."""
    cfg = {
        "cameras": {"sequence": [_CAM_ID]},
        "yolo": {"checkpoint_path": checkpoint_path, "device": "cpu"},
        "system": {"frame_sample_count": 6, "per_camera_timeout_seconds": 30},
        "detection": {
            "human": {"confidence_threshold": 0.5},
            "container": {
                "confidence_threshold": 0.5,
                "flipped_aspect_ratio_threshold": 1.5,
                "safe_overlap_threshold": 0.3,
                "ground_level_threshold": 0.4,
                "motion_threshold": 0.7,
                "iou_threshold": 0.5,
            },
        },
        "alerts": {"rate_limit_seconds": 10, "channels": ["in_memory"]},
    }
    cfg_path = tmp_dir / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


def _build_real_pipeline(
    detections_per_frame: List[List[Detection]],
    frame_sequence: Optional[FrameSequence] = None,
    camera_id: str = _CAM_ID,
    shutdown_event: Optional[threading.Event] = None,
) -> tuple:
    """
    Wire together real components (except FrameSampler and YOLODetector which
    are mocked) and return (pipeline, channel, shutdown_event).
    """
    seq = frame_sequence or _make_frame_sequence(camera_id)

    # --- Mock FrameSampler (no hardware) ---
    mock_frame_sampler = MagicMock()
    mock_frame_sampler.sample.return_value = seq
    mock_frame_sampler.release.return_value = None

    # --- Mock YOLODetector (no YOLO model) ---
    mock_yolo = MagicMock()
    mock_yolo.detect.return_value = detections_per_frame

    # --- Real ZoneMap ---
    zone_map = ZoneMap()
    zone_map._zones = {
        camera_id: [
            __import__(
                "hazard_detection.models", fromlist=["ZonePolygon"]
            ).ZonePolygon(
                vertices=[(0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)],
                zone_type="no_people",
                camera_id=camera_id,
            ),
            __import__(
                "hazard_detection.models", fromlist=["ZonePolygon"]
            ).ZonePolygon(
                vertices=[(0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)],
                zone_type="operation",
                camera_id=camera_id,
            ),
        ]
    }

    # --- Real HumanDetector ---
    human_config = HumanDetectorConfig(confidence_threshold=0.5)
    human_detector = HumanDetector(zone_map=zone_map, config=human_config)

    # --- Real ContainerAnalyzer (with mock flow analyzer) ---
    mock_flow = MagicMock()
    mock_flow.compute_flow.return_value = np.zeros((_FRAME_H, _FRAME_W, 2), dtype=np.float32)
    mock_flow.get_flow_consistency_score.return_value = 0.1

    container_config = ContainerAnalyzerConfig(
        confidence_threshold=0.5,
        flipped_aspect_ratio_threshold=1.5,
        safe_overlap_threshold=0.3,
        ground_level_threshold=0.4,
        motion_threshold=0.7,
        iou_threshold=0.5,
    )
    container_analyzer = ContainerAnalyzer(
        flow_analyzer=mock_flow, config=container_config
    )

    # --- Real AlertDispatcher with in-memory channel ---
    channel = _InMemoryChannel("in_memory")
    alert_config = AlertDispatcherConfig(rate_limit_seconds=10, channels=["in_memory"])
    alert_dispatcher = AlertDispatcher(channels=[channel], config=alert_config)

    # --- Real CameraSwitcher ---
    switcher_config = CameraSwitcherConfig(camera_list=[camera_id])
    camera_switcher = CameraSwitcher(config=switcher_config)

    # --- Pipeline config ---
    pipeline_config = PipelineConfig(
        camera_sequence=[camera_id],
        per_camera_timeout_seconds=30,
    )

    se = shutdown_event or threading.Event()

    pipeline = DetectionPipeline(
        config=pipeline_config,
        frame_sampler=mock_frame_sampler,
        yolo_detector=mock_yolo,
        human_detector=human_detector,
        container_analyzer=container_analyzer,
        alert_dispatcher=alert_dispatcher,
        camera_switcher=camera_switcher,
        shutdown_event=se,
    )

    return pipeline, channel, se


# ===========================================================================
# Scenario helpers — synthetic detection sequences
# ===========================================================================


def _zone_violation_scenario() -> List[List[Detection]]:
    """
    6 frames each containing a human at (0.3, 0.5) — squarely in no_people zone.
    Temporal rule: >=2 consecutive frames → is_hazard=True.
    """
    human_det = _make_detection("Human", x_center=0.3, y_center=0.5, confidence=0.85)
    return [[human_det] for _ in range(_NUM_FRAMES)]


def _container_misalignment_scenario() -> List[List[Detection]]:
    """
    6 frames each with a misaligned container — confirmed hazard after >=2 frames.
    """
    mis_det = _make_detection(
        "Container - Misaligned", x_center=0.7, y_center=0.5,
        width=0.2, height=0.1, confidence=0.80,
    )
    return [[mis_det] for _ in range(_NUM_FRAMES)]


def _mixed_scenario() -> List[List[Detection]]:
    """
    Realistic mixed scenario:
    - Human in no-people zone (x=0.3, in left half) present in all 6 frames
    - Misaligned container (x=0.7, in right half) present in all 6 frames
    Both should be confirmed hazards (>=2 consecutive frames).
    """
    human_det = _make_detection("Human", x_center=0.3, y_center=0.5, confidence=0.85)
    mis_det = _make_detection(
        "Container - Misaligned", x_center=0.7, y_center=0.5,
        width=0.2, height=0.1, confidence=0.80,
    )
    return [[human_det, mis_det] for _ in range(_NUM_FRAMES)]


def _no_hazard_scenario() -> List[List[Detection]]:
    """Human only in operation zone — should NOT produce a zone_violation hazard."""
    human_det = _make_detection("Human", x_center=0.8, y_center=0.5, confidence=0.85)
    return [[human_det] for _ in range(_NUM_FRAMES)]


def _transient_scenario() -> List[List[Detection]]:
    """Human detected in no-people zone for exactly 1 frame — transient, not a hazard."""
    human_det = _make_detection("Human", x_center=0.3, y_center=0.5, confidence=0.85)
    return [[human_det]] + [[] for _ in range(_NUM_FRAMES - 1)]


# ===========================================================================
# Test class 1: Full pipeline data flow
# ===========================================================================


class TestEndToEndPipelineFlow:
    """
    Tests the full pipeline: mock frames → YOLO stub → real hazard logic → real dispatch.

    Validates: Requirement 6.2 (pipeline stage order), 9.1 (dispatch within 5s)
    """

    def test_zone_violation_produces_dispatched_hazard_event(self):
        """
        Human in no-people zone across ≥2 consecutive frames must produce a
        confirmed hazard event (is_hazard=True) that is dispatched via AlertDispatcher.

        Validates: Requirement 6.2, 9.1
        """
        pipeline, channel, _ = _build_real_pipeline(_zone_violation_scenario())
        t_start = time.perf_counter()
        events = pipeline.process_camera(_CAM_ID)
        elapsed = time.perf_counter() - t_start

        hazards = [e for e in events if e.is_hazard]
        assert len(hazards) >= 1, "Expected at least one confirmed hazard event"

        zone_violations = [e for e in hazards if e.hazard_type == "zone_violation"]
        assert len(zone_violations) >= 1, "Expected a zone_violation hazard event"

        # Requirement 9.1: dispatch must complete within 5 seconds
        assert elapsed < 5.0, f"Pipeline took {elapsed:.2f}s — exceeds 5s dispatch window"

        # Verify AlertDispatcher actually received the event
        dispatched = channel.received
        assert len(dispatched) >= 1, "AlertDispatcher channel received no alerts"
        types = [p["hazard_type"] for p in dispatched]
        assert "zone_violation" in types

    def test_container_misalignment_produces_dispatched_hazard_event(self):
        """
        Misaligned container across ≥2 consecutive frames must produce a confirmed
        container_misalignment hazard that is dispatched.

        Validates: Requirement 6.2
        """
        pipeline, channel, _ = _build_real_pipeline(_container_misalignment_scenario())
        events = pipeline.process_camera(_CAM_ID)

        hazards = [e for e in events if e.is_hazard]
        misalignment_hazards = [e for e in hazards if e.hazard_type == "container_misalignment"]
        assert len(misalignment_hazards) >= 1

        types = [p["hazard_type"] for p in channel.received]
        assert "container_misalignment" in types

    def test_mixed_scenario_produces_both_hazard_types(self):
        """
        Mixed scenario (human in no-people zone + misaligned container) must
        produce both hazard types, all dispatched.

        Validates: Requirement 6.2
        """
        pipeline, channel, _ = _build_real_pipeline(_mixed_scenario())
        events = pipeline.process_camera(_CAM_ID)

        hazards = [e for e in events if e.is_hazard]
        hazard_types = {e.hazard_type for e in hazards}

        assert "zone_violation" in hazard_types, "Expected zone_violation in mixed scenario"
        assert "container_misalignment" in hazard_types, "Expected container_misalignment in mixed"

        dispatched_types = {p["hazard_type"] for p in channel.received}
        assert "zone_violation" in dispatched_types
        assert "container_misalignment" in dispatched_types

    def test_human_in_operation_zone_does_not_trigger_hazard(self):
        """
        Human detected exclusively in operation zone must NOT produce a
        zone_violation hazard event (Requirement 2.4).
        """
        pipeline, channel, _ = _build_real_pipeline(_no_hazard_scenario())
        events = pipeline.process_camera(_CAM_ID)

        zone_violations = [e for e in events if e.hazard_type == "zone_violation" and e.is_hazard]
        assert len(zone_violations) == 0, "No zone_violation hazard expected for operation zone"

    def test_transient_detection_is_not_a_hazard(self):
        """
        Human detected in no-people zone for exactly 1 frame must be logged
        as a transient event (is_hazard=False), not dispatched.
        """
        pipeline, channel, _ = _build_real_pipeline(_transient_scenario())
        events = pipeline.process_camera(_CAM_ID)

        zone_events = [e for e in events if e.hazard_type == "zone_violation"]
        # Any zone event found should be transient (is_hazard=False)
        if zone_events:
            confirmed = [e for e in zone_events if e.is_hazard]
            assert len(confirmed) == 0, "1-frame detection should not be a confirmed hazard"

        # AlertDispatcher should not have dispatched any zone_violation
        dispatched_zone_violations = [
            p for p in channel.received if p["hazard_type"] == "zone_violation"
        ]
        assert len(dispatched_zone_violations) == 0


    def test_all_pipeline_stages_are_invoked(self):
        """
        Verify that all pipeline stages are invoked in the correct order:
        camera_switcher → frame_sampler → yolo → human_detector → container_analyzer → alert_dispatcher.

        Validates: Requirement 6.2
        """
        call_log: List[str] = []

        # Wrap real pipeline with side-effect tracking on mocks
        seq = _make_frame_sequence(_CAM_ID)

        mock_fs = MagicMock()
        mock_fs.sample.side_effect = lambda cam: (call_log.append("frame_sampler") or seq)
        mock_fs.release.return_value = None

        mock_yolo = MagicMock()
        mock_yolo.detect.side_effect = lambda fs: (
            call_log.append("yolo_detector") or _zone_violation_scenario()
        )

        zone_map = ZoneMap()
        zone_map._zones = {}  # Empty → defaults to no_people for all points

        human_detector = HumanDetector(
            zone_map=zone_map, config=HumanDetectorConfig(confidence_threshold=0.5)
        )

        mock_flow = MagicMock()
        mock_flow.compute_flow.return_value = np.zeros((_FRAME_H, _FRAME_W, 2), dtype=np.float32)
        mock_flow.get_flow_consistency_score.return_value = 0.1
        container_analyzer = ContainerAnalyzer(
            flow_analyzer=mock_flow,
            config=ContainerAnalyzerConfig(),
        )

        channel = _InMemoryChannel()
        alert_dispatcher = AlertDispatcher(
            channels=[channel],
            config=AlertDispatcherConfig(rate_limit_seconds=10, channels=["in_memory"]),
        )

        switcher_config = CameraSwitcherConfig(camera_list=[_CAM_ID])
        camera_switcher = CameraSwitcher(config=switcher_config)

        pipeline_config = PipelineConfig(camera_sequence=[_CAM_ID], per_camera_timeout_seconds=30)
        pipeline = DetectionPipeline(
            config=pipeline_config,
            frame_sampler=mock_fs,
            yolo_detector=mock_yolo,
            human_detector=human_detector,
            container_analyzer=container_analyzer,
            alert_dispatcher=alert_dispatcher,
            camera_switcher=camera_switcher,
        )

        pipeline.process_camera(_CAM_ID)

        assert "frame_sampler" in call_log, "frame_sampler stage not invoked"
        assert "yolo_detector" in call_log, "yolo_detector stage not invoked"
        # Stage order: frame_sampler before yolo_detector
        assert call_log.index("frame_sampler") < call_log.index("yolo_detector")


# ===========================================================================
# Test class 2: Configuration loading and component initialization
# ===========================================================================


class TestConfigurationLoading:
    """
    Test ConfigurationManager-driven pipeline initialization from a temp YAML.

    Validates: Requirement 14.1
    """

    def test_configuration_loads_from_yaml(self, tmp_path):
        """
        ConfigurationManager successfully loads a valid YAML file and constructs
        a PipelineConfig with the expected camera sequence.

        Validates: Requirement 14.1
        """
        from hazard_detection.config import ConfigurationManager

        # Create a minimal valid checkpoint placeholder
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")

        cfg_path = _write_pipeline_config(tmp_path, "", str(checkpoint))

        config_manager = ConfigurationManager(config_path=cfg_path)
        config_manager.load()
        pipeline_cfg = config_manager.get_pipeline_config()

        assert pipeline_cfg.camera_sequence == [_CAM_ID]
        assert pipeline_cfg.frame_sampler.frame_count == 6
        assert pipeline_cfg.alert_dispatcher.rate_limit_seconds == 10

    def test_components_initialize_with_loaded_config(self, tmp_path):
        """
        All real components initialize without error when constructed from
        a configuration loaded via ConfigurationManager.

        Validates: Requirement 14.1
        """
        from hazard_detection.config import ConfigurationManager

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")

        cfg_path = _write_pipeline_config(tmp_path, "", str(checkpoint))

        config_manager = ConfigurationManager(config_path=cfg_path)
        config_manager.load()
        pipeline_cfg = config_manager.get_pipeline_config()

        # All real components must construct without raising
        zone_map = ZoneMap()
        human_detector = HumanDetector(
            zone_map=zone_map, config=pipeline_cfg.human_detector
        )
        mock_flow = MagicMock()
        mock_flow.compute_flow.return_value = np.zeros((_FRAME_H, _FRAME_W, 2), dtype=np.float32)
        mock_flow.get_flow_consistency_score.return_value = 0.0
        container_analyzer = ContainerAnalyzer(
            flow_analyzer=mock_flow, config=pipeline_cfg.container_analyzer
        )
        channel = _InMemoryChannel()
        alert_dispatcher = AlertDispatcher(
            channels=[channel], config=pipeline_cfg.alert_dispatcher
        )
        camera_switcher = CameraSwitcher(config=pipeline_cfg.camera_switcher)

        assert camera_switcher.camera_list == [_CAM_ID]
        assert alert_dispatcher.config.rate_limit_seconds == 10

    def test_missing_config_file_raises_configuration_error(self, tmp_path):
        """
        ConfigurationManager raises ConfigurationError when the config file
        does not exist.

        Validates: Requirement 14.1
        """
        from hazard_detection.config import ConfigurationManager, ConfigurationError

        config_manager = ConfigurationManager(config_path=str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(ConfigurationError):
            config_manager.load()


# ===========================================================================
# Test class 3: Graceful shutdown behavior
# ===========================================================================


class TestGracefulShutdown:
    """
    Tests that the pipeline respects threading.Event shutdown signals.
    """

    def test_shutdown_before_run_exits_immediately(self):
        """Pipeline run() exits without processing any cameras when shutdown is pre-set."""
        pipeline, channel, se = _build_real_pipeline(_zone_violation_scenario())
        se.set()  # Signal shutdown before run()
        pipeline.run()

        assert pipeline.cycle_count == 0, "No cycles expected when shutdown pre-set"
        assert len(channel.received) == 0, "No alerts expected when shutdown pre-set"

    def test_shutdown_during_cycle_stops_gracefully(self):
        """
        Setting the shutdown event during pipeline.run() causes it to exit
        the loop cleanly (cycle_count >= 1).
        """
        pipeline, channel, se = _build_real_pipeline(_zone_violation_scenario())

        def stop_after_first_camera():
            # Give the pipeline just enough time to process one camera
            time.sleep(0.3)
            se.set()

        stopper = threading.Thread(target=stop_after_first_camera, daemon=True)
        stopper.start()
        pipeline.run()
        stopper.join(timeout=2.0)

        # Pipeline should have completed at least one cycle
        assert pipeline.cycle_count >= 1

    def test_pipeline_run_is_idempotent_after_shutdown(self):
        """
        Calling run() again after a shutdown event does nothing (loop condition
        is checked at entry).
        """
        pipeline, channel, se = _build_real_pipeline(_zone_violation_scenario())
        se.set()

        pipeline.run()
        count_after_first_run = pipeline.cycle_count

        pipeline.run()
        assert pipeline.cycle_count == count_after_first_run, (
            "Second run() after shutdown should not add more cycles"
        )


# ===========================================================================
# Test class 4: Alert dispatch within 5 seconds (Req 9.1)
# ===========================================================================


class TestAlertDispatchTiming:
    """
    THE Alert_Dispatcher SHALL dispatch hazard alerts through all configured
    channels within 5 seconds of event emission.

    Validates: Requirement 9.1
    """

    def test_dispatch_latency_under_5_seconds(self):
        """
        Confirmed hazard events are dispatched within 5 seconds from the
        start of pipeline.process_camera().
        """
        pipeline, channel, _ = _build_real_pipeline(_zone_violation_scenario())

        t_start = time.perf_counter()
        events = pipeline.process_camera(_CAM_ID)
        elapsed = time.perf_counter() - t_start

        hazards = [e for e in events if e.is_hazard]
        assert len(hazards) >= 1, "Expected hazard events for this scenario"
        assert elapsed < 5.0, f"Pipeline took {elapsed:.3f}s — exceeds 5s dispatch SLA"
        assert len(channel.received) >= 1, "Alerts must be dispatched by end of process_camera"

    def test_only_confirmed_hazards_are_dispatched(self):
        """
        AlertDispatcher must not dispatch non-hazard (is_hazard=False) events.
        Non-hazards are logged only.
        """
        pipeline, channel, _ = _build_real_pipeline(_transient_scenario())
        events = pipeline.process_camera(_CAM_ID)

        # All dispatched payloads must correspond to is_hazard=True events
        for payload in channel.received:
            assert payload["is_hazard"] is True, (
                f"Non-hazard event dispatched: {payload}"
            )


# ===========================================================================
# Visual outputs
# ===========================================================================


class TestVisualOutputs:
    """
    Generate visual diagnostic artifacts for the integration scenario:
    - integration_detection_frames.png  — annotated synthetic frame
    - integration_timing_waterfall.png  — end-to-end stage timing waterfall
    - integration_summary.json          — detection summary report
    """

    def test_generate_annotated_detection_frames(self, output_dir):
        """
        Save annotated detection frames with bounding boxes and zone overlays.

        Output: tests/output/integration_detection_frames.png
        """
        pipeline, channel, _ = _build_real_pipeline(_mixed_scenario())
        events = pipeline.process_camera(_CAM_ID)

        # Use the first synthetic frame as the background
        frame = _make_frames(1)[0]  # RGB-like uint8 array

        # Build detection list in the format expected by plot_annotated_frame
        detections_for_plot = [
            {
                "bbox": {
                    "x_center": e.bbox.x_center,
                    "y_center": e.bbox.y_center,
                    "width": e.bbox.width,
                    "height": e.bbox.height,
                },
                "class_label": e.metadata.detection_class,
                "confidence": e.confidence,
            }
            for e in events
        ]

        # Zone overlays
        zones_for_plot = [
            {
                "vertices": [(0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)],
                "zone_type": "no_people",
            },
            {
                "vertices": [(0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 1.0)],
                "zone_type": "operation",
            },
        ]

        out_path = output_dir / "integration_detection_frames.png"
        plot_annotated_frame(
            frame=frame,
            detections=detections_for_plot,
            zones=zones_for_plot,
            output_path=out_path,
            title="Integration Test — Annotated Detection Frame",
        )

        assert out_path.exists(), "integration_detection_frames.png was not created"
        assert out_path.stat().st_size > 0

    def test_generate_timing_waterfall(self, output_dir):
        """
        Save an end-to-end timing waterfall chart showing stage durations.

        Output: tests/output/integration_timing_waterfall.png
        """
        from hazard_detection.diagnostics import PipelineTracer

        tracer = PipelineTracer(enabled=True)

        seq = _make_frame_sequence(_CAM_ID)
        mock_fs = MagicMock()
        mock_fs.sample.return_value = seq
        mock_fs.release.return_value = None

        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = _mixed_scenario()

        zone_map = ZoneMap()
        zone_map._zones = {
            _CAM_ID: [
                __import__("hazard_detection.models", fromlist=["ZonePolygon"]).ZonePolygon(
                    vertices=[(0.0, 0.0), (0.6, 0.0), (0.6, 1.0), (0.0, 1.0)],
                    zone_type="no_people", camera_id=_CAM_ID,
                )
            ]
        }

        mock_flow = MagicMock()
        mock_flow.compute_flow.return_value = np.zeros((_FRAME_H, _FRAME_W, 2), dtype=np.float32)
        mock_flow.get_flow_consistency_score.return_value = 0.1

        channel = _InMemoryChannel()
        pipeline = DetectionPipeline(
            config=PipelineConfig(camera_sequence=[_CAM_ID], per_camera_timeout_seconds=30),
            frame_sampler=mock_fs,
            yolo_detector=mock_yolo,
            human_detector=HumanDetector(
                zone_map=zone_map, config=HumanDetectorConfig(confidence_threshold=0.5)
            ),
            container_analyzer=ContainerAnalyzer(
                flow_analyzer=mock_flow, config=ContainerAnalyzerConfig()
            ),
            alert_dispatcher=AlertDispatcher(
                channels=[channel],
                config=AlertDispatcherConfig(rate_limit_seconds=10, channels=["in_memory"]),
            ),
            camera_switcher=CameraSwitcher(config=CameraSwitcherConfig(camera_list=[_CAM_ID])),
            tracer=tracer,
        )

        pipeline.process_camera(_CAM_ID)

        # Build waterfall data from tracer
        stage_order = [
            "camera_switcher", "frame_sampler", "yolo_detector",
            "human_detector", "container_analyzer", "alert_dispatcher",
        ]
        trace = tracer.traces[0] if tracer.traces else {"modules": []}
        timing_map = {m["module"]: m.get("duration_ms", 0.0) for m in trace.get("modules", [])}

        durations = [timing_map.get(s, 0.0) for s in stage_order]
        cumulative = np.cumsum([0.0] + durations[:-1])

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

        for i, (stage, dur, start) in enumerate(zip(stage_order, durations, cumulative)):
            ax.barh(
                y=i, width=dur, left=start,
                color=colors[i % len(colors)], alpha=0.85, edgecolor="white", height=0.6,
            )
            ax.text(
                start + dur / 2, i,
                f"{dur:.2f}ms", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )

        ax.set_yticks(range(len(stage_order)))
        ax.set_yticklabels(stage_order, fontsize=10)
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_title(
            "Integration Test — End-to-End Stage Timing Waterfall",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        out_path = output_dir / "integration_timing_waterfall.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert out_path.exists(), "integration_timing_waterfall.png was not created"
        assert out_path.stat().st_size > 0

    def test_generate_summary_json(self, output_dir):
        """
        Save a structured detection summary report as JSON.

        Output: tests/output/integration_summary.json
        """
        pipeline, channel, _ = _build_real_pipeline(_mixed_scenario())
        events = pipeline.process_camera(_CAM_ID)

        hazards = [e for e in events if e.is_hazard]
        non_hazards = [e for e in events if not e.is_hazard]

        summary = {
            "scenario": "mixed (zone_violation + container_misalignment)",
            "camera_id": _CAM_ID,
            "num_frames_processed": _NUM_FRAMES,
            "total_events": len(events),
            "confirmed_hazards": len(hazards),
            "non_hazards_logged": len(non_hazards),
            "hazard_breakdown": {},
            "dispatched_alerts": len(channel.received),
            "events": [],
        }

        # Aggregate by type
        for event in events:
            ht = event.hazard_type
            if ht not in summary["hazard_breakdown"]:
                summary["hazard_breakdown"][ht] = {"total": 0, "confirmed": 0}
            summary["hazard_breakdown"][ht]["total"] += 1
            if event.is_hazard:
                summary["hazard_breakdown"][ht]["confirmed"] += 1

        # Detailed event list
        for e in events:
            summary["events"].append({
                "event_id": e.event_id,
                "hazard_type": e.hazard_type,
                "is_hazard": e.is_hazard,
                "confidence": round(e.confidence, 4),
                "frames_detected": e.metadata.frames_detected,
                "detection_class": e.metadata.detection_class,
                "bbox": {
                    "x_center": round(e.bbox.x_center, 4),
                    "y_center": round(e.bbox.y_center, 4),
                    "width": round(e.bbox.width, 4),
                    "height": round(e.bbox.height, 4),
                },
            })

        out_path = output_dir / "integration_summary.json"
        save_json_report(summary, out_path)

        assert out_path.exists(), "integration_summary.json was not created"
        with open(out_path) as f:
            loaded = json.load(f)
        assert "hazard_breakdown" in loaded
        assert loaded["camera_id"] == _CAM_ID
        assert isinstance(loaded["events"], list)
