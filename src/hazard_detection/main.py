"""
Main entry point and CLI interface for the Hazard Detection System.

Parses CLI arguments, loads configuration, initializes all pipeline components,
registers signal handlers for graceful shutdown, and runs the detection pipeline.

Requirements covered:
- 14.1: Load configuration from YAML file specified via CLI argument
- 6.1: Process cameras one at a time in configurable sequence
- 6.6: Restart from first camera after completing sequence, cycle continuously

CLI arguments:
  --config      Path to YAML configuration file (required or default path used)
  --log-level   Override log level for all modules (DEBUG|INFO|WARNING|ERROR)
  --dump-dir    Override diagnostics dump directory

Startup behaviour:
  1. Parse CLI args
  2. Load and validate ConfigurationManager
  3. Initialize diagnostics (structured logging, dumper, tracer)
  4. Initialize all pipeline components
  5. Register SIGINT/SIGTERM handlers → set shutdown threading.Event
  6. Log startup summary (config values, system info, component versions)
  7. Run DetectionPipeline.run() in the main thread

Shutdown behaviour:
  1. Signal handler sets the shutdown Event
  2. pipeline.run() exits its loop gracefully
  3. Final stats + pipeline state are logged before process exits

Degraded mode:
  If the configured YOLO checkpoint does not exist, a warning is logged and
  YOLODetector is replaced with a StubYOLODetector that always returns no
  detections. This allows development/testing without a real model file.
"""

import argparse
import logging
import os
import platform
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap path so that 'src/' sub-packages are importable when this file is
# run directly (e.g. python -m hazard_detection.main or python src/hazard_detection/main.py).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Hazard-detection imports
# ---------------------------------------------------------------------------
from hazard_detection.config import ConfigurationManager, ConfigurationError
from hazard_detection.diagnostics import (
    get_logger,
    initialize_diagnostics,
    DiagnosticDumper,
    PipelineTracer,
)
from hazard_detection.models import (
    FrameSamplerConfig,
    YOLOConfig,
    HumanDetectorConfig,
    ContainerAnalyzerConfig,
    AlertDispatcherConfig,
    CameraSwitcherConfig,
    PipelineConfig,
    Detection,
    FrameSequence,
)
from hazard_detection.zone_map import ZoneMap
from hazard_detection.camera_switcher import CameraSwitcher
from hazard_detection.human_detector import HumanDetector
from hazard_detection.container_analyzer import ContainerAnalyzer
from hazard_detection.alert_dispatcher import AlertDispatcher, AlertChannelAdapter
from hazard_detection.detection_pipeline import DetectionPipeline

# Lazy-import heavy dependencies so that import errors give clear messages.
try:
    from hazard_detection.yolo_detector import YOLODetector
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLODetector = None  # type: ignore[assignment, misc]

try:
    from hazard_detection.frame_sampler import FrameSampler
    from acquisition.frame_acquisition import FrameAcquisitionModule
    _FRAME_SAMPLER_AVAILABLE = True
except ImportError:
    _FRAME_SAMPLER_AVAILABLE = False
    FrameSampler = None  # type: ignore[assignment, misc]
    FrameAcquisitionModule = None  # type: ignore[assignment, misc]

try:
    from cv.flow_analyzer import OpticalFlowAnalyzer
    _FLOW_ANALYZER_AVAILABLE = True
except ImportError:
    _FLOW_ANALYZER_AVAILABLE = False
    OpticalFlowAnalyzer = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Module logger (set up properly after diagnostics init)
# ---------------------------------------------------------------------------
logger = get_logger("main")

# ---------------------------------------------------------------------------
# Version / system-info constants
# ---------------------------------------------------------------------------
SYSTEM_VERSION = "1.0.0"


# ===========================================================================
# Stub Components
# ===========================================================================


class _StubYOLODetector:
    """
    Development stub used when the configured checkpoint does not exist.

    Returns an empty detection list for every frame so the rest of the
    pipeline can still run without a real model file.
    """

    def detect(self, frame_sequence: FrameSequence, flow_maps=None) -> List[List[Detection]]:
        logger.debug(
            "StubYOLODetector.detect called — returning empty detections",
            extra={"component": "stub_yolo_detector"},
        )
        return [[] for _ in frame_sequence.frames]


class _StubFrameSampler:
    """
    Development stub used when FrameAcquisitionModule is unavailable.

    Returns a synthetic FrameSequence of blank frames so the pipeline
    can run end-to-end without real cameras.
    """

    def __init__(self, config: FrameSamplerConfig):
        self._config = config
        self._current_sequence: Optional[FrameSequence] = None

    def sample(self, camera_id: str) -> Optional[FrameSequence]:
        import numpy as np

        now = time.time()
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(self._config.frame_count)
        ]
        timestamps = [now + i * 0.033 for i in range(self._config.frame_count)]
        self._current_sequence = FrameSequence(
            frames=frames,
            camera_id=camera_id,
            timestamps=timestamps,
        )
        logger.debug(
            f"StubFrameSampler: generated {self._config.frame_count} blank frames "
            f"for camera '{camera_id}'",
            extra={"component": "stub_frame_sampler"},
        )
        return self._current_sequence

    def release(self) -> None:
        self._current_sequence = None


class _StubOpticalFlowAnalyzer:
    """Stub optical flow analyzer that always returns a zero-variance score."""

    def get_flow_consistency_score(self, *args, **kwargs) -> float:
        return 0.0

    def compute_flow(self, *args, **kwargs):
        import numpy as np
        return np.zeros((10, 10), dtype=np.float32)


class _LogOnlyChannel:
    """
    Alert channel that only logs alerts without sending them anywhere.

    Used when no real alert channels are configured or available.
    """

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        logger.info(
            "Alert (log-only channel): %s",
            alert_payload,
            extra={"component": "log_only_channel"},
        )
        return True

    def get_name(self) -> str:
        return "log_only"


# ===========================================================================
# CLI parsing
# ===========================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the hazard detection system.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with .config, .log_level, .dump_dir attributes.
    """
    parser = argparse.ArgumentParser(
        prog="hazard-detection",
        description=(
            "Hazard Detection System — monitors industrial yard cameras for "
            "zone violations, container misalignment, and unsafe orientations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the YAML configuration file. "
             "Defaults to 'config/hazard_detection.yaml'.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Override log level for all modules (DEBUG|INFO|WARNING|ERROR).",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Override the diagnostics dump directory for intermediate state snapshots.",
    )
    return parser.parse_args(argv)


# ===========================================================================
# Component initialisation helpers
# ===========================================================================


def _build_frame_sampler(config: PipelineConfig, dumper: DiagnosticDumper):
    """
    Create a FrameSampler (real or stub).

    Falls back to _StubFrameSampler if FrameAcquisitionModule is unavailable.
    """
    if _FRAME_SAMPLER_AVAILABLE:
        acquisition = FrameAcquisitionModule(
            buffer_size_per_camera=config.frame_sampler.frame_count,
        )
        sampler = FrameSampler(
            frame_acquisition=acquisition,
            config=config.frame_sampler,
        )
        logger.info(
            "FrameSampler initialised (real acquisition module)",
            extra={"component": "main"},
        )
        return sampler
    else:
        logger.warning(
            "FrameAcquisitionModule not available — using StubFrameSampler. "
            "Frame data will be synthetic blank frames.",
            extra={"component": "main"},
        )
        return _StubFrameSampler(config=config.frame_sampler)


def _build_yolo_detector(config: PipelineConfig) -> Any:
    """
    Create a YOLODetector (real or stub).

    If the checkpoint file does not exist, logs a warning and returns a
    _StubYOLODetector so the pipeline can still run in degraded mode.
    Requirement 13.4 says YOLODetector should raise on missing checkpoint,
    but the task spec asks for graceful degradation in main.py.
    """
    if not _YOLO_AVAILABLE:
        logger.warning(
            "ultralytics package not installed — using StubYOLODetector. "
            "No object detections will be produced.",
            extra={"component": "main"},
        )
        return _StubYOLODetector()

    checkpoint = config.yolo.checkpoint_path
    if not os.path.isfile(checkpoint):
        logger.warning(
            "YOLO checkpoint '%s' not found — using StubYOLODetector. "
            "Detection functionality disabled (degraded mode).",
            checkpoint,
            extra={"component": "main"},
        )
        return _StubYOLODetector()

    try:
        detector = YOLODetector(config=config.yolo)
        logger.info(
            "YOLODetector initialised from checkpoint '%s'",
            checkpoint,
            extra={"component": "main"},
        )
        return detector
    except FileNotFoundError as exc:
        logger.warning(
            "YOLODetector failed to load checkpoint (%s) — using StubYOLODetector.",
            exc,
            extra={"component": "main"},
        )
        return _StubYOLODetector()


def _build_flow_analyzer() -> Any:
    """Return a real or stub OpticalFlowAnalyzer."""
    if _FLOW_ANALYZER_AVAILABLE:
        return OpticalFlowAnalyzer()
    logger.warning(
        "OpticalFlowAnalyzer not available — using stub. "
        "Flow-based container analysis will be skipped.",
        extra={"component": "main"},
    )
    return _StubOpticalFlowAnalyzer()


def _build_alert_channels(config: AlertDispatcherConfig) -> List[AlertChannelAdapter]:
    """
    Build alert channel adapters from configuration.

    Currently wires a log-only channel for each configured channel name.
    Replace with real adapters (email, dashboard, etc.) as they are implemented.
    """
    channels: List[AlertChannelAdapter] = []
    for channel_name in config.channels:
        channels.append(_LogOnlyChannel())
        logger.info(
            "Alert channel registered: '%s' (log-only stub)",
            channel_name,
            extra={"component": "main"},
        )
    if not channels:
        # Safety fallback: always have at least one channel
        channels.append(_LogOnlyChannel())
    return channels


def _resolve_zone_map_config(raw_config: Dict[str, Any]) -> Optional[str]:
    """
    Extract the first available zone map config path from raw YAML config.

    Zone maps are per-camera in the YAML; ZoneMap accepts a single config
    file that may contain all camera zones. We use the first entry if only
    one camera has a zone file, or None to use defaults.
    """
    zone_maps: Dict[str, str] = raw_config.get("zone_maps", {})
    if not zone_maps:
        return None
    # Return the first valid path found
    for path in zone_maps.values():
        if isinstance(path, str) and os.path.isfile(path):
            return path
    return None


# ===========================================================================
# Startup / shutdown diagnostics
# ===========================================================================


def _log_startup_summary(
    config_manager: ConfigurationManager,
    pipeline_config: PipelineConfig,
    args: argparse.Namespace,
) -> None:
    """
    Log a comprehensive startup summary including all resolved config values,
    component info, and system details.

    Satisfies the startup diagnostic dump requirement in task 15.1.
    """
    raw = config_manager.raw_config

    logger.info("=" * 70, extra={"component": "main"})
    logger.info("HAZARD DETECTION SYSTEM — STARTUP", extra={"component": "main"})
    logger.info("=" * 70, extra={"component": "main"})

    # System information
    logger.info(
        "System info: version=%s, python=%s, os=%s, platform=%s",
        SYSTEM_VERSION,
        sys.version.split()[0],
        os.name,
        platform.platform(),
        extra={"component": "main"},
    )

    # Config file
    logger.info(
        "Configuration file: %s",
        config_manager.config_path,
        extra={"component": "main"},
    )

    # Camera sequence
    logger.info(
        "Camera sequence (%d cameras): %s",
        len(pipeline_config.camera_sequence),
        pipeline_config.camera_sequence,
        extra={"component": "main"},
    )

    # Per-camera timeout
    logger.info(
        "Per-camera timeout: %ds",
        pipeline_config.per_camera_timeout_seconds,
        extra={"component": "main"},
    )

    # Frame sampler
    fs = pipeline_config.frame_sampler
    logger.info(
        "FrameSampler config: frame_count=%d, timeout_ms=%d, max_retries=%d",
        fs.frame_count,
        fs.timeout_ms,
        fs.max_retries,
        extra={"component": "main"},
    )

    # YOLO
    yolo = pipeline_config.yolo
    logger.info(
        "YOLO config: checkpoint='%s', device='%s', resolution=%d, "
        "confidence_threshold=%.2f",
        yolo.checkpoint_path,
        yolo.device,
        yolo.input_resolution,
        yolo.confidence_threshold,
        extra={"component": "main"},
    )

    # Human detector
    hd = pipeline_config.human_detector
    logger.info(
        "HumanDetector config: confidence_threshold=%.2f",
        hd.confidence_threshold,
        extra={"component": "main"},
    )

    # Container analyzer
    ca = pipeline_config.container_analyzer
    logger.info(
        "ContainerAnalyzer config: confidence_threshold=%.2f, "
        "flipped_ar=%.2f, safe_overlap=%.2f, ground_level=%.2f, "
        "motion_threshold=%.2f, iou_threshold=%.2f",
        ca.confidence_threshold,
        ca.flipped_aspect_ratio_threshold,
        ca.safe_overlap_threshold,
        ca.ground_level_threshold,
        ca.motion_threshold,
        ca.iou_threshold,
        extra={"component": "main"},
    )

    # Alert dispatcher
    ad = pipeline_config.alert_dispatcher
    logger.info(
        "AlertDispatcher config: rate_limit_seconds=%d, channels=%s",
        ad.rate_limit_seconds,
        ad.channels,
        extra={"component": "main"},
    )

    # Defaults applied
    if config_manager.defaults_applied:
        logger.info(
            "Defaults applied for: %s",
            config_manager.defaults_applied,
            extra={"component": "main"},
        )

    # CLI overrides
    if args.log_level:
        logger.info(
            "Log level override via CLI: %s", args.log_level,
            extra={"component": "main"},
        )
    if args.dump_dir:
        logger.info(
            "Dump directory override via CLI: %s", args.dump_dir,
            extra={"component": "main"},
        )

    logger.info("=" * 70, extra={"component": "main"})
    logger.info("Pipeline starting...", extra={"component": "main"})


def _log_shutdown_summary(pipeline: DetectionPipeline) -> None:
    """
    Log final pipeline state, accumulated statistics, and timing summaries.

    Satisfies the shutdown diagnostic dump requirement in task 15.1.
    """
    stats = pipeline.get_statistics()

    logger.info("=" * 70, extra={"component": "main"})
    logger.info("HAZARD DETECTION SYSTEM — SHUTDOWN", extra={"component": "main"})
    logger.info("=" * 70, extra={"component": "main"})
    logger.info(
        "Final statistics: cycles=%d, total_detections=%d, total_hazards=%d",
        stats.get("cycle_count", 0),
        stats.get("total_detections", 0),
        stats.get("total_hazards", 0),
        extra={"component": "main"},
    )
    logger.info(
        "Camera sequence processed: %s",
        stats.get("camera_sequence", []),
        extra={"component": "main"},
    )
    logger.info(
        "Per-camera timeout configured: %ds",
        stats.get("per_camera_timeout_seconds", "N/A"),
        extra={"component": "main"},
    )
    logger.info(
        "Shutdown timestamp: %s",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        extra={"component": "main"},
    )
    logger.info("=" * 70, extra={"component": "main"})


# ===========================================================================
# Signal handling
# ===========================================================================


def _register_signal_handlers(shutdown_event: threading.Event) -> None:
    """
    Register SIGINT and SIGTERM handlers to set the shutdown event.

    On Windows, only SIGINT (Ctrl+C) is reliably available; SIGTERM is
    registered where supported.
    """

    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(
            "Signal %s received — initiating graceful shutdown...",
            sig_name,
            extra={"component": "main"},
        )
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (OSError, ValueError):
        # SIGTERM may not be available on all platforms (e.g., Windows)
        logger.debug(
            "SIGTERM not available on this platform; only SIGINT registered.",
            extra={"component": "main"},
        )


# ===========================================================================
# Main entry point
# ===========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Hazard Detection System.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on clean shutdown, 1 on configuration error,
        2 on unexpected error.
    """
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    try:
        config_manager = ConfigurationManager(config_path=args.config)
        config_manager.load()
    except ConfigurationError as exc:
        # Use basic logging before diagnostics are up
        logging.basicConfig(level=logging.ERROR)
        logging.error("Configuration error: %s", exc)
        return 1

    pipeline_config = config_manager.get_pipeline_config()
    raw_config = config_manager.raw_config

    # ------------------------------------------------------------------
    # 2. Initialise diagnostics (structured logging, dumper, tracer)
    # ------------------------------------------------------------------
    diagnostics_section = raw_config.get("diagnostics", {})

    # Apply CLI overrides
    if args.log_level:
        # Override all per-module log levels
        diagnostics_section = dict(diagnostics_section)
        diagnostics_section["log_levels"] = {
            k: args.log_level
            for k in diagnostics_section.get("log_levels", {}).keys()
        }
        # Also apply globally
        logging.root.setLevel(getattr(logging, args.log_level, logging.INFO))

    if args.dump_dir:
        diagnostics_section = dict(diagnostics_section)
        diagnostics_section["dump_directory"] = args.dump_dir

    diag = initialize_diagnostics(diagnostics_section)
    dumper: DiagnosticDumper = diag["dumper"]
    tracer: PipelineTracer = diag["tracer"]

    # Re-get logger now that logging is properly configured
    global logger
    logger = get_logger("main")

    # ------------------------------------------------------------------
    # 3. Log startup summary
    # ------------------------------------------------------------------
    _log_startup_summary(config_manager, pipeline_config, args)

    # Dump full resolved config as a startup diagnostic snapshot
    dumper.dump(
        stage="startup_config",
        state={
            "version": SYSTEM_VERSION,
            "config_path": config_manager.config_path,
            "camera_sequence": pipeline_config.camera_sequence,
            "per_camera_timeout_seconds": pipeline_config.per_camera_timeout_seconds,
            "frame_count": pipeline_config.frame_sampler.frame_count,
            "yolo_checkpoint": pipeline_config.yolo.checkpoint_path,
            "yolo_device": pipeline_config.yolo.device,
            "yolo_resolution": pipeline_config.yolo.input_resolution,
            "confidence_thresholds": {
                "yolo": pipeline_config.yolo.confidence_threshold,
                "human": pipeline_config.human_detector.confidence_threshold,
                "container": pipeline_config.container_analyzer.confidence_threshold,
            },
            "alert_rate_limit_seconds": pipeline_config.alert_dispatcher.rate_limit_seconds,
            "defaults_applied": config_manager.defaults_applied,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )

    # ------------------------------------------------------------------
    # 4. Build the shutdown event and register signal handlers
    # ------------------------------------------------------------------
    shutdown_event = threading.Event()
    _register_signal_handlers(shutdown_event)

    # ------------------------------------------------------------------
    # 5. Initialise all pipeline components
    # ------------------------------------------------------------------
    try:
        # Frame sampler (real or stub)
        frame_sampler = _build_frame_sampler(pipeline_config, dumper)

        # YOLO detector (real or stub)
        yolo_detector = _build_yolo_detector(pipeline_config)

        # Optical flow analyzer (real or stub)
        flow_analyzer = _build_flow_analyzer()

        # Zone map — attempt to load per-camera zone configs from raw YAML
        zone_map_path = _resolve_zone_map_config(raw_config)
        if zone_map_path:
            logger.info(
                "ZoneMap loading from '%s'", zone_map_path,
                extra={"component": "main"},
            )
        else:
            logger.info(
                "No zone map config file found — defaulting entire FOV "
                "to no-people zone for all cameras.",
                extra={"component": "main"},
            )
        zone_map = ZoneMap(config_path=zone_map_path)

        # Human detector
        human_detector = HumanDetector(
            zone_map=zone_map,
            config=pipeline_config.human_detector,
            diagnostic_dumper=dumper,
        )

        # Container analyzer
        container_analyzer = ContainerAnalyzer(
            flow_analyzer=flow_analyzer,
            config=pipeline_config.container_analyzer,
            dumper=dumper,
        )

        # Alert dispatcher — build channel adapters
        alert_channels = _build_alert_channels(pipeline_config.alert_dispatcher)
        alert_dispatcher = AlertDispatcher(
            channels=alert_channels,
            config=pipeline_config.alert_dispatcher,
        )

        # Camera switcher
        camera_switcher = CameraSwitcher(config=pipeline_config.camera_switcher)

        logger.info(
            "All pipeline components initialised successfully.",
            extra={"component": "main"},
        )

    except Exception as exc:
        logger.error(
            "Failed to initialise pipeline components: %s",
            exc,
            exc_info=True,
            extra={"component": "main"},
        )
        return 2

    # ------------------------------------------------------------------
    # 6. Create and run DetectionPipeline
    # ------------------------------------------------------------------
    pipeline = DetectionPipeline(
        config=pipeline_config,
        frame_sampler=frame_sampler,
        yolo_detector=yolo_detector,
        human_detector=human_detector,
        container_analyzer=container_analyzer,
        alert_dispatcher=alert_dispatcher,
        camera_switcher=camera_switcher,
        shutdown_event=shutdown_event,
        tracer=tracer,
        dumper=dumper,
    )

    logger.info(
        "Entering main detection loop. Send SIGINT (Ctrl+C) or SIGTERM to stop.",
        extra={"component": "main"},
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        # Secondary Ctrl+C while pipeline is winding down
        logger.info(
            "KeyboardInterrupt caught — forcing shutdown.",
            extra={"component": "main"},
        )
        shutdown_event.set()
    except Exception as exc:
        logger.error(
            "Unexpected error in pipeline: %s",
            exc,
            exc_info=True,
            extra={"component": "main"},
        )
        shutdown_event.set()
        _log_shutdown_summary(pipeline)
        return 2

    # ------------------------------------------------------------------
    # 7. Shutdown diagnostics dump
    # ------------------------------------------------------------------
    _log_shutdown_summary(pipeline)

    final_stats = pipeline.get_statistics()
    dumper.dump(
        stage="shutdown_stats",
        state={
            "cycle_count": final_stats.get("cycle_count", 0),
            "total_detections": final_stats.get("total_detections", 0),
            "total_hazards": final_stats.get("total_hazards", 0),
            "camera_sequence": final_stats.get("camera_sequence", []),
            "per_camera_timeout_seconds": final_stats.get(
                "per_camera_timeout_seconds", 0
            ),
            "shutdown_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
    )

    logger.info("Hazard Detection System exited cleanly.", extra={"component": "main"})
    return 0


# ===========================================================================
# Script entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
