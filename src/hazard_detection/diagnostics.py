"""
Debugging and diagnostic instrumentation for the Hazard Detection System.

Provides structured logging (JSON-formatted), performance timing, intermediate
state dumps, and pipeline execution tracing for troubleshooting and monitoring.

This module is standalone with minimal imports — it uses only Python's built-in
logging, json, time, uuid, contextlib, os, and pathlib modules.

Requirements covered:
- 6.4: Log start and end time for each camera's processing cycle
- 6.5: Log errors with module name and camera identifier on exceptions
"""

import contextlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Structured JSON Log Formatter
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON entries with timestamps, module names,
    and correlation IDs for structured log analysis.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Args:
            correlation_id: Optional correlation ID to include in all log entries.
                          If None, a new UUID is generated per formatter instance.
        """
        super().__init__()
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "correlation_id": self.correlation_id,
        }

        # Include extra fields if present
        if hasattr(record, "camera_id"):
            log_entry["camera_id"] = record.camera_id
        if hasattr(record, "component"):
            log_entry["component"] = record.component
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data"):
            log_entry["extra_data"] = record.extra_data

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(
    log_levels: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None,
    log_file: Optional[str] = None,
) -> str:
    """
    Configure structured JSON logging for the hazard detection system.

    Sets up logging with JSON formatting, per-module log levels, and
    optional file output. Returns the correlation ID used for this session.

    Args:
        log_levels: Dictionary mapping module names to log level strings.
                   Example: {"hazard_detection.yolo_detector": "DEBUG",
                             "hazard_detection.frame_sampler": "INFO"}
                   If None, all modules default to INFO level.
        correlation_id: Optional correlation ID for all log entries.
                       If None, a new UUID is generated.
        log_file: Optional file path for log output. If None, logs to stderr only.

    Returns:
        The correlation ID string used for this logging session.
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    formatter = JSONFormatter(correlation_id=correlation_id)

    # Configure root logger for hazard_detection namespace
    root_logger = logging.getLogger("hazard_detection")
    root_logger.setLevel(logging.DEBUG)  # Allow all levels, filter at handler/module level

    # Remove existing handlers to avoid duplicates on re-initialization
    root_logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Default console level
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

    # Apply per-module log levels
    if log_levels:
        for module_name, level_str in log_levels.items():
            level = getattr(logging, level_str.upper(), logging.INFO)
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(level)

    return correlation_id


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the hazard_detection namespace.

    Args:
        module_name: The module name (e.g., "frame_sampler", "yolo_detector").

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(f"hazard_detection.{module_name}")


# =============================================================================
# PerformanceTimer Context Manager
# =============================================================================


class PerformanceTimer:
    """
    Context manager for measuring per-module timing breakdowns.

    Records elapsed time for pipeline stages (frame sampling, YOLO inference,
    hazard logic, alert dispatch) and logs the result as a structured entry.

    Usage:
        with PerformanceTimer("yolo_inference", camera_id="cam_01") as timer:
            # ... perform inference ...
        print(timer.elapsed_ms)  # time in milliseconds
    """

    def __init__(
        self,
        operation: str,
        camera_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            operation: Name of the operation being timed (e.g., "frame_sampling",
                      "yolo_inference", "hazard_logic", "alert_dispatch").
            camera_id: Optional camera identifier for context.
            logger: Optional logger instance. If None, uses the diagnostics logger.
        """
        self.operation = operation
        self.camera_id = camera_id
        self._logger = logger or get_logger("performance")
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "PerformanceTimer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop the timer and log the result."""
        self._end_time = time.perf_counter()
        self.elapsed_ms = (self._end_time - self._start_time) * 1000.0

        extra = {
            "duration_ms": round(self.elapsed_ms, 3),
            "component": self.operation,
        }
        if self.camera_id:
            extra["camera_id"] = self.camera_id

        self._logger.info(
            f"{self.operation} completed in {self.elapsed_ms:.3f}ms"
            + (f" [camera: {self.camera_id}]" if self.camera_id else ""),
            extra=extra,
        )

        # Do not suppress exceptions
        return False

    @property
    def start_time(self) -> float:
        """Return the start timestamp (perf_counter value)."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """Return the end timestamp (perf_counter value)."""
        return self._end_time


# =============================================================================
# DiagnosticDumper
# =============================================================================


class DiagnosticDumper:
    """
    Saves intermediate pipeline state snapshots as JSON files for debugging.

    Captures and persists detections, zone lookups, flow scores, and other
    pipeline state at configurable points during execution.

    State files are saved to a configurable dump directory with timestamped
    filenames for easy correlation with log entries.
    """

    def __init__(
        self,
        dump_directory: str = "diagnostics/dumps",
        enabled: bool = True,
        max_dumps: int = 1000,
    ):
        """
        Args:
            dump_directory: Directory path where JSON dump files are saved.
            enabled: Whether dumping is active. Set False to disable without
                    removing dump calls from code.
            max_dumps: Maximum number of dump files to retain. Oldest are removed
                      when exceeded.
        """
        self.dump_directory = Path(dump_directory)
        self.enabled = enabled
        self.max_dumps = max_dumps
        self._logger = get_logger("diagnostic_dumper")
        self._dump_count = 0

        if self.enabled:
            self.dump_directory.mkdir(parents=True, exist_ok=True)

    def dump(
        self,
        stage: str,
        state: Dict[str, Any],
        camera_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a pipeline state snapshot to a JSON file.

        Args:
            stage: Pipeline stage name (e.g., "post_detection", "zone_lookup",
                  "flow_scoring", "temporal_confirmation").
            state: Dictionary of state data to serialize. Values should be
                  JSON-serializable (use _serialize_value for complex objects).
            camera_id: Optional camera identifier for context.
            correlation_id: Optional correlation ID to link with log entries.

        Returns:
            The file path of the dump file, or None if dumping is disabled.
        """
        if not self.enabled:
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        filename = f"{timestamp}_{stage}"
        if camera_id:
            filename += f"_{camera_id}"
        filename += ".json"

        dump_data = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "stage": stage,
            "camera_id": camera_id,
            "correlation_id": correlation_id,
            "state": self._serialize_state(state),
        }

        filepath = self.dump_directory / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dump_data, f, indent=2, default=str)
            self._dump_count += 1
            self._logger.debug(
                f"Diagnostic dump saved: {filepath}",
                extra={"component": "diagnostic_dumper", "camera_id": camera_id or ""},
            )
            self._cleanup_old_dumps()
            return str(filepath)
        except (OSError, TypeError) as e:
            self._logger.warning(
                f"Failed to save diagnostic dump to {filepath}: {e}",
                extra={"component": "diagnostic_dumper"},
            )
            return None

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively serialize state dictionary for JSON output.

        Handles common non-serializable types (numpy arrays, dataclasses, etc.)
        by converting them to basic Python types.
        """
        serialized = {}
        for key, value in state.items():
            serialized[key] = self._serialize_value(value)
        return serialized

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to a JSON-compatible type."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "__dataclass_fields__"):
            # Dataclass: convert to dict
            return {
                field_name: self._serialize_value(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
        if hasattr(value, "shape"):
            # Numpy array: return shape info
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        # Fallback: string representation
        return str(value)

    def _cleanup_old_dumps(self) -> None:
        """Remove oldest dump files if max_dumps is exceeded."""
        if not self.dump_directory.exists():
            return

        dump_files = sorted(self.dump_directory.glob("*.json"))
        if len(dump_files) > self.max_dumps:
            files_to_remove = dump_files[: len(dump_files) - self.max_dumps]
            for f in files_to_remove:
                try:
                    f.unlink()
                except OSError:
                    pass


# =============================================================================
# PipelineTracer
# =============================================================================


class PipelineTracer:
    """
    Records execution trace of the detection pipeline with entry/exit of
    each module, input/output shapes, and timing information.

    Provides a structured trace log that shows the flow of data through
    the pipeline stages, useful for debugging and performance profiling.

    Usage:
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_01")

        tracer.enter_module("frame_sampler", input_info={"camera_id": "cam_01"})
        # ... do work ...
        tracer.exit_module("frame_sampler", output_info={"frame_count": 6})

        trace = tracer.end_trace()
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: Whether tracing is active. Set False to disable without
                    removing trace calls from code.
        """
        self.enabled = enabled
        self._logger = get_logger("pipeline_tracer")
        self._current_trace: Optional[Dict[str, Any]] = None
        self._module_stack: List[Dict[str, Any]] = []
        self._traces: List[Dict[str, Any]] = []

    def start_trace(
        self,
        camera_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Begin a new pipeline execution trace.

        Args:
            camera_id: Camera being processed in this trace.
            correlation_id: ID to correlate with log entries.
        """
        if not self.enabled:
            return

        self._current_trace = {
            "trace_id": str(uuid.uuid4()),
            "camera_id": camera_id,
            "correlation_id": correlation_id,
            "start_time": time.perf_counter(),
            "start_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "modules": [],
        }
        self._module_stack = []

        self._logger.debug(
            f"Pipeline trace started for camera: {camera_id}",
            extra={"component": "pipeline_tracer", "camera_id": camera_id or ""},
        )

    def enter_module(
        self,
        module_name: str,
        input_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record entry into a pipeline module.

        Args:
            module_name: Name of the module being entered (e.g., "frame_sampler",
                        "yolo_detector", "human_detector", "container_analyzer",
                        "alert_dispatcher").
            input_info: Optional dictionary describing the input data
                       (e.g., shapes, counts, identifiers).
        """
        if not self.enabled or self._current_trace is None:
            return

        module_entry = {
            "module": module_name,
            "enter_time": time.perf_counter(),
            "enter_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "input_info": input_info or {},
            "output_info": {},
            "exit_time": None,
            "duration_ms": None,
        }

        self._module_stack.append(module_entry)

        self._logger.debug(
            f"Entering module: {module_name}",
            extra={
                "component": "pipeline_tracer",
                "extra_data": input_info,
            },
        )

    def exit_module(
        self,
        module_name: str,
        output_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record exit from a pipeline module.

        Args:
            module_name: Name of the module being exited. Must match the most
                        recent enter_module call.
            output_info: Optional dictionary describing the output data
                        (e.g., detection counts, hazard events produced).
        """
        if not self.enabled or self._current_trace is None:
            return

        if not self._module_stack:
            self._logger.warning(
                f"exit_module called for '{module_name}' but no module is on the stack",
                extra={"component": "pipeline_tracer"},
            )
            return

        module_entry = self._module_stack.pop()

        if module_entry["module"] != module_name:
            self._logger.warning(
                f"exit_module mismatch: expected '{module_entry['module']}', "
                f"got '{module_name}'",
                extra={"component": "pipeline_tracer"},
            )

        exit_time = time.perf_counter()
        module_entry["exit_time"] = exit_time
        module_entry["duration_ms"] = round(
            (exit_time - module_entry["enter_time"]) * 1000.0, 3
        )
        module_entry["output_info"] = output_info or {}

        self._current_trace["modules"].append(module_entry)

        self._logger.debug(
            f"Exiting module: {module_name} ({module_entry['duration_ms']:.3f}ms)",
            extra={
                "component": "pipeline_tracer",
                "duration_ms": module_entry["duration_ms"],
                "extra_data": output_info,
            },
        )

    def end_trace(self) -> Optional[Dict[str, Any]]:
        """
        End the current pipeline trace and return the complete trace record.

        Returns:
            Dictionary containing the full trace with all module entries,
            or None if tracing is disabled or no trace was started.
        """
        if not self.enabled or self._current_trace is None:
            return None

        end_time = time.perf_counter()
        self._current_trace["end_time"] = end_time
        self._current_trace["end_timestamp"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        self._current_trace["total_duration_ms"] = round(
            (end_time - self._current_trace["start_time"]) * 1000.0, 3
        )

        # Flush any remaining modules on the stack (shouldn't happen in normal flow)
        while self._module_stack:
            orphan = self._module_stack.pop()
            orphan["exit_time"] = end_time
            orphan["duration_ms"] = round(
                (end_time - orphan["enter_time"]) * 1000.0, 3
            )
            orphan["output_info"] = {"status": "orphaned_on_trace_end"}
            self._current_trace["modules"].append(orphan)

        trace = self._current_trace
        self._traces.append(trace)
        self._current_trace = None

        self._logger.debug(
            f"Pipeline trace ended: {trace['total_duration_ms']:.3f}ms total, "
            f"{len(trace['modules'])} modules",
            extra={
                "component": "pipeline_tracer",
                "duration_ms": trace["total_duration_ms"],
            },
        )

        return trace

    def get_trace_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current trace (if active) or the most recent trace.

        Returns:
            Dictionary with trace summary including per-module timing breakdown.
        """
        trace = self._current_trace or (self._traces[-1] if self._traces else None)
        if trace is None:
            return {"status": "no_trace_available"}

        module_timings = {}
        for module_entry in trace.get("modules", []):
            name = module_entry["module"]
            duration = module_entry.get("duration_ms", 0.0)
            if name not in module_timings:
                module_timings[name] = {"total_ms": 0.0, "calls": 0}
            module_timings[name]["total_ms"] += duration
            module_timings[name]["calls"] += 1

        return {
            "trace_id": trace.get("trace_id"),
            "camera_id": trace.get("camera_id"),
            "total_duration_ms": trace.get("total_duration_ms"),
            "module_count": len(trace.get("modules", [])),
            "module_timings": module_timings,
        }

    @property
    def traces(self) -> List[Dict[str, Any]]:
        """Return all completed traces."""
        return list(self._traces)

    @property
    def is_tracing(self) -> bool:
        """Return whether a trace is currently active."""
        return self._current_trace is not None


# =============================================================================
# Diagnostic Configuration Loader
# =============================================================================


def load_diagnostics_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load diagnostics configuration from a YAML config section.

    Expected config structure:
        diagnostics:
          log_levels:
            hazard_detection.frame_sampler: DEBUG
            hazard_detection.yolo_detector: INFO
            hazard_detection.human_detector: INFO
            hazard_detection.container_analyzer: DEBUG
            hazard_detection.alert_dispatcher: INFO
            hazard_detection.performance: DEBUG
            hazard_detection.pipeline_tracer: DEBUG
            hazard_detection.diagnostic_dumper: DEBUG
          dump_directory: "diagnostics/dumps"
          dump_enabled: true
          max_dumps: 1000
          tracing_enabled: true
          log_file: null

    Args:
        config: Dictionary with diagnostics configuration section.
               If None, returns defaults.

    Returns:
        Resolved diagnostics configuration dictionary.
    """
    defaults = {
        "log_levels": {
            "hazard_detection.frame_sampler": "INFO",
            "hazard_detection.yolo_detector": "INFO",
            "hazard_detection.human_detector": "INFO",
            "hazard_detection.container_analyzer": "INFO",
            "hazard_detection.alert_dispatcher": "INFO",
            "hazard_detection.performance": "INFO",
            "hazard_detection.pipeline_tracer": "DEBUG",
            "hazard_detection.diagnostic_dumper": "DEBUG",
        },
        "dump_directory": "diagnostics/dumps",
        "dump_enabled": True,
        "max_dumps": 1000,
        "tracing_enabled": True,
        "log_file": None,
    }

    if config is None:
        return defaults

    # Merge with defaults
    resolved = dict(defaults)
    if "log_levels" in config and isinstance(config["log_levels"], dict):
        resolved["log_levels"] = {**defaults["log_levels"], **config["log_levels"]}
    if "dump_directory" in config:
        resolved["dump_directory"] = config["dump_directory"]
    if "dump_enabled" in config:
        resolved["dump_enabled"] = bool(config["dump_enabled"])
    if "max_dumps" in config:
        resolved["max_dumps"] = int(config["max_dumps"])
    if "tracing_enabled" in config:
        resolved["tracing_enabled"] = bool(config["tracing_enabled"])
    if "log_file" in config:
        resolved["log_file"] = config["log_file"]

    return resolved


# =============================================================================
# Convenience: Initialize all diagnostics from config
# =============================================================================


def initialize_diagnostics(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Initialize all diagnostic components from configuration.

    This is the main entry point for setting up diagnostics. It configures
    logging, creates a DiagnosticDumper, and creates a PipelineTracer.

    Args:
        config: Diagnostics configuration section (see load_diagnostics_config).

    Returns:
        Dictionary containing initialized components:
        - "correlation_id": The session correlation ID
        - "dumper": DiagnosticDumper instance
        - "tracer": PipelineTracer instance
    """
    resolved = load_diagnostics_config(config)

    correlation_id = setup_logging(
        log_levels=resolved["log_levels"],
        log_file=resolved.get("log_file"),
    )

    dumper = DiagnosticDumper(
        dump_directory=resolved["dump_directory"],
        enabled=resolved["dump_enabled"],
        max_dumps=resolved["max_dumps"],
    )

    tracer = PipelineTracer(enabled=resolved["tracing_enabled"])

    return {
        "correlation_id": correlation_id,
        "dumper": dumper,
        "tracer": tracer,
    }
