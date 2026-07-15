"""
Unit tests for the diagnostics instrumentation module.

Tests cover:
- Structured JSON logging (format, fields, correlation IDs)
- PerformanceTimer context manager (timing accuracy, logging output)
- DiagnosticDumper (state serialization, file output, cleanup)
- PipelineTracer (trace recording, module entry/exit, timing)
- Log level configuration per module

Requirements validated: 6.4, 6.5
"""

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.hazard_detection.diagnostics import (
    JSONFormatter,
    PerformanceTimer,
    DiagnosticDumper,
    PipelineTracer,
    setup_logging,
    get_logger,
    load_diagnostics_config,
    initialize_diagnostics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_dump_dir(tmp_path):
    """Create a temporary dump directory for tests."""
    dump_dir = tmp_path / "test_dumps"
    dump_dir.mkdir()
    return str(dump_dir)


@pytest.fixture
def tmp_log_file(tmp_path):
    """Create a temporary log file path."""
    return str(tmp_path / "test.log")


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before each test to avoid handler accumulation."""
    yield
    # Clean up hazard_detection loggers
    logger = logging.getLogger("hazard_detection")
    logger.handlers.clear()


# =============================================================================
# JSONFormatter Tests
# =============================================================================


class TestJSONFormatter:
    """Tests for the JSON log formatter."""

    def test_format_produces_valid_json(self):
        """Log entries are valid JSON strings."""
        formatter = JSONFormatter(correlation_id="test-123")
        record = logging.LogRecord(
            name="hazard_detection.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_format_includes_required_fields(self):
        """Log entries contain timestamp, level, module, message, and correlation_id."""
        formatter = JSONFormatter(correlation_id="corr-456")
        record = logging.LogRecord(
            name="hazard_detection.frame_sampler",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Frame timeout",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" in parsed
        assert parsed["level"] == "WARNING"
        assert parsed["module"] == "hazard_detection.frame_sampler"
        assert parsed["message"] == "Frame timeout"
        assert parsed["correlation_id"] == "corr-456"

    def test_format_includes_extra_fields(self):
        """Log entries include camera_id and component extras when set."""
        formatter = JSONFormatter(correlation_id="test-789")
        record = logging.LogRecord(
            name="hazard_detection.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Detection found",
            args=None,
            exc_info=None,
        )
        record.camera_id = "cam_03"
        record.component = "yolo_detector"
        record.duration_ms = 42.5

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["camera_id"] == "cam_03"
        assert parsed["component"] == "yolo_detector"
        assert parsed["duration_ms"] == 42.5

    def test_format_generates_correlation_id_when_none(self):
        """A UUID correlation_id is generated if none is provided."""
        formatter = JSONFormatter(correlation_id=None)
        assert formatter.correlation_id is not None
        assert len(formatter.correlation_id) == 36  # UUID format

    def test_format_handles_exception_info(self):
        """Exception info is included in the log entry."""
        formatter = JSONFormatter(correlation_id="exc-test")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="hazard_detection.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=None,
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


# =============================================================================
# setup_logging Tests
# =============================================================================


class TestSetupLogging:
    """Tests for the logging setup function."""

    def test_setup_returns_correlation_id(self):
        """setup_logging returns a valid correlation ID string."""
        corr_id = setup_logging()
        assert isinstance(corr_id, str)
        assert len(corr_id) == 36  # UUID format

    def test_setup_uses_provided_correlation_id(self):
        """setup_logging uses the provided correlation ID."""
        corr_id = setup_logging(correlation_id="my-custom-id")
        assert corr_id == "my-custom-id"

    def test_setup_configures_per_module_levels(self):
        """Per-module log levels are applied correctly."""
        setup_logging(
            log_levels={
                "hazard_detection.frame_sampler": "DEBUG",
                "hazard_detection.yolo_detector": "WARNING",
            }
        )
        fs_logger = logging.getLogger("hazard_detection.frame_sampler")
        yolo_logger = logging.getLogger("hazard_detection.yolo_detector")

        assert fs_logger.level == logging.DEBUG
        assert yolo_logger.level == logging.WARNING

    def test_setup_creates_log_file(self, tmp_log_file):
        """setup_logging creates a log file when specified."""
        setup_logging(log_file=tmp_log_file)
        logger = get_logger("test_module")
        logger.info("Test file logging")

        # Flush handlers
        for handler in logging.getLogger("hazard_detection").handlers:
            handler.flush()

        assert Path(tmp_log_file).exists()

    def test_setup_removes_duplicate_handlers(self):
        """Calling setup_logging multiple times does not duplicate handlers."""
        setup_logging()
        setup_logging()
        root = logging.getLogger("hazard_detection")
        # Should have exactly 1 handler (console)
        assert len(root.handlers) == 1


# =============================================================================
# PerformanceTimer Tests
# =============================================================================


class TestPerformanceTimer:
    """Tests for the PerformanceTimer context manager."""

    def test_timer_measures_elapsed_time(self):
        """Timer measures elapsed time in milliseconds."""
        with PerformanceTimer("test_op") as timer:
            time.sleep(0.05)  # 50ms

        # Should be at least 40ms (allowing for timing variance)
        assert timer.elapsed_ms >= 40.0
        # Should be less than 200ms (generous upper bound)
        assert timer.elapsed_ms < 200.0

    def test_timer_records_start_and_end(self):
        """Timer records start and end perf_counter values."""
        with PerformanceTimer("test_op") as timer:
            time.sleep(0.01)

        assert timer.start_time > 0
        assert timer.end_time > timer.start_time

    def test_timer_does_not_suppress_exceptions(self):
        """Timer context manager does not suppress exceptions."""
        with pytest.raises(ValueError, match="test error"):
            with PerformanceTimer("error_op") as timer:
                raise ValueError("test error")

        # Timer still records time even on exception
        assert timer.elapsed_ms >= 0

    def test_timer_includes_camera_id_in_log(self):
        """Timer logs include camera_id when provided."""
        setup_logging()
        with PerformanceTimer("frame_sampling", camera_id="cam_01") as timer:
            pass

        assert timer.camera_id == "cam_01"
        assert timer.operation == "frame_sampling"

    def test_timer_works_with_zero_duration(self):
        """Timer handles near-zero duration operations."""
        with PerformanceTimer("fast_op") as timer:
            pass  # Essentially instant

        assert timer.elapsed_ms >= 0.0

    def test_timer_uses_custom_logger(self):
        """Timer uses a custom logger when provided."""
        custom_logger = logging.getLogger("custom.test")
        with PerformanceTimer("custom_op", logger=custom_logger) as timer:
            pass

        assert timer.elapsed_ms >= 0.0


# =============================================================================
# DiagnosticDumper Tests
# =============================================================================


class TestDiagnosticDumper:
    """Tests for the DiagnosticDumper state snapshot utility."""

    def test_dump_creates_json_file(self, tmp_dump_dir):
        """dump() creates a JSON file in the dump directory."""
        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
        state = {"detections": 5, "zone_lookups": {"cam_01": "no_people"}}

        filepath = dumper.dump("post_detection", state, camera_id="cam_01")

        assert filepath is not None
        assert Path(filepath).exists()
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["stage"] == "post_detection"
        assert data["camera_id"] == "cam_01"
        assert data["state"]["detections"] == 5

    def test_dump_disabled_returns_none(self, tmp_dump_dir):
        """dump() returns None when disabled."""
        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir, enabled=False)
        result = dumper.dump("test", {"key": "value"})
        assert result is None

    def test_dump_serializes_nested_state(self, tmp_dump_dir):
        """dump() correctly serializes nested dictionaries and lists."""
        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
        state = {
            "detections": [
                {"class": "Human", "confidence": 0.85},
                {"class": "Container - Misaligned", "confidence": 0.72},
            ],
            "flow_scores": {"container_1": 0.45, "container_2": 0.91},
        }

        filepath = dumper.dump("flow_scoring", state)
        assert filepath is not None

        with open(filepath, "r") as f:
            data = json.load(f)
        assert len(data["state"]["detections"]) == 2
        assert data["state"]["flow_scores"]["container_1"] == 0.45

    def test_dump_handles_numpy_arrays(self, tmp_dump_dir):
        """dump() serializes numpy arrays as shape/dtype info."""
        try:
            import numpy as np
            dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
            state = {"frame": np.zeros((480, 640, 3), dtype=np.uint8)}
            filepath = dumper.dump("frame_data", state)
            assert filepath is not None

            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["state"]["frame"]["type"] == "ndarray"
            assert data["state"]["frame"]["shape"] == [480, 640, 3]
        except ImportError:
            pytest.skip("numpy not available")

    def test_dump_includes_correlation_id(self, tmp_dump_dir):
        """dump() includes correlation_id in output."""
        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
        filepath = dumper.dump(
            "test", {"key": "val"}, correlation_id="trace-abc-123"
        )
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["correlation_id"] == "trace-abc-123"

    def test_dump_cleanup_respects_max_dumps(self, tmp_dump_dir):
        """Old dump files are removed when max_dumps is exceeded."""
        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir, max_dumps=3)

        # Create 5 dumps
        for i in range(5):
            dumper.dump(f"stage_{i}", {"index": i})
            time.sleep(0.01)  # Ensure unique timestamps

        # Should only keep 3 files
        dump_files = list(Path(tmp_dump_dir).glob("*.json"))
        assert len(dump_files) <= 3

    def test_dump_creates_directory_if_missing(self, tmp_path):
        """Dumper creates the dump directory if it doesn't exist."""
        new_dir = str(tmp_path / "new" / "nested" / "dumps")
        dumper = DiagnosticDumper(dump_directory=new_dir)
        assert Path(new_dir).exists()

    def test_dump_serializes_dataclasses(self, tmp_dump_dir):
        """dump() can serialize dataclass instances."""
        from src.hazard_detection.models import BBox

        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
        bbox = BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.2)
        state = {"bbox": bbox}

        filepath = dumper.dump("dataclass_test", state)
        assert filepath is not None

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["state"]["bbox"]["x_center"] == 0.5
        assert data["state"]["bbox"]["height"] == 0.2


# =============================================================================
# PipelineTracer Tests
# =============================================================================


class TestPipelineTracer:
    """Tests for the PipelineTracer execution trace recorder."""

    def test_trace_records_modules(self):
        """Tracer records entry and exit of modules."""
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_01")

        tracer.enter_module("frame_sampler", input_info={"camera_id": "cam_01"})
        time.sleep(0.01)
        tracer.exit_module("frame_sampler", output_info={"frame_count": 6})

        trace = tracer.end_trace()

        assert trace is not None
        assert trace["camera_id"] == "cam_01"
        assert len(trace["modules"]) == 1
        assert trace["modules"][0]["module"] == "frame_sampler"
        assert trace["modules"][0]["input_info"] == {"camera_id": "cam_01"}
        assert trace["modules"][0]["output_info"] == {"frame_count": 6}
        assert trace["modules"][0]["duration_ms"] >= 5.0

    def test_trace_multiple_modules(self):
        """Tracer correctly records multiple sequential modules."""
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_02")

        tracer.enter_module("frame_sampler")
        tracer.exit_module("frame_sampler", output_info={"frames": 6})

        tracer.enter_module("yolo_detector")
        tracer.exit_module("yolo_detector", output_info={"detections": 12})

        tracer.enter_module("human_detector")
        tracer.exit_module("human_detector", output_info={"hazards": 1})

        trace = tracer.end_trace()

        assert len(trace["modules"]) == 3
        assert trace["modules"][0]["module"] == "frame_sampler"
        assert trace["modules"][1]["module"] == "yolo_detector"
        assert trace["modules"][2]["module"] == "human_detector"

    def test_trace_total_duration(self):
        """Tracer calculates total trace duration."""
        tracer = PipelineTracer()
        tracer.start_trace()

        tracer.enter_module("fast_module")
        time.sleep(0.02)
        tracer.exit_module("fast_module")

        trace = tracer.end_trace()

        assert trace["total_duration_ms"] >= 15.0

    def test_trace_disabled_returns_none(self):
        """Disabled tracer returns None from all operations."""
        tracer = PipelineTracer(enabled=False)
        tracer.start_trace(camera_id="cam_01")
        tracer.enter_module("test")
        tracer.exit_module("test")
        result = tracer.end_trace()

        assert result is None

    def test_trace_summary(self):
        """get_trace_summary returns per-module timing breakdown."""
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_01")

        tracer.enter_module("module_a")
        time.sleep(0.01)
        tracer.exit_module("module_a")

        tracer.enter_module("module_b")
        time.sleep(0.01)
        tracer.exit_module("module_b")

        tracer.end_trace()
        summary = tracer.get_trace_summary()

        assert summary["camera_id"] == "cam_01"
        assert summary["module_count"] == 2
        assert "module_a" in summary["module_timings"]
        assert "module_b" in summary["module_timings"]
        assert summary["module_timings"]["module_a"]["calls"] == 1

    def test_trace_includes_timestamps(self):
        """Trace includes ISO 8601 start and end timestamps."""
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_01")
        tracer.end_trace()

        trace = tracer.traces[-1]
        assert "start_timestamp" in trace
        assert "end_timestamp" in trace
        assert "T" in trace["start_timestamp"]  # ISO 8601 format

    def test_trace_is_tracing_property(self):
        """is_tracing reflects whether a trace is active."""
        tracer = PipelineTracer()
        assert tracer.is_tracing is False

        tracer.start_trace()
        assert tracer.is_tracing is True

        tracer.end_trace()
        assert tracer.is_tracing is False

    def test_trace_handles_orphaned_modules(self):
        """Modules not explicitly exited are marked as orphaned on trace end."""
        tracer = PipelineTracer()
        tracer.start_trace()
        tracer.enter_module("incomplete_module")
        # Don't call exit_module

        trace = tracer.end_trace()
        assert len(trace["modules"]) == 1
        assert trace["modules"][0]["output_info"]["status"] == "orphaned_on_trace_end"

    def test_trace_with_correlation_id(self):
        """Trace includes the provided correlation ID."""
        tracer = PipelineTracer()
        tracer.start_trace(camera_id="cam_01", correlation_id="corr-xyz")

        trace = tracer.end_trace()
        assert trace["correlation_id"] == "corr-xyz"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestDiagnosticsConfig:
    """Tests for diagnostics configuration loading."""

    def test_load_defaults_when_none(self):
        """load_diagnostics_config returns defaults when config is None."""
        config = load_diagnostics_config(None)

        assert "log_levels" in config
        assert config["dump_enabled"] is True
        assert config["tracing_enabled"] is True
        assert config["dump_directory"] == "diagnostics/dumps"
        assert config["max_dumps"] == 1000
        assert config["log_file"] is None

    def test_load_merges_with_defaults(self):
        """Provided config values override defaults."""
        user_config = {
            "dump_directory": "/custom/dumps",
            "dump_enabled": False,
            "max_dumps": 500,
            "log_levels": {
                "hazard_detection.yolo_detector": "DEBUG",
            },
        }
        config = load_diagnostics_config(user_config)

        assert config["dump_directory"] == "/custom/dumps"
        assert config["dump_enabled"] is False
        assert config["max_dumps"] == 500
        # Merged log levels: user override + defaults
        assert config["log_levels"]["hazard_detection.yolo_detector"] == "DEBUG"
        assert config["log_levels"]["hazard_detection.frame_sampler"] == "INFO"

    def test_initialize_diagnostics_returns_components(self):
        """initialize_diagnostics returns all expected components."""
        result = initialize_diagnostics({"dump_directory": "test_dumps", "dump_enabled": False})

        assert "correlation_id" in result
        assert "dumper" in result
        assert "tracer" in result
        assert isinstance(result["dumper"], DiagnosticDumper)
        assert isinstance(result["tracer"], PipelineTracer)
        assert isinstance(result["correlation_id"], str)


# =============================================================================
# Integration Test
# =============================================================================


class TestDiagnosticsIntegration:
    """Integration tests combining multiple diagnostics components."""

    def test_full_pipeline_instrumentation(self, tmp_dump_dir):
        """Simulate full pipeline instrumentation with timer, dumper, and tracer."""
        setup_logging(correlation_id="integration-test")

        dumper = DiagnosticDumper(dump_directory=tmp_dump_dir)
        tracer = PipelineTracer()

        # Simulate pipeline execution
        tracer.start_trace(camera_id="cam_01", correlation_id="integration-test")

        # Frame sampling stage
        tracer.enter_module("frame_sampler", input_info={"camera_id": "cam_01"})
        with PerformanceTimer("frame_sampling", camera_id="cam_01") as fs_timer:
            time.sleep(0.01)
        tracer.exit_module("frame_sampler", output_info={"frame_count": 6})

        # YOLO inference stage
        tracer.enter_module("yolo_detector", input_info={"frame_count": 6})
        with PerformanceTimer("yolo_inference", camera_id="cam_01") as yolo_timer:
            time.sleep(0.01)
        dumper.dump(
            "post_detection",
            {"detection_count": 8, "classes": ["Human", "Container - Misaligned"]},
            camera_id="cam_01",
            correlation_id="integration-test",
        )
        tracer.exit_module("yolo_detector", output_info={"detections": 8})

        # End trace
        trace = tracer.end_trace()

        # Verify results
        assert trace is not None
        assert len(trace["modules"]) == 2
        assert trace["total_duration_ms"] > 0
        assert fs_timer.elapsed_ms > 0
        assert yolo_timer.elapsed_ms > 0

        # Verify dump file was created
        dump_files = list(Path(tmp_dump_dir).glob("*.json"))
        assert len(dump_files) == 1
