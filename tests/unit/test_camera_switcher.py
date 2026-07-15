"""
Unit tests for Camera Switcher stub component.

Tests cover:
- Stub returns success for valid camera IDs (Requirement 7.1, 7.2)
- Stub returns failure for unrecognized IDs (Requirement 7.5)
- Logging of transitions includes correct metadata (Requirement 7.2)
- Configuration supports up to 16 cameras (Requirement 7.3)

Validates: Requirements 7.1, 7.2, 7.5
"""

import logging

import pytest

from hazard_detection.camera_switcher import CameraSwitcher
from hazard_detection.models import CameraSwitcherConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_config() -> CameraSwitcherConfig:
    """Config with 4 cameras for basic tests."""
    return CameraSwitcherConfig(
        camera_list=["cam_01", "cam_02", "cam_03", "cam_04"],
        connection_types={"cam_01": "rtsp", "cam_02": "rtsp"},
        transition_params={"delay_ms": "0"},
    )


@pytest.fixture
def switcher(basic_config) -> CameraSwitcher:
    """CameraSwitcher instance with basic 4-camera config."""
    return CameraSwitcher(config=basic_config)


@pytest.fixture
def max_camera_config() -> CameraSwitcherConfig:
    """Config with maximum 16 cameras."""
    cameras = [f"cam_{i:02d}" for i in range(1, 17)]
    return CameraSwitcherConfig(camera_list=cameras)


@pytest.fixture
def max_switcher(max_camera_config) -> CameraSwitcher:
    """CameraSwitcher instance with 16-camera config."""
    return CameraSwitcher(config=max_camera_config)


# ---------------------------------------------------------------------------
# Tests: Stub returns success for valid camera IDs (Req 7.1, 7.2)
# ---------------------------------------------------------------------------


class TestTransitionSuccess:
    """Test that the stub returns True for all recognized camera IDs."""

    def test_transition_to_valid_camera_returns_true(self, switcher):
        """A known camera ID should yield a successful transition."""
        assert switcher.transition("cam_01") is True

    def test_transition_to_each_configured_camera(self, switcher):
        """Each camera in the configured list should return success."""
        for cam_id in ["cam_01", "cam_02", "cam_03", "cam_04"]:
            assert switcher.transition(cam_id) is True

    def test_transition_introduces_no_delay(self, switcher):
        """Stub transitions should complete without any intentional delay."""
        import time

        start = time.perf_counter()
        switcher.transition("cam_01")
        elapsed = time.perf_counter() - start
        # Should complete in well under 100ms (no hardware ops)
        assert elapsed < 0.1

    def test_multiple_transitions_to_same_camera(self, switcher):
        """Multiple transitions to the same camera should all succeed."""
        for _ in range(10):
            assert switcher.transition("cam_02") is True


# ---------------------------------------------------------------------------
# Tests: Failure for unrecognized IDs (Req 7.5)
# ---------------------------------------------------------------------------


class TestTransitionFailure:
    """Test that unrecognized camera IDs return False."""

    def test_unrecognized_camera_returns_false(self, switcher):
        """A camera ID not in the list should return failure."""
        assert switcher.transition("cam_99") is False

    def test_empty_string_camera_id_returns_false(self, switcher):
        """An empty string camera ID should return failure."""
        assert switcher.transition("") is False

    def test_none_like_string_returns_false(self, switcher):
        """A string that looks like 'None' but isn't configured should fail."""
        assert switcher.transition("None") is False

    def test_similar_but_wrong_camera_id(self, switcher):
        """Camera IDs that are close but not exact matches should fail."""
        assert switcher.transition("cam_1") is False
        assert switcher.transition("CAM_01") is False
        assert switcher.transition("cam_01 ") is False


# ---------------------------------------------------------------------------
# Tests: Logging includes correct metadata (Req 7.2)
# ---------------------------------------------------------------------------


class TestTransitionLogging:
    """Test that transitions log the correct metadata."""

    def test_successful_transition_logs_target_camera_id(self, switcher, caplog):
        """Successful transition should log the target camera ID."""
        with caplog.at_level(logging.INFO, logger="hazard_detection.camera_switcher"):
            switcher.transition("cam_01")

        # Camera ID is in structured log extras, not the message text
        transition_records = [
            r for r in caplog.records
            if hasattr(r, "target_camera_id") and r.target_camera_id == "cam_01"
        ]
        assert len(transition_records) > 0

    def test_successful_transition_logs_success_status(self, switcher, caplog):
        """Successful transition should include success=True in log extras."""
        with caplog.at_level(logging.INFO, logger="hazard_detection.camera_switcher"):
            switcher.transition("cam_02")

        # Check structured log extra fields
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        transition_records = [
            r for r in info_records if hasattr(r, "target_camera_id")
        ]
        assert len(transition_records) > 0
        record = transition_records[-1]
        assert record.target_camera_id == "cam_02"
        assert record.success is True

    def test_successful_transition_logs_timestamp(self, switcher, caplog):
        """Successful transition should include a timestamp in log extras."""
        with caplog.at_level(logging.INFO, logger="hazard_detection.camera_switcher"):
            switcher.transition("cam_01")

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        transition_records = [
            r for r in info_records if hasattr(r, "timestamp")
        ]
        assert len(transition_records) > 0
        record = transition_records[-1]
        # Timestamp should be an ISO format string
        assert "T" in record.timestamp

    def test_failed_transition_logs_error(self, switcher, caplog):
        """Failed transition should log at ERROR level with target camera."""
        with caplog.at_level(logging.ERROR, logger="hazard_detection.camera_switcher"):
            switcher.transition("cam_unknown")

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) > 0
        record = error_records[0]
        assert hasattr(record, "target_camera_id")
        assert record.target_camera_id == "cam_unknown"
        assert record.success is False

    def test_failed_transition_logs_timestamp(self, switcher, caplog):
        """Failed transition should also include timestamp metadata."""
        with caplog.at_level(logging.ERROR, logger="hazard_detection.camera_switcher"):
            switcher.transition("invalid_cam")

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) > 0
        assert hasattr(error_records[0], "timestamp")
        assert "T" in error_records[0].timestamp


# ---------------------------------------------------------------------------
# Tests: Configuration supports up to 16 cameras (Req 7.3)
# ---------------------------------------------------------------------------


class TestCameraConfiguration:
    """Test configuration constraints for camera list."""

    def test_16_cameras_supported(self, max_switcher):
        """A configuration with exactly 16 cameras should work."""
        assert len(max_switcher.camera_list) == 16

    def test_transition_to_all_16_cameras(self, max_switcher):
        """All 16 configured cameras should be reachable."""
        for i in range(1, 17):
            cam_id = f"cam_{i:02d}"
            assert max_switcher.transition(cam_id) is True

    def test_exceeding_16_cameras_raises_error(self):
        """Attempting to configure more than 16 cameras should raise ValueError."""
        cameras = [f"cam_{i:02d}" for i in range(1, 18)]  # 17 cameras
        with pytest.raises(ValueError, match="at most 16 entries"):
            CameraSwitcherConfig(camera_list=cameras)

    def test_empty_camera_list_is_valid(self):
        """An empty camera list is a valid (though useless) configuration."""
        config = CameraSwitcherConfig(camera_list=[])
        switcher = CameraSwitcher(config=config)
        # No cameras configured, so any transition should fail
        assert switcher.transition("cam_01") is False

    def test_single_camera_configuration(self):
        """A single-camera configuration should work correctly."""
        config = CameraSwitcherConfig(camera_list=["cam_solo"])
        switcher = CameraSwitcher(config=config)
        assert switcher.transition("cam_solo") is True
        assert switcher.transition("cam_other") is False

    def test_config_exposes_camera_list(self, max_switcher):
        """The switcher should expose its configured camera list."""
        expected = [f"cam_{i:02d}" for i in range(1, 17)]
        assert max_switcher.camera_list == expected

    def test_config_preserves_connection_types(self, basic_config):
        """Connection types placeholders should be preserved in config."""
        switcher = CameraSwitcher(config=basic_config)
        assert switcher.config.connection_types == {"cam_01": "rtsp", "cam_02": "rtsp"}

    def test_config_preserves_transition_params(self, basic_config):
        """Transition params placeholders should be preserved in config."""
        switcher = CameraSwitcher(config=basic_config)
        assert switcher.config.transition_params == {"delay_ms": "0"}
