"""
Unit tests for the Frame Sampler component.

Tests cover:
- Capturing exactly N frames per camera (Requirement 1.1)
- Holding at most one FrameSequence in memory (Requirement 1.2)
- Releasing frame data (Requirement 1.3)
- Timeout handling with partial sequence discard (Requirement 1.4)
- Integration with FrameAcquisitionModule buffer (Requirement 1.5)
- Retry logic for frame failures (Requirement 1.6)

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.acquisition.frame_acquisition import (
    CameraSource,
    FrameAcquisitionModule,
)
from hazard_detection.frame_sampler import FrameSampler
from hazard_detection.models import FrameSamplerConfig, FrameSequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_frames():
    """Generate a list of 10 mock frames for testing."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture
def frame_acquisition_module(mock_frames):
    """Create a FrameAcquisitionModule with mock camera sources."""
    module = FrameAcquisitionModule(
        sync_tolerance_ms=50.0,
        buffer_size_per_camera=10,
    )
    # Set up 4 cameras with mock frames
    camera_sources = []
    for cam_id in range(4):
        source = CameraSource(
            camera_id=cam_id,
            source=mock_frames.copy(),
            resolution=(640, 480),
            fps=30,
        )
        camera_sources.append(source)
    module.initialize_cameras(camera_sources)
    return module


@pytest.fixture
def default_config():
    """Default FrameSamplerConfig with 6 frames."""
    return FrameSamplerConfig(frame_count=6, timeout_ms=2000, max_retries=3)


@pytest.fixture
def sampler(frame_acquisition_module, default_config):
    """FrameSampler with default configuration and mock cameras."""
    return FrameSampler(
        frame_acquisition=frame_acquisition_module,
        config=default_config,
    )


@pytest.fixture
def min_frame_config():
    """Config with minimum frame count (5)."""
    return FrameSamplerConfig(frame_count=5, timeout_ms=2000, max_retries=3)


@pytest.fixture
def max_frame_config():
    """Config with maximum frame count (8)."""
    return FrameSamplerConfig(frame_count=8, timeout_ms=2000, max_retries=3)


# ---------------------------------------------------------------------------
# Tests: Capture exactly N frames (Requirement 1.1)
# ---------------------------------------------------------------------------


class TestFrameCapture:
    """Test that FrameSampler captures exactly the configured number of frames."""

    def test_captures_6_frames_with_default_config(self, sampler):
        """Default config should capture exactly 6 frames."""
        result = sampler.sample("0")
        assert result is not None
        assert result.frame_count == 6
        assert len(result.frames) == 6
        assert len(result.timestamps) == 6

    def test_captures_5_frames_with_min_config(self, frame_acquisition_module, min_frame_config):
        """Minimum config should capture exactly 5 frames."""
        sampler = FrameSampler(frame_acquisition_module, min_frame_config)
        result = sampler.sample("0")
        assert result is not None
        assert result.frame_count == 5

    def test_captures_8_frames_with_max_config(self, frame_acquisition_module, max_frame_config):
        """Maximum config should capture exactly 8 frames."""
        sampler = FrameSampler(frame_acquisition_module, max_frame_config)
        result = sampler.sample("0")
        assert result is not None
        assert result.frame_count == 8

    def test_captures_frames_as_numpy_arrays(self, sampler):
        """Each frame should be a numpy ndarray."""
        result = sampler.sample("0")
        assert result is not None
        for frame in result.frames:
            assert isinstance(frame, np.ndarray)

    def test_captures_correct_camera_id(self, sampler):
        """The FrameSequence should carry the correct camera_id."""
        result = sampler.sample("0")
        assert result is not None
        assert result.camera_id == "0"

    def test_timestamps_are_monotonically_increasing(self, sampler):
        """Timestamps within a sequence should be increasing."""
        result = sampler.sample("0")
        assert result is not None
        for i in range(1, len(result.timestamps)):
            assert result.timestamps[i] > result.timestamps[i - 1]


# ---------------------------------------------------------------------------
# Tests: At most one FrameSequence in memory (Requirement 1.2)
# ---------------------------------------------------------------------------


class TestMemoryConstraint:
    """Test that at most one FrameSequence is held in memory."""

    def test_second_sample_releases_first(self, frame_acquisition_module):
        """Calling sample() again should release the previous sequence."""
        config = FrameSamplerConfig(frame_count=5, timeout_ms=2000, max_retries=3)
        sampler = FrameSampler(frame_acquisition_module, config)

        first = sampler.sample("0")
        assert first is not None
        assert sampler.current_sequence is first

        # Second sample should replace
        second = sampler.sample("1")
        assert second is not None
        assert sampler.current_sequence is second
        assert sampler.current_sequence is not first

    def test_current_sequence_is_none_initially(self, sampler):
        """Before sampling, current_sequence should be None."""
        assert sampler.current_sequence is None

    def test_current_sequence_set_after_sample(self, sampler):
        """After sampling, current_sequence should be the returned sequence."""
        result = sampler.sample("0")
        assert sampler.current_sequence is result


# ---------------------------------------------------------------------------
# Tests: Release frame data (Requirement 1.3)
# ---------------------------------------------------------------------------


class TestRelease:
    """Test explicit frame data release."""

    def test_release_clears_current_sequence(self, sampler):
        """release() should set current_sequence to None."""
        sampler.sample("0")
        assert sampler.current_sequence is not None
        sampler.release()
        assert sampler.current_sequence is None

    def test_release_is_idempotent(self, sampler):
        """Calling release() when no sequence is held should not raise."""
        sampler.release()  # No sequence held
        assert sampler.current_sequence is None

    def test_release_after_sample_frees_memory(self, sampler):
        """After release, the frame data should be eligible for GC."""
        result = sampler.sample("0")
        assert result is not None
        sampler.release()
        # The sampler no longer holds a reference
        assert sampler.current_sequence is None


# ---------------------------------------------------------------------------
# Tests: Timeout handling (Requirement 1.4)
# ---------------------------------------------------------------------------


class TestTimeout:
    """Test timeout handling with partial sequence discard."""

    def test_timeout_returns_none(self, frame_acquisition_module):
        """When timeout is exceeded, sample() should return None."""
        # Use very short timeout
        config = FrameSamplerConfig(frame_count=6, timeout_ms=1, max_retries=3)
        sampler = FrameSampler(frame_acquisition_module, config)

        # Patch acquire_frame to be slow
        original_acquire = frame_acquisition_module.acquire_frame

        def slow_acquire(camera_id):
            time.sleep(0.01)  # 10ms - exceeds 1ms timeout
            return original_acquire(camera_id)

        with patch.object(frame_acquisition_module, "acquire_frame", side_effect=slow_acquire):
            result = sampler.sample("0")

        assert result is None

    def test_timeout_discards_partial_sequence(self, frame_acquisition_module):
        """On timeout, no partial FrameSequence should be retained."""
        config = FrameSamplerConfig(frame_count=6, timeout_ms=1, max_retries=3)
        sampler = FrameSampler(frame_acquisition_module, config)

        def slow_acquire(camera_id):
            time.sleep(0.01)
            return None

        with patch.object(frame_acquisition_module, "acquire_frame", side_effect=slow_acquire):
            result = sampler.sample("0")

        assert result is None
        assert sampler.current_sequence is None

    def test_normal_operation_within_timeout(self, sampler):
        """Normal mock frames should complete well within 2000ms timeout."""
        result = sampler.sample("0")
        assert result is not None
        assert result.frame_count == 6


# ---------------------------------------------------------------------------
# Tests: Retry logic (Requirement 1.6)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test retry logic for individual frame failures."""

    def test_successful_after_retries(self, frame_acquisition_module, default_config):
        """Frame acquisition that fails then succeeds should still produce a sequence."""
        sampler = FrameSampler(frame_acquisition_module, default_config)

        call_count = [0]
        original_acquire = frame_acquisition_module.acquire_frame

        def flaky_acquire(camera_id):
            call_count[0] += 1
            # Fail every other call, but succeed eventually
            if call_count[0] % 2 == 1:
                return None
            return original_acquire(camera_id)

        with patch.object(frame_acquisition_module, "acquire_frame", side_effect=flaky_acquire):
            result = sampler.sample("0")

        assert result is not None
        assert result.frame_count == 6

    def test_all_retries_exhausted_returns_none(self, frame_acquisition_module, default_config):
        """When all retries fail for a frame, sample() should return None."""
        sampler = FrameSampler(frame_acquisition_module, default_config)

        # Always fail
        with patch.object(frame_acquisition_module, "acquire_frame", return_value=None):
            result = sampler.sample("0")

        assert result is None

    def test_retry_count_respects_max_retries(self, frame_acquisition_module):
        """Should attempt exactly 1 + max_retries times per frame before giving up."""
        config = FrameSamplerConfig(frame_count=6, timeout_ms=2000, max_retries=3)
        sampler = FrameSampler(frame_acquisition_module, config)

        call_count = [0]

        def always_fail(camera_id):
            call_count[0] += 1
            return None

        with patch.object(frame_acquisition_module, "acquire_frame", side_effect=always_fail):
            sampler.sample("0")

        # Should be called 1 initial + 3 retries = 4 times for the first frame
        assert call_count[0] == 4


# ---------------------------------------------------------------------------
# Tests: Camera ID resolution
# ---------------------------------------------------------------------------


class TestCameraIdResolution:
    """Test that various camera ID formats are resolved correctly."""

    def test_numeric_string_camera_id(self, sampler):
        """Numeric string '0' should resolve to camera index 0."""
        result = sampler.sample("0")
        assert result is not None

    def test_prefixed_camera_id(self, sampler):
        """Prefixed camera ID like 'cam_0' should resolve correctly."""
        result = sampler.sample("cam_0")
        assert result is not None

    def test_invalid_camera_id_returns_none(self, sampler):
        """A completely unresolvable camera ID should return None."""
        result = sampler.sample("invalid_no_digits")
        assert result is None

    def test_out_of_range_camera_id_returns_none(self, sampler):
        """Camera ID beyond range 0-3 should return None."""
        result = sampler.sample("99")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Integration with FrameAcquisitionModule (Requirement 1.5)
# ---------------------------------------------------------------------------


class TestAcquisitionIntegration:
    """Test integration with the existing FrameAcquisitionModule buffer."""

    def test_uses_frame_acquisition_acquire_frame(self, frame_acquisition_module, default_config):
        """Should call acquire_frame on the FrameAcquisitionModule."""
        sampler = FrameSampler(frame_acquisition_module, default_config)

        with patch.object(
            frame_acquisition_module, "acquire_frame", wraps=frame_acquisition_module.acquire_frame
        ) as mock_acquire:
            sampler.sample("0")

        assert mock_acquire.call_count >= 6  # At least once per frame

    def test_frame_data_comes_from_acquisition_module(self, frame_acquisition_module, default_config):
        """Frames in the sequence should be the actual data from the module."""
        sampler = FrameSampler(frame_acquisition_module, default_config)
        result = sampler.sample("0")
        assert result is not None
        # Each frame should be a valid numpy array with expected shape
        for frame in result.frames:
            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8
