"""
Frame Sampler Module

Wraps the existing FrameAcquisitionModule to capture fixed-length frame sequences
(5-8 frames) for a single camera. Enforces the memory constraint of holding at most
one FrameSequence in memory at a time.

Includes retry logic (up to 3 retries per frame failure), timeout handling (2000ms),
performance timing instrumentation, and diagnostic metadata collection.

Requirements covered:
- 1.1: Capture configured frame count (5-8) per camera
- 1.2: Hold at most one FrameSequence in memory at any time
- 1.3: Release frame data before camera transition
- 1.4: Timeout handling (2000ms) with partial sequence discard and logging
- 1.5: Reuse FrameAcquisitionModule buffer infrastructure
- 1.6: Retry acquisition up to 3 times per frame failure
"""

import sys
import time
from typing import Optional

import numpy as np

from acquisition.frame_acquisition import FrameAcquisitionModule
from hazard_detection.models import FrameSamplerConfig, FrameSequence
from hazard_detection.diagnostics import get_logger, PerformanceTimer


logger = get_logger("frame_sampler")


class FrameSampler:
    """
    Captures fixed-length frame sequences from a camera feed using the
    existing FrameAcquisitionModule buffer infrastructure.

    Key constraints:
    - Captures exactly N frames (5-8) per camera as configured
    - Holds at most one FrameSequence in memory at a time
    - Retries individual frame failures up to max_retries (default 3)
    - Enforces a timeout (default 2000ms) - discards partial sequence on timeout
    - Instruments performance timing for each frame and total sequence
    - Collects diagnostic metadata: timestamps, memory usage, retry counts
    """

    def __init__(
        self,
        frame_acquisition: FrameAcquisitionModule,
        config: FrameSamplerConfig,
    ):
        """
        Initialize the FrameSampler.

        Args:
            frame_acquisition: Existing FrameAcquisitionModule instance providing
                             buffer infrastructure and frame acquisition.
            config: FrameSamplerConfig with frame_count (5-8), timeout_ms (2000),
                   max_retries (3).
        """
        self._acquisition = frame_acquisition
        self._config = config
        self._current_sequence: Optional[FrameSequence] = None

        # Diagnostic counters (reset per sample call)
        self._retry_counts: dict = {}
        self._frame_timings: list = []
        self._total_retries: int = 0

    @property
    def config(self) -> FrameSamplerConfig:
        """Return the current configuration."""
        return self._config

    @property
    def current_sequence(self) -> Optional[FrameSequence]:
        """Return the currently held FrameSequence, or None if released."""
        return self._current_sequence

    def sample(self, camera_id: str) -> Optional[FrameSequence]:
        """
        Capture a frame sequence from the specified camera.

        Acquires exactly config.frame_count frames from the camera feed.
        Retries individual frame failures up to config.max_retries times.
        If the total sampling time exceeds config.timeout_ms, discards the
        partial sequence and returns None.

        Enforces the constraint of holding at most one FrameSequence in memory.
        If a previous sequence exists, it is released before sampling.

        Args:
            camera_id: Identifier of the camera to sample from.

        Returns:
            A FrameSequence containing the captured frames, or None if the
            camera feed is unavailable or a timeout occurred.
        """
        # Release any previously held sequence (Requirement 1.2)
        if self._current_sequence is not None:
            self.release()

        # Reset diagnostics for this sampling session
        self._retry_counts = {}
        self._frame_timings = []
        self._total_retries = 0

        frames: list = []
        timestamps: list = []
        timeout_seconds = self._config.timeout_ms / 1000.0
        start_time = time.perf_counter()

        logger.info(
            f"Starting frame sampling for camera '{camera_id}', "
            f"target frames: {self._config.frame_count}, "
            f"timeout: {self._config.timeout_ms}ms",
            extra={"camera_id": camera_id, "component": "frame_sampler"},
        )

        with PerformanceTimer(
            "frame_sequence_sampling", camera_id=camera_id, logger=logger
        ) as sequence_timer:
            for frame_idx in range(self._config.frame_count):
                # Check timeout before attempting each frame
                elapsed = time.perf_counter() - start_time
                if elapsed >= timeout_seconds:
                    # Timeout: discard partial sequence, log failure (Req 1.4)
                    logger.warning(
                        f"Timeout ({self._config.timeout_ms}ms) exceeded during "
                        f"frame sampling for camera '{camera_id}'. "
                        f"Frames captured before timeout: {len(frames)}",
                        extra={
                            "camera_id": camera_id,
                            "component": "frame_sampler",
                        },
                    )
                    # Discard partial frames
                    frames.clear()
                    timestamps.clear()
                    return None

                # Attempt to acquire a single frame with retries (Req 1.6)
                frame_result = self._acquire_frame_with_retry(
                    camera_id, frame_idx, start_time, timeout_seconds
                )

                if frame_result is None:
                    # Check if it was due to timeout
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= timeout_seconds:
                        logger.warning(
                            f"Timeout ({self._config.timeout_ms}ms) exceeded during "
                            f"retry for camera '{camera_id}'. "
                            f"Frames captured before timeout: {len(frames)}",
                            extra={
                                "camera_id": camera_id,
                                "component": "frame_sampler",
                            },
                        )
                    else:
                        logger.warning(
                            f"Frame acquisition failed for camera '{camera_id}' "
                            f"at frame index {frame_idx} after "
                            f"{self._config.max_retries} retries. "
                            f"Frames captured: {len(frames)}",
                            extra={
                                "camera_id": camera_id,
                                "component": "frame_sampler",
                            },
                        )
                    # Discard partial sequence (Req 1.4)
                    frames.clear()
                    timestamps.clear()
                    return None

                frame_data, frame_timestamp = frame_result
                frames.append(frame_data)
                timestamps.append(frame_timestamp)

        # Build the FrameSequence (Req 1.1)
        self._current_sequence = FrameSequence(
            frames=frames,
            camera_id=camera_id,
            timestamps=timestamps,
        )

        # Log diagnostic metadata
        memory_usage = self._estimate_memory_usage(frames)
        logger.info(
            f"Frame sequence complete for camera '{camera_id}': "
            f"{len(frames)} frames, "
            f"total time: {sequence_timer.elapsed_ms:.1f}ms, "
            f"memory: {memory_usage / 1024:.1f}KB, "
            f"total retries: {self._total_retries}",
            extra={
                "camera_id": camera_id,
                "component": "frame_sampler",
                "extra_data": {
                    "frame_count": len(frames),
                    "total_time_ms": round(sequence_timer.elapsed_ms, 3),
                    "memory_bytes": memory_usage,
                    "retry_counts": self._retry_counts,
                    "frame_timings_ms": self._frame_timings,
                    "timestamps": timestamps,
                },
            },
        )

        return self._current_sequence

    def release(self) -> None:
        """
        Explicitly release frame data from memory (Requirement 1.3).

        Clears the held FrameSequence to free memory before camera transition.
        """
        if self._current_sequence is not None:
            camera_id = self._current_sequence.camera_id
            frame_count = self._current_sequence.frame_count
            # Clear frame data
            self._current_sequence = None
            logger.debug(
                f"Released frame sequence for camera '{camera_id}' "
                f"({frame_count} frames freed)",
                extra={"camera_id": camera_id, "component": "frame_sampler"},
            )

    def _acquire_frame_with_retry(
        self,
        camera_id: str,
        frame_idx: int,
        start_time: float,
        timeout_seconds: float,
    ) -> Optional[tuple]:
        """
        Acquire a single frame with retry logic.

        Retries up to config.max_retries times on failure. Checks timeout
        between retries.

        Args:
            camera_id: Camera identifier.
            frame_idx: Index of the frame being acquired (0-based).
            start_time: perf_counter value when sampling started.
            timeout_seconds: Maximum allowed elapsed time.

        Returns:
            Tuple of (frame_ndarray, timestamp_float) or None if all retries
            exhausted or timeout exceeded.
        """
        retries_used = 0

        for attempt in range(1 + self._config.max_retries):
            # Check timeout before each attempt
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout_seconds:
                return None

            with PerformanceTimer(
                "single_frame_capture",
                camera_id=camera_id,
                logger=logger,
            ) as frame_timer:
                result = self._acquire_single_frame(camera_id)

            if result is not None:
                # Record timing for this frame
                self._frame_timings.append(round(frame_timer.elapsed_ms, 3))
                self._retry_counts[frame_idx] = retries_used
                return result

            # Frame acquisition failed - count retry
            if attempt < self._config.max_retries:
                retries_used += 1
                self._total_retries += 1
                logger.debug(
                    f"Frame {frame_idx} acquisition failed for camera "
                    f"'{camera_id}', retry {retries_used}/{self._config.max_retries}",
                    extra={
                        "camera_id": camera_id,
                        "component": "frame_sampler",
                    },
                )

        # All retries exhausted
        self._retry_counts[frame_idx] = retries_used
        return None

    def _acquire_single_frame(self, camera_id: str) -> Optional[tuple]:
        """
        Acquire a single frame from the FrameAcquisitionModule.

        This adapter method integrates with the existing FrameAcquisitionModule's
        acquire_frame interface. It converts the camera_id string to the integer
        index expected by the acquisition module.

        Args:
            camera_id: Camera identifier string.

        Returns:
            Tuple of (frame as np.ndarray, timestamp as float in seconds)
            or None if acquisition failed.
        """
        # Map camera_id to an integer index for FrameAcquisitionModule
        # The module uses integer camera IDs (0-3)
        camera_index = self._resolve_camera_index(camera_id)
        if camera_index is None:
            return None

        result = self._acquisition.acquire_frame(camera_index)
        if result is None:
            return None

        frame, timestamp_us = result
        # Convert microsecond timestamp to seconds for FrameSequence
        timestamp_seconds = timestamp_us / 1_000_000.0
        return (frame, timestamp_seconds)

    def _resolve_camera_index(self, camera_id: str) -> Optional[int]:
        """
        Resolve a camera_id string to an integer index.

        Supports both numeric strings ("0", "1", "2", "3") and prefixed
        identifiers ("cam_0", "cam_01", "camera_2").

        Args:
            camera_id: Camera identifier string.

        Returns:
            Integer camera index (0-3), or None if unresolvable.
        """
        # Try direct integer parse
        try:
            index = int(camera_id)
            if 0 <= index <= 3:
                return index
        except ValueError:
            pass

        # Try extracting trailing digits from prefixed identifiers
        # e.g., "cam_0", "cam_01", "camera_2"
        digits = "".join(c for c in camera_id if c.isdigit())
        if digits:
            try:
                index = int(digits)
                if 0 <= index <= 3:
                    return index
            except ValueError:
                pass

        # Check if we have cameras registered and try matching
        for cam_id in self._acquisition.cameras:
            if str(cam_id) in camera_id or camera_id in str(cam_id):
                return cam_id

        logger.error(
            f"Cannot resolve camera_id '{camera_id}' to a valid camera index (0-3)",
            extra={"camera_id": camera_id, "component": "frame_sampler"},
        )
        return None

    @staticmethod
    def _estimate_memory_usage(frames: list) -> int:
        """
        Estimate memory usage of the frame list in bytes.

        Args:
            frames: List of numpy ndarrays.

        Returns:
            Total estimated memory in bytes.
        """
        total = 0
        for frame in frames:
            if isinstance(frame, np.ndarray):
                total += frame.nbytes
            else:
                total += sys.getsizeof(frame)
        return total
