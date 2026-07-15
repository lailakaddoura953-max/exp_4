"""
Alert Dispatcher Module

Wraps the existing AlertSystem for hazard-specific dispatch with binary
classification: confirmed hazards (is_hazard=True) are dispatched through
all configured channels; non-hazards (is_hazard=False) are logged only.

Properties validated:
- Property 15: Binary dispatch (is_hazard=True → dispatch, is_hazard=False → log only)
- Property 16: Alert rate limiting (same camera_id + hazard_type within window suppressed)
- Property 17: Channel failure resilience (partial failure continues, total failure retains for retry)

Requirements covered:
- 9.1: Dispatch through all configured channels within 5 seconds of event emission
- 9.2: Rate limiting by camera_id + hazard_type within configurable window (10-300s, default 60s)
- 9.3: Reuse existing AlertSystem infrastructure
- 9.4: Include hazard_type, camera_id, timestamp, confidence in each alert
- 9.5: On partial failure, continue with remaining channels; report as sent if at least one succeeds
- 9.6: On total failure, retain event for retry, log error
"""

import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

from hazard_detection.diagnostics import PerformanceTimer, get_logger
from hazard_detection.models import AlertDispatcherConfig, HazardEvent


logger = get_logger("alert_dispatcher")


# ---------------------------------------------------------------------------
# Channel Protocol — adapter interface for alert channels
# ---------------------------------------------------------------------------


class AlertChannelAdapter(Protocol):
    """Protocol that alert channel adapters must satisfy."""

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        """
        Send an alert payload through this channel.

        Args:
            alert_payload: Dictionary containing hazard_type, camera_id,
                          timestamp, confidence, and event metadata.

        Returns:
            True if the alert was delivered successfully, False otherwise.
        """
        ...

    def get_name(self) -> str:
        """Return the channel name for logging and identification."""
        ...


# ---------------------------------------------------------------------------
# Alert Dispatcher
# ---------------------------------------------------------------------------


class AlertDispatcher:
    """
    Routes HazardEvent instances through configured alert channels.

    Binary dispatch model:
    - is_hazard=True → dispatch through all configured channels within 5 seconds
    - is_hazard=False → log only, no channel dispatch

    Wraps the existing AlertSystem infrastructure and adds:
    - Hazard-type + camera_id composite rate limiting
    - Structured logging of dispatch decisions
    - Retry queue for total delivery failures
    """

    def __init__(
        self,
        channels: List[AlertChannelAdapter],
        config: AlertDispatcherConfig,
    ):
        """
        Initialize the Alert Dispatcher.

        Args:
            channels: List of alert channel adapters to dispatch through.
            config: AlertDispatcherConfig with rate_limit_seconds (10-300, default 60)
                   and channel names.
        """
        self.channels = channels
        self.config = config

        # Rate limiting state: (camera_id, hazard_type) → last dispatch timestamp
        self._rate_limit_state: Dict[Tuple[str, str], float] = {}

        # Retry queue for events that failed total delivery
        self._retry_queue: List[HazardEvent] = []

        # Statistics
        self.alerts_dispatched: int = 0
        self.alerts_rate_limited: int = 0
        self.alerts_logged_only: int = 0
        self.alerts_total_failure: int = 0

        logger.info(
            "AlertDispatcher initialized",
            extra={
                "extra_data": {
                    "rate_limit_seconds": config.rate_limit_seconds,
                    "channel_count": len(channels),
                    "channel_names": [ch.get_name() for ch in channels],
                }
            },
        )

    def dispatch(self, event: HazardEvent) -> bool:
        """
        Route a hazard event through configured channels.

        Binary dispatch logic:
        - is_hazard=True: attempt dispatch through all channels within 5s
        - is_hazard=False: log the event only, no channel dispatch

        Rate limiting suppresses duplicate alerts for the same camera_id +
        hazard_type within the configured window.

        On partial channel failure, continues with remaining channels.
        On total failure, retains event in retry queue and logs error.

        Args:
            event: HazardEvent to dispatch.

        Returns:
            True if the alert was dispatched successfully (at least one channel
            succeeded), or if the event was correctly logged as non-hazard.
            False if all channels failed for a confirmed hazard.
        """
        with PerformanceTimer("alert_dispatch", camera_id=event.camera_id):
            # Non-hazard events: log only, no dispatch
            if not event.is_hazard:
                self._log_non_hazard(event)
                self.alerts_logged_only += 1
                return True

            # Check rate limiting
            if self._is_rate_limited(event.camera_id, event.hazard_type):
                self._log_rate_limited(event)
                self.alerts_rate_limited += 1
                return True  # Suppressed is not a failure

            # Dispatch to all channels
            return self._dispatch_to_channels(event)

    def _dispatch_to_channels(self, event: HazardEvent) -> bool:
        """
        Send the alert payload to all configured channels.

        Continues on partial failure; retains for retry on total failure.

        Returns:
            True if at least one channel succeeded, False if all failed.
        """
        payload = self._build_payload(event)

        channels_succeeded: List[str] = []
        channels_failed: List[str] = []

        for channel in self.channels:
            channel_name = channel.get_name()
            try:
                success = channel.send(payload)
                if success:
                    channels_succeeded.append(channel_name)
                else:
                    channels_failed.append(channel_name)
            except Exception as exc:
                channels_failed.append(channel_name)
                logger.warning(
                    f"Channel '{channel_name}' raised exception during dispatch",
                    extra={
                        "extra_data": {
                            "channel": channel_name,
                            "event_id": event.event_id,
                            "error": str(exc),
                        }
                    },
                )

        # Log dispatch result
        self._log_dispatch_result(event, channels_succeeded, channels_failed)

        if channels_succeeded:
            # At least one channel succeeded — update rate limit state
            self._update_rate_limit(event.camera_id, event.hazard_type)
            self.alerts_dispatched += 1
            return True
        else:
            # Total failure — retain for retry
            self._retry_queue.append(event)
            self.alerts_total_failure += 1
            logger.error(
                f"Total delivery failure for event {event.event_id}; "
                f"retained for retry",
                extra={
                    "extra_data": {
                        "event_id": event.event_id,
                        "hazard_type": event.hazard_type,
                        "camera_id": event.camera_id,
                        "channels_attempted": [ch.get_name() for ch in self.channels],
                    }
                },
            )
            return False

    def retry_failed(self) -> int:
        """
        Retry dispatching events that previously failed total delivery.

        Returns:
            Number of events successfully dispatched on retry.
        """
        if not self._retry_queue:
            return 0

        events_to_retry = list(self._retry_queue)
        self._retry_queue.clear()
        retried = 0

        for event in events_to_retry:
            logger.info(
                f"Retrying dispatch for event {event.event_id}",
                extra={
                    "extra_data": {
                        "event_id": event.event_id,
                        "hazard_type": event.hazard_type,
                        "camera_id": event.camera_id,
                    }
                },
            )
            if self._dispatch_to_channels(event):
                retried += 1

        return retried

    # -----------------------------------------------------------------------
    # Rate Limiting
    # -----------------------------------------------------------------------

    def _is_rate_limited(self, camera_id: str, hazard_type: str) -> bool:
        """
        Check if a dispatch for this camera_id + hazard_type is rate-limited.

        Args:
            camera_id: Camera identifier.
            hazard_type: Hazard type string.

        Returns:
            True if the event should be suppressed.
        """
        key = (camera_id, hazard_type)
        if key not in self._rate_limit_state:
            return False

        elapsed = time.time() - self._rate_limit_state[key]
        return elapsed < self.config.rate_limit_seconds

    def _update_rate_limit(self, camera_id: str, hazard_type: str) -> None:
        """Record the current time as the last dispatch time for rate limiting."""
        self._rate_limit_state[(camera_id, hazard_type)] = time.time()

    def clear_rate_limit(
        self,
        camera_id: Optional[str] = None,
        hazard_type: Optional[str] = None,
    ) -> None:
        """
        Clear rate limit state.

        Args:
            camera_id: If provided with hazard_type, clears that specific key.
                      If only camera_id, clears all entries for that camera.
                      If None, clears all rate limit state.
            hazard_type: Used together with camera_id for specific key clearing.
        """
        if camera_id is None:
            self._rate_limit_state.clear()
        elif hazard_type is not None:
            self._rate_limit_state.pop((camera_id, hazard_type), None)
        else:
            keys_to_remove = [
                k for k in self._rate_limit_state if k[0] == camera_id
            ]
            for k in keys_to_remove:
                del self._rate_limit_state[k]

    # -----------------------------------------------------------------------
    # Payload Construction
    # -----------------------------------------------------------------------

    def _build_payload(self, event: HazardEvent) -> Dict[str, Any]:
        """
        Build the alert payload dictionary from a HazardEvent.

        Includes hazard_type, camera_id, timestamp, and confidence as
        required by Requirement 9.4.
        """
        return {
            "event_id": event.event_id,
            "hazard_type": event.hazard_type,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp,
            "confidence": event.confidence,
            "is_hazard": event.is_hazard,
            "bbox": {
                "x_center": event.bbox.x_center,
                "y_center": event.bbox.y_center,
                "width": event.bbox.width,
                "height": event.bbox.height,
            },
            "metadata": {
                "frame_index": event.metadata.frame_index,
                "detection_class": event.metadata.detection_class,
                "frames_detected": event.metadata.frames_detected,
                "flow_consistency_score": event.metadata.flow_consistency_score,
            },
        }

    # -----------------------------------------------------------------------
    # Structured Logging
    # -----------------------------------------------------------------------

    def _log_non_hazard(self, event: HazardEvent) -> None:
        """Log a non-hazard event (no dispatch)."""
        logger.info(
            f"Non-hazard event logged (no dispatch): "
            f"type={event.hazard_type} camera={event.camera_id} "
            f"confidence={event.confidence:.3f}",
            extra={
                "extra_data": {
                    "event_id": event.event_id,
                    "hazard_type": event.hazard_type,
                    "camera_id": event.camera_id,
                    "confidence": event.confidence,
                    "is_hazard": False,
                    "decision": "log_only",
                }
            },
        )

    def _log_rate_limited(self, event: HazardEvent) -> None:
        """Log a rate-limited (suppressed) event."""
        key = (event.camera_id, event.hazard_type)
        last_time = self._rate_limit_state.get(key, 0.0)
        elapsed = time.time() - last_time if last_time else 0.0
        remaining = max(0.0, self.config.rate_limit_seconds - elapsed)

        logger.info(
            f"Alert rate-limited (suppressed): "
            f"type={event.hazard_type} camera={event.camera_id} "
            f"window_remaining={remaining:.1f}s",
            extra={
                "extra_data": {
                    "event_id": event.event_id,
                    "hazard_type": event.hazard_type,
                    "camera_id": event.camera_id,
                    "confidence": event.confidence,
                    "decision": "rate_limited",
                    "rate_limit_key": f"{event.camera_id}:{event.hazard_type}",
                    "window_remaining_seconds": round(remaining, 1),
                    "rate_limit_window": self.config.rate_limit_seconds,
                }
            },
        )

    def _log_dispatch_result(
        self,
        event: HazardEvent,
        channels_succeeded: List[str],
        channels_failed: List[str],
    ) -> None:
        """Log the result of a dispatch attempt."""
        total = len(channels_succeeded) + len(channels_failed)
        status = (
            "success" if channels_succeeded and not channels_failed
            else "partial_success" if channels_succeeded
            else "total_failure"
        )

        logger.info(
            f"Alert dispatch {status}: "
            f"type={event.hazard_type} camera={event.camera_id} "
            f"channels={len(channels_succeeded)}/{total} succeeded",
            extra={
                "extra_data": {
                    "event_id": event.event_id,
                    "hazard_type": event.hazard_type,
                    "camera_id": event.camera_id,
                    "confidence": event.confidence,
                    "decision": "dispatched",
                    "status": status,
                    "channels_succeeded": channels_succeeded,
                    "channels_failed": channels_failed,
                }
            },
        )

    # -----------------------------------------------------------------------
    # Statistics & Introspection
    # -----------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return dispatch statistics."""
        return {
            "alerts_dispatched": self.alerts_dispatched,
            "alerts_rate_limited": self.alerts_rate_limited,
            "alerts_logged_only": self.alerts_logged_only,
            "alerts_total_failure": self.alerts_total_failure,
            "retry_queue_size": len(self._retry_queue),
            "rate_limit_entries": len(self._rate_limit_state),
        }

    @property
    def retry_queue(self) -> List[HazardEvent]:
        """Return a copy of the current retry queue."""
        return list(self._retry_queue)

    @property
    def rate_limit_state(self) -> Dict[Tuple[str, str], float]:
        """Return a copy of the current rate limit state."""
        return dict(self._rate_limit_state)
