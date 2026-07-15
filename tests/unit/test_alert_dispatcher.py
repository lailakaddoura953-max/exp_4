"""
Unit tests for the Alert Dispatcher module.

Tests cover:
- Binary dispatch: confirmed hazards (is_hazard=True) dispatched, non-hazards logged only
- Property 16: Alert rate limiting — same camera_id + hazard_type within window suppressed
- Property 17: Channel failure resilience — partial failure continues, total failure retains
- Rate limiting with various time windows
- All-channel-failure retry behavior

Validates: Requirements 9.1, 9.2, 9.3, 9.6
"""

import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from hazard_detection.alert_dispatcher import AlertChannelAdapter, AlertDispatcher
from hazard_detection.models import AlertDispatcherConfig


# =============================================================================
# Mock Channel Adapters
# =============================================================================


class MockChannel:
    """Mock channel adapter that records calls and returns configurable results."""

    def __init__(self, name: str = "mock_channel", should_succeed: bool = True):
        self._name = name
        self._should_succeed = should_succeed
        self.sent_payloads: List[Dict[str, Any]] = []
        self.send_count: int = 0

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        self.send_count += 1
        self.sent_payloads.append(alert_payload)
        return self._should_succeed

    def get_name(self) -> str:
        return self._name


class RaisingChannel:
    """Mock channel adapter that raises an exception on send."""

    def __init__(self, name: str = "raising_channel"):
        self._name = name
        self.send_count: int = 0

    def send(self, alert_payload: Dict[str, Any]) -> bool:
        self.send_count += 1
        raise ConnectionError(f"Channel '{self._name}' connection failed")

    def get_name(self) -> str:
        return self._name


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> AlertDispatcherConfig:
    """Default alert dispatcher config with 60s rate limit window."""
    return AlertDispatcherConfig(rate_limit_seconds=60, channels=["email", "dashboard"])


@pytest.fixture
def short_window_config() -> AlertDispatcherConfig:
    """Alert dispatcher config with minimum 10s rate limit window."""
    return AlertDispatcherConfig(rate_limit_seconds=10, channels=["email"])


@pytest.fixture
def long_window_config() -> AlertDispatcherConfig:
    """Alert dispatcher config with maximum 300s rate limit window."""
    return AlertDispatcherConfig(rate_limit_seconds=300, channels=["sms", "email", "dashboard"])


@pytest.fixture
def success_channel() -> MockChannel:
    """A channel that always succeeds."""
    return MockChannel(name="email", should_succeed=True)


@pytest.fixture
def fail_channel() -> MockChannel:
    """A channel that always fails (returns False)."""
    return MockChannel(name="dashboard", should_succeed=False)


@pytest.fixture
def raising_channel() -> RaisingChannel:
    """A channel that raises exceptions."""
    return RaisingChannel(name="sms")


# =============================================================================
# Binary Dispatch: is_hazard=True → dispatch, is_hazard=False → log only
# =============================================================================


class TestBinaryDispatch:
    """
    Test the core binary dispatch logic:
    - Confirmed hazards (is_hazard=True) are dispatched through channels
    - Non-hazards (is_hazard=False) are logged only, no dispatch

    **Validates: Requirements 9.1, 9.2**
    """

    def test_confirmed_hazard_dispatched_to_channels(
        self, hazard_event_factory, default_config
    ):
        """is_hazard=True events are sent through all configured channels."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event = hazard_event_factory(is_hazard=True, hazard_type="zone_violation")
        result = dispatcher.dispatch(event)

        assert result is True
        assert channel.send_count == 1
        assert channel.sent_payloads[0]["hazard_type"] == "zone_violation"
        assert channel.sent_payloads[0]["camera_id"] == event.camera_id
        assert channel.sent_payloads[0]["timestamp"] == event.timestamp
        assert channel.sent_payloads[0]["confidence"] == event.confidence

    def test_non_hazard_logged_only_no_dispatch(
        self, hazard_event_factory, default_config
    ):
        """is_hazard=False events are logged but NOT sent to any channel."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event = hazard_event_factory(is_hazard=False, hazard_type="zone_violation")
        result = dispatcher.dispatch(event)

        assert result is True
        assert channel.send_count == 0
        assert len(channel.sent_payloads) == 0

    def test_multiple_channels_all_receive_hazard(
        self, hazard_event_factory, default_config
    ):
        """All configured channels receive the alert for confirmed hazards."""
        ch1 = MockChannel(name="email")
        ch2 = MockChannel(name="dashboard")
        dispatcher = AlertDispatcher(channels=[ch1, ch2], config=default_config)

        event = hazard_event_factory(is_hazard=True)
        dispatcher.dispatch(event)

        assert ch1.send_count == 1
        assert ch2.send_count == 1

    def test_non_hazard_does_not_update_statistics(
        self, hazard_event_factory, default_config
    ):
        """Non-hazard events increment logged_only counter, not dispatched."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event = hazard_event_factory(is_hazard=False)
        dispatcher.dispatch(event)

        stats = dispatcher.get_statistics()
        assert stats["alerts_logged_only"] == 1
        assert stats["alerts_dispatched"] == 0

    def test_hazard_event_payload_includes_required_fields(
        self, hazard_event_factory, default_config
    ):
        """Alert payload includes hazard_type, camera_id, timestamp, confidence."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event = hazard_event_factory(
            is_hazard=True,
            hazard_type="container_misalignment",
            camera_id="cam_03",
            confidence=0.92,
        )
        dispatcher.dispatch(event)

        payload = channel.sent_payloads[0]
        assert payload["hazard_type"] == "container_misalignment"
        assert payload["camera_id"] == "cam_03"
        assert payload["confidence"] == 0.92
        assert "timestamp" in payload
        assert "event_id" in payload


# =============================================================================
# Property 16: Alert rate limiting
# =============================================================================


class TestAlertRateLimiting:
    """
    Property 16: For any two Hazard_Events with the same camera_id and hazard_type
    emitted within the configured rate_limit_window (10–300 seconds), the
    Alert_Dispatcher SHALL suppress the second alert.

    **Validates: Requirements 9.3**
    """

    def test_same_camera_and_type_within_window_suppressed(
        self, hazard_event_factory, default_config
    ):
        """Second alert for same camera_id + hazard_type within window is suppressed."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event1 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        event2 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )

        result1 = dispatcher.dispatch(event1)
        result2 = dispatcher.dispatch(event2)

        assert result1 is True
        assert result2 is True  # Suppression is not a failure
        assert channel.send_count == 1  # Only first dispatched

    def test_different_camera_same_type_not_suppressed(
        self, hazard_event_factory, default_config
    ):
        """Different camera_id with same hazard_type is NOT rate-limited."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event1 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        event2 = hazard_event_factory(
            is_hazard=True, camera_id="cam_02", hazard_type="zone_violation"
        )

        dispatcher.dispatch(event1)
        dispatcher.dispatch(event2)

        assert channel.send_count == 2

    def test_same_camera_different_type_not_suppressed(
        self, hazard_event_factory, default_config
    ):
        """Same camera_id with different hazard_type is NOT rate-limited."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event1 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        event2 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="container_misalignment"
        )

        dispatcher.dispatch(event1)
        dispatcher.dispatch(event2)

        assert channel.send_count == 2

    def test_rate_limit_expires_after_window(
        self, hazard_event_factory, short_window_config
    ):
        """After the rate limit window expires, the same key can dispatch again."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=short_window_config)

        event1 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        dispatcher.dispatch(event1)
        assert channel.send_count == 1

        # Simulate time passing beyond the window by manipulating the state
        key = ("cam_01", "zone_violation")
        dispatcher._rate_limit_state[key] = time.time() - 11  # 11s ago, window is 10s

        event2 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        dispatcher.dispatch(event2)
        assert channel.send_count == 2

    def test_rate_limit_with_minimum_window_10s(self, hazard_event_factory):
        """Rate limiting works at the minimum boundary (10s)."""
        config = AlertDispatcherConfig(rate_limit_seconds=10, channels=["email"])
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=config)

        event1 = hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")
        event2 = hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")

        dispatcher.dispatch(event1)
        dispatcher.dispatch(event2)

        assert channel.send_count == 1
        assert dispatcher.alerts_rate_limited == 1

    def test_rate_limit_with_maximum_window_300s(self, hazard_event_factory):
        """Rate limiting works at the maximum boundary (300s)."""
        config = AlertDispatcherConfig(rate_limit_seconds=300, channels=["email"])
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=config)

        event1 = hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")
        event2 = hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")

        dispatcher.dispatch(event1)
        dispatcher.dispatch(event2)

        assert channel.send_count == 1
        assert dispatcher.alerts_rate_limited == 1

    def test_rate_limit_counter_increments(
        self, hazard_event_factory, default_config
    ):
        """Rate-limited events increment the rate_limited counter."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        for _ in range(5):
            event = hazard_event_factory(
                is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
            )
            dispatcher.dispatch(event)

        assert dispatcher.alerts_dispatched == 1
        assert dispatcher.alerts_rate_limited == 4

    def test_non_hazard_does_not_trigger_rate_limit(
        self, hazard_event_factory, default_config
    ):
        """Non-hazard events do not affect rate limit state."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        # Dispatch a non-hazard first
        non_hazard = hazard_event_factory(
            is_hazard=False, camera_id="cam_01", hazard_type="zone_violation"
        )
        dispatcher.dispatch(non_hazard)

        # Then dispatch a real hazard — should NOT be rate-limited
        hazard = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        result = dispatcher.dispatch(hazard)

        assert result is True
        assert channel.send_count == 1
        assert dispatcher.alerts_rate_limited == 0

    def test_clear_rate_limit_specific_key(
        self, hazard_event_factory, default_config
    ):
        """Clearing rate limit for a specific camera_id + hazard_type allows re-dispatch."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        event = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        dispatcher.dispatch(event)
        assert channel.send_count == 1

        # Clear rate limit for that specific key
        dispatcher.clear_rate_limit(camera_id="cam_01", hazard_type="zone_violation")

        event2 = hazard_event_factory(
            is_hazard=True, camera_id="cam_01", hazard_type="zone_violation"
        )
        dispatcher.dispatch(event2)
        assert channel.send_count == 2

    def test_clear_rate_limit_all(self, hazard_event_factory, default_config):
        """Clearing all rate limits allows all keys to re-dispatch."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        # Dispatch multiple different keys
        for cam in ["cam_01", "cam_02"]:
            event = hazard_event_factory(is_hazard=True, camera_id=cam, hazard_type="zone_violation")
            dispatcher.dispatch(event)
        assert channel.send_count == 2

        # Clear all
        dispatcher.clear_rate_limit()

        # Dispatch same keys again — should succeed
        for cam in ["cam_01", "cam_02"]:
            event = hazard_event_factory(is_hazard=True, camera_id=cam, hazard_type="zone_violation")
            dispatcher.dispatch(event)
        assert channel.send_count == 4


# =============================================================================
# Property 17: Channel failure resilience
# =============================================================================


class TestChannelFailureResilience:
    """
    Property 17: For any alert dispatch where one or more channels fail, the
    Alert_Dispatcher SHALL continue sending to remaining channels. The alert
    SHALL be reported as sent if at least one channel succeeds.

    **Validates: Requirements 9.6**
    """

    def test_partial_failure_continues_to_remaining_channels(
        self, hazard_event_factory, default_config
    ):
        """When one channel fails, dispatch continues to remaining channels."""
        success_ch = MockChannel(name="email", should_succeed=True)
        fail_ch = MockChannel(name="dashboard", should_succeed=False)
        dispatcher = AlertDispatcher(
            channels=[fail_ch, success_ch], config=default_config
        )

        event = hazard_event_factory(is_hazard=True)
        result = dispatcher.dispatch(event)

        assert result is True  # At least one succeeded
        assert fail_ch.send_count == 1  # Attempted
        assert success_ch.send_count == 1  # Attempted and succeeded

    def test_partial_failure_reports_as_sent(
        self, hazard_event_factory, default_config
    ):
        """Alert is reported as sent when at least one channel succeeds."""
        success_ch = MockChannel(name="email", should_succeed=True)
        fail_ch1 = MockChannel(name="dashboard", should_succeed=False)
        fail_ch2 = MockChannel(name="sms", should_succeed=False)
        dispatcher = AlertDispatcher(
            channels=[fail_ch1, fail_ch2, success_ch], config=default_config
        )

        event = hazard_event_factory(is_hazard=True)
        result = dispatcher.dispatch(event)

        assert result is True
        assert dispatcher.alerts_dispatched == 1
        assert dispatcher.alerts_total_failure == 0

    def test_exception_in_channel_treated_as_failure(
        self, hazard_event_factory, default_config
    ):
        """Channel that raises exception is treated as failed, others continue."""
        raising_ch = RaisingChannel(name="sms")
        success_ch = MockChannel(name="email", should_succeed=True)
        dispatcher = AlertDispatcher(
            channels=[raising_ch, success_ch], config=default_config
        )

        event = hazard_event_factory(is_hazard=True)
        result = dispatcher.dispatch(event)

        assert result is True
        assert raising_ch.send_count == 1
        assert success_ch.send_count == 1

    def test_total_failure_returns_false(
        self, hazard_event_factory, default_config
    ):
        """When all channels fail, dispatch returns False."""
        fail_ch1 = MockChannel(name="email", should_succeed=False)
        fail_ch2 = MockChannel(name="dashboard", should_succeed=False)
        dispatcher = AlertDispatcher(
            channels=[fail_ch1, fail_ch2], config=default_config
        )

        event = hazard_event_factory(is_hazard=True)
        result = dispatcher.dispatch(event)

        assert result is False

    def test_total_failure_retains_event_for_retry(
        self, hazard_event_factory, default_config
    ):
        """On total failure, the event is added to the retry queue."""
        fail_ch = MockChannel(name="email", should_succeed=False)
        dispatcher = AlertDispatcher(channels=[fail_ch], config=default_config)

        event = hazard_event_factory(is_hazard=True)
        dispatcher.dispatch(event)

        assert len(dispatcher.retry_queue) == 1
        assert dispatcher.retry_queue[0].event_id == event.event_id

    def test_total_failure_increments_failure_counter(
        self, hazard_event_factory, default_config
    ):
        """Total failure increments the total_failure counter."""
        fail_ch = MockChannel(name="email", should_succeed=False)
        dispatcher = AlertDispatcher(channels=[fail_ch], config=default_config)

        event = hazard_event_factory(is_hazard=True)
        dispatcher.dispatch(event)

        stats = dispatcher.get_statistics()
        assert stats["alerts_total_failure"] == 1
        assert stats["alerts_dispatched"] == 0

    def test_all_channels_exception_retains_for_retry(
        self, hazard_event_factory, default_config
    ):
        """When all channels raise exceptions, event is retained for retry."""
        raising_ch1 = RaisingChannel(name="email")
        raising_ch2 = RaisingChannel(name="sms")
        dispatcher = AlertDispatcher(
            channels=[raising_ch1, raising_ch2], config=default_config
        )

        event = hazard_event_factory(is_hazard=True)
        result = dispatcher.dispatch(event)

        assert result is False
        assert len(dispatcher.retry_queue) == 1

    def test_retry_succeeds_after_channel_recovery(
        self, hazard_event_factory, default_config
    ):
        """Retry successfully dispatches after channel recovers."""
        fail_ch = MockChannel(name="email", should_succeed=False)
        dispatcher = AlertDispatcher(channels=[fail_ch], config=default_config)

        event = hazard_event_factory(is_hazard=True)
        dispatcher.dispatch(event)
        assert len(dispatcher.retry_queue) == 1

        # "Recover" the channel
        fail_ch._should_succeed = True
        # Clear rate limit since the first dispatch didn't update it (total failure)
        retried = dispatcher.retry_failed()

        assert retried == 1
        assert len(dispatcher.retry_queue) == 0

    def test_retry_queue_empty_returns_zero(self, default_config):
        """retry_failed() returns 0 when no events are queued."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        retried = dispatcher.retry_failed()
        assert retried == 0

    def test_multiple_total_failures_accumulate_in_retry_queue(
        self, hazard_event_factory, default_config
    ):
        """Multiple total failures accumulate events in the retry queue."""
        fail_ch = MockChannel(name="email", should_succeed=False)
        dispatcher = AlertDispatcher(channels=[fail_ch], config=default_config)

        # Dispatch 3 different events (different keys to avoid rate limiting)
        for i in range(3):
            event = hazard_event_factory(
                is_hazard=True,
                camera_id=f"cam_{i:02d}",
                hazard_type="zone_violation",
            )
            dispatcher.dispatch(event)

        assert len(dispatcher.retry_queue) == 3
        assert dispatcher.alerts_total_failure == 3


# =============================================================================
# Statistics and introspection
# =============================================================================


class TestDispatcherStatistics:
    """Test dispatch statistics tracking."""

    def test_initial_statistics_all_zero(self, default_config):
        """Fresh dispatcher has all-zero statistics."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        stats = dispatcher.get_statistics()
        assert stats["alerts_dispatched"] == 0
        assert stats["alerts_rate_limited"] == 0
        assert stats["alerts_logged_only"] == 0
        assert stats["alerts_total_failure"] == 0
        assert stats["retry_queue_size"] == 0
        assert stats["rate_limit_entries"] == 0

    def test_mixed_dispatch_scenario_statistics(
        self, hazard_event_factory, default_config
    ):
        """Statistics reflect a mix of dispatched, logged, and rate-limited events."""
        channel = MockChannel(name="email")
        dispatcher = AlertDispatcher(channels=[channel], config=default_config)

        # 1 dispatched hazard
        dispatcher.dispatch(
            hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")
        )
        # 1 non-hazard logged
        dispatcher.dispatch(
            hazard_event_factory(is_hazard=False, camera_id="cam_01", hazard_type="zone_violation")
        )
        # 1 rate-limited (same key)
        dispatcher.dispatch(
            hazard_event_factory(is_hazard=True, camera_id="cam_01", hazard_type="zone_violation")
        )

        stats = dispatcher.get_statistics()
        assert stats["alerts_dispatched"] == 1
        assert stats["alerts_logged_only"] == 1
        assert stats["alerts_rate_limited"] == 1
