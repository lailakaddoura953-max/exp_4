"""
Unit tests for HazardStore (src/dashboard/hazard_store.py).

Covers:
  - __init__: deque is created with the correct capacity
  - append: events are stored; oldest is evicted when at capacity
  - get_recent: returns newest-first; respects n; handles fewer-than-n events
  - count: reflects the current number of stored events

Requirements: 10.4, 10.5, 10.9
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from hazard_detection.models import BBox
from dashboard.hazard_store import HazardStore
from dashboard.models import HazardEvent, LocationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(hazard_type: str = "misaligned_container", camera_id: str = "cam_stub_01") -> HazardEvent:
    """Create a minimal HazardEvent for testing purposes."""
    return HazardEvent(
        event_id=str(uuid.uuid4()),
        hazard_type=hazard_type,
        camera_id=camera_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.85,
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2),
        annotated_image=None,
        location=LocationContext.from_camera_id(camera_id),
    )


# ---------------------------------------------------------------------------
# HazardStore.__init__
# ---------------------------------------------------------------------------

class TestHazardStoreInit:
    """Tests for HazardStore initialisation."""

    def test_default_capacity_is_20(self):
        """Default capacity should be 20 (Requirement 10.9)."""
        store = HazardStore()
        assert store._capacity == 20

    def test_custom_capacity_is_stored(self):
        """Custom capacity should be respected."""
        store = HazardStore(capacity=5)
        assert store._capacity == 5

    def test_store_starts_empty(self):
        """A new store must have zero events."""
        store = HazardStore()
        assert store.count() == 0

    def test_get_recent_on_empty_store_returns_empty_list(self):
        """get_recent on an empty store must return []."""
        store = HazardStore()
        assert store.get_recent() == []

    def test_capacity_one_is_valid(self):
        """Edge case: capacity=1 should work without error."""
        store = HazardStore(capacity=1)
        assert store.count() == 0


# ---------------------------------------------------------------------------
# HazardStore.append and HazardStore.count
# ---------------------------------------------------------------------------

class TestHazardStoreAppend:
    """Tests for append() and count()."""

    def test_count_increments_on_append(self):
        """count() should reflect the number of appended events."""
        store = HazardStore()
        assert store.count() == 0
        store.append(_make_event())
        assert store.count() == 1
        store.append(_make_event())
        assert store.count() == 2

    def test_count_does_not_exceed_capacity(self):
        """count() must never exceed the configured capacity (Requirement 10.9)."""
        capacity = 5
        store = HazardStore(capacity=capacity)
        for _ in range(capacity + 10):
            store.append(_make_event())
        assert store.count() == capacity

    def test_oldest_event_is_evicted_when_full(self):
        """When at capacity, the oldest event should be discarded (Requirement 10.9)."""
        store = HazardStore(capacity=3)
        e1 = _make_event(hazard_type="event_1")
        e2 = _make_event(hazard_type="event_2")
        e3 = _make_event(hazard_type="event_3")
        e4 = _make_event(hazard_type="event_4")

        store.append(e1)
        store.append(e2)
        store.append(e3)
        store.append(e4)  # should evict e1

        recent = store.get_recent(n=3)
        hazard_types = [e.hazard_type for e in recent]
        assert "event_1" not in hazard_types
        assert "event_4" in hazard_types

    def test_capacity_one_retains_only_latest(self):
        """With capacity=1, only the most recently appended event is retained."""
        store = HazardStore(capacity=1)
        e1 = _make_event(hazard_type="first")
        e2 = _make_event(hazard_type="second")
        store.append(e1)
        store.append(e2)
        assert store.count() == 1
        assert store.get_recent(n=1)[0].hazard_type == "second"

    def test_append_stores_event_reference(self):
        """The exact event object should be retrievable after appending."""
        store = HazardStore(capacity=5)
        event = _make_event(hazard_type="ppe_violation")
        store.append(event)
        result = store.get_recent(n=1)
        assert result[0] is event


# ---------------------------------------------------------------------------
# HazardStore.get_recent
# ---------------------------------------------------------------------------

class TestHazardStoreGetRecent:
    """Tests for get_recent(n)."""

    def test_default_n_is_3(self):
        """get_recent() with no argument returns at most 3 events (Requirement 10.5)."""
        store = HazardStore()
        for _ in range(10):
            store.append(_make_event())
        assert len(store.get_recent()) == 3

    def test_get_recent_returns_newest_first(self):
        """Events must be returned in reverse insertion order (newest first)."""
        store = HazardStore()
        events = [_make_event(hazard_type=f"event_{i}") for i in range(5)]
        for e in events:
            store.append(e)

        # Newest is events[4], oldest events[0]
        recent = store.get_recent(n=5)
        expected_order = [f"event_{i}" for i in range(4, -1, -1)]
        assert [e.hazard_type for e in recent] == expected_order

    def test_get_recent_fewer_than_n_events_returns_all(self):
        """When fewer than n events exist, all available events are returned (no padding)."""
        store = HazardStore()
        store.append(_make_event(hazard_type="only_one"))
        result = store.get_recent(n=5)
        assert len(result) == 1
        assert result[0].hazard_type == "only_one"

    def test_get_recent_does_not_pad_with_none(self):
        """The returned list must not contain None values."""
        store = HazardStore()
        store.append(_make_event())
        result = store.get_recent(n=10)
        assert None not in result

    def test_get_recent_n_zero_returns_empty_list(self):
        """get_recent(n=0) should return an empty list."""
        store = HazardStore()
        store.append(_make_event())
        assert store.get_recent(n=0) == []

    def test_get_recent_n_larger_than_count(self):
        """Requesting more events than stored returns only what is available."""
        store = HazardStore(capacity=10)
        store.append(_make_event(hazard_type="a"))
        store.append(_make_event(hazard_type="b"))
        result = store.get_recent(n=100)
        assert len(result) == 2

    def test_get_recent_after_eviction_order_is_correct(self):
        """Newest-first order must be preserved after eviction of old events."""
        store = HazardStore(capacity=3)
        for i in range(5):
            store.append(_make_event(hazard_type=f"event_{i}"))

        # Store should contain events 2, 3, 4 (0 and 1 evicted)
        recent = store.get_recent(n=3)
        assert recent[0].hazard_type == "event_4"
        assert recent[1].hazard_type == "event_3"
        assert recent[2].hazard_type == "event_2"

    def test_get_recent_returns_list_type(self):
        """get_recent must return a list (not a deque or other iterable)."""
        store = HazardStore()
        store.append(_make_event())
        result = store.get_recent()
        assert isinstance(result, list)

    def test_single_event_returned_as_first_element(self):
        """With one event, get_recent(n=3) returns that event at index 0."""
        store = HazardStore()
        event = _make_event(hazard_type="sole_event")
        store.append(event)
        result = store.get_recent(n=3)
        assert len(result) == 1
        assert result[0].hazard_type == "sole_event"

    def test_get_recent_does_not_modify_store(self):
        """Calling get_recent must not remove events from the store."""
        store = HazardStore()
        for _ in range(5):
            store.append(_make_event())
        count_before = store.count()
        store.get_recent(n=3)
        assert store.count() == count_before
