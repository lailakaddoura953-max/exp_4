"""
HazardStore — in-memory ring-buffer of recent hazard events.

Backed by ``collections.deque(maxlen=capacity)`` so the oldest event is
automatically discarded when the store reaches capacity.

Threading note
--------------
This implementation is intentionally NOT thread-safe.  It is designed for use
with Flask's built-in single-threaded development server where all requests are
handled sequentially.

For production deployment (multi-threaded WSGI server such as Gunicorn or
uWSGI, or any async framework), replace the plain ``deque`` access with
``threading.Lock``-guarded reads/writes, or swap the backing store for a
thread-safe queue implementation (e.g. ``queue.Queue``).

Requirements: 10.4, 10.5, 10.9
"""

from __future__ import annotations

import collections
from typing import List

from dashboard.models import HazardEvent


class HazardStore:
    """
    In-memory store for the most recent N hazard events.

    Events are appended chronologically.  When the store is at capacity, the
    oldest event is automatically discarded by the underlying deque.

    Parameters
    ----------
    capacity : int
        Maximum number of events to retain.  Defaults to 20 (Requirement 10.9).

    Requirements: 10.4, 10.5, 10.9
    """

    def __init__(self, capacity: int = 20) -> None:
        # deque with maxlen handles oldest-first eviction automatically.
        # Threading note: single-threaded Flask dev server only.
        # TODO (production): wrap _store access with threading.Lock when
        #   deploying under a multi-threaded WSGI server (Gunicorn, uWSGI).
        self._store: collections.deque[HazardEvent] = collections.deque(
            maxlen=capacity
        )
        self._capacity = capacity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, event: HazardEvent) -> None:
        """
        Append a ``HazardEvent`` to the store.

        When the store is already at capacity, the oldest event is discarded
        automatically by the deque before the new event is added.

        Parameters
        ----------
        event : HazardEvent
            The event to store.

        Requirement: 10.4, 10.9
        """
        self._store.append(event)

    def get_recent(self, n: int = 3) -> List[HazardEvent]:
        """
        Return the most recent ``n`` events, newest first.

        If the store contains fewer than ``n`` events, all available events are
        returned (the list is NOT padded with ``None``).

        Parameters
        ----------
        n : int
            Number of recent events to retrieve.  Defaults to 3.

        Returns
        -------
        List[HazardEvent]
            Events in reverse insertion order (newest first).

        Requirement: 10.5
        """
        # Convert to list, reverse to get newest-first, then slice to n.
        return list(reversed(list(self._store)))[:n]

    def count(self) -> int:
        """
        Return the current number of stored events.

        Returns
        -------
        int
            Number of events currently in the store (0 ≤ count ≤ capacity).
        """
        return len(self._store)
