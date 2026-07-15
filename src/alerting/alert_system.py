"""
Alert System Module

Multi-channel alert dispatching with rate limiting and bounded history.

Properties validated:
- Property 14: Alert Delivery for High Severity Events (HIGH and CRITICAL)
"""

import time
from typing import Dict, List, Optional, Protocol
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from src.models.core import MisalignmentEvent, Severity


class AlertChannel(Protocol):
    """Protocol for alert channels"""
    
    def send_alert(self, event: MisalignmentEvent) -> bool:
        """
        Send alert for misalignment event
        
        Args:
            event: MisalignmentEvent to alert on
        
        Returns:
            True if alert sent successfully
        """
        ...
    
    def get_channel_name(self) -> str:
        """Get channel name for identification"""
        ...


@dataclass
class AlertRecord:
    """Record of sent alert"""
    event_id: str
    camera_id: int
    severity: Severity
    timestamp: datetime
    channels_sent: List[str]
    success: bool


class MockAlertChannel:
    """Mock alert channel for testing"""
    
    def __init__(self, name: str, fail_on_send: bool = False):
        """
        Initialize mock channel
        
        Args:
            name: Channel name
            fail_on_send: If True, send_alert will fail
        """
        self.name = name
        self.fail_on_send = fail_on_send
        self.alerts_received: List[MisalignmentEvent] = []
    
    def send_alert(self, event: MisalignmentEvent) -> bool:
        """Send alert (mock)"""
        if self.fail_on_send:
            return False
        
        self.alerts_received.append(event)
        return True
    
    def get_channel_name(self) -> str:
        """Get channel name"""
        return self.name
    
    def get_alert_count(self) -> int:
        """Get number of alerts received"""
        return len(self.alerts_received)
    
    def clear_alerts(self):
        """Clear received alerts"""
        self.alerts_received.clear()


class AlertSystem:
    """
    Multi-channel alert dispatching system with rate limiting
    
    Sends alerts for HIGH and CRITICAL severity events (Property 14).
    """
    
    def __init__(
        self,
        channels: List[AlertChannel],
        rate_limit_seconds: float = 60.0,
        max_history_size: int = 1000
    ):
        """
        Initialize alert system
        
        Args:
            channels: List of alert channels
            rate_limit_seconds: Minimum seconds between alerts for same camera
            max_history_size: Maximum alert history size (bounded)
        """
        if rate_limit_seconds < 0:
            raise ValueError(f"rate_limit_seconds must be non-negative, got {rate_limit_seconds}")
        
        if max_history_size <= 0:
            raise ValueError(f"max_history_size must be positive, got {max_history_size}")
        
        self.channels = channels
        self.rate_limit_seconds = rate_limit_seconds
        self.max_history_size = max_history_size
        
        # Rate limiting state: camera_id -> last_alert_time
        self.last_alert_times: Dict[int, float] = {}
        
        # Alert history (bounded)
        self.alert_history: deque = deque(maxlen=max_history_size)
        
        # Statistics
        self.alerts_sent = 0
        self.alerts_rate_limited = 0
        self.alerts_filtered = 0  # LOW/MEDIUM severity
    
    def process_event(self, event: MisalignmentEvent) -> bool:
        """
        Process misalignment event and send alerts if needed
        
        Property 14: Sends alerts for HIGH and CRITICAL severity only.
        
        Args:
            event: MisalignmentEvent to process
        
        Returns:
            True if alert was sent
        """
        # Property 14: Only alert on HIGH and CRITICAL severity
        if event.severity not in [Severity.HIGH, Severity.CRITICAL]:
            self.alerts_filtered += 1
            return False
        
        # Check rate limiting
        if self._is_rate_limited(event.camera_id):
            self.alerts_rate_limited += 1
            return False
        
        # Send to all channels
        channels_sent = []
        any_success = False
        
        for channel in self.channels:
            try:
                success = channel.send_alert(event)
                if success:
                    channels_sent.append(channel.get_channel_name())
                    any_success = True
            except Exception:
                # Continue with other channels on failure
                pass
        
        # Record in history
        if any_success:
            record = AlertRecord(
                event_id=event.event_id,
                camera_id=event.camera_id,
                severity=event.severity,
                timestamp=event.timestamp,
                channels_sent=channels_sent,
                success=True
            )
            self.alert_history.append(record)
            self.alerts_sent += 1
            
            # Update rate limit timestamp
            self.last_alert_times[event.camera_id] = time.time()
            
            return True
        
        return False
    
    def process_events(self, events: List[MisalignmentEvent]) -> int:
        """
        Process multiple events
        
        Args:
            events: List of events to process
        
        Returns:
            Number of alerts sent
        """
        count = 0
        for event in events:
            if self.process_event(event):
                count += 1
        return count
    
    def _is_rate_limited(self, camera_id: int) -> bool:
        """
        Check if camera is rate limited
        
        Args:
            camera_id: Camera ID to check
        
        Returns:
            True if rate limited
        """
        if camera_id not in self.last_alert_times:
            return False
        
        elapsed = time.time() - self.last_alert_times[camera_id]
        return elapsed < self.rate_limit_seconds
    
    def get_alert_history(
        self,
        camera_id: Optional[int] = None,
        severity: Optional[Severity] = None,
        limit: Optional[int] = None
    ) -> List[AlertRecord]:
        """
        Get alert history with optional filters
        
        Args:
            camera_id: Filter by camera ID (optional)
            severity: Filter by severity (optional)
            limit: Limit number of results (optional)
        
        Returns:
            List of alert records (most recent first)
        """
        # Convert deque to list (most recent last)
        history = list(self.alert_history)
        
        # Apply filters
        if camera_id is not None:
            history = [r for r in history if r.camera_id == camera_id]
        
        if severity is not None:
            history = [r for r in history if r.severity == severity]
        
        # Reverse to get most recent first
        history.reverse()
        
        # Apply limit
        if limit is not None and limit > 0:
            history = history[:limit]
        
        return history
    
    def clear_rate_limit(self, camera_id: Optional[int] = None):
        """
        Clear rate limit for camera(s)
        
        Args:
            camera_id: Camera ID to clear (None = clear all)
        """
        if camera_id is None:
            self.last_alert_times.clear()
        else:
            if camera_id in self.last_alert_times:
                del self.last_alert_times[camera_id]
    
    def get_statistics(self) -> Dict:
        """Get alert system statistics"""
        return {
            'alerts_sent': self.alerts_sent,
            'alerts_rate_limited': self.alerts_rate_limited,
            'alerts_filtered': self.alerts_filtered,
            'history_size': len(self.alert_history),
            'history_capacity': self.max_history_size
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.alerts_sent = 0
        self.alerts_rate_limited = 0
        self.alerts_filtered = 0
