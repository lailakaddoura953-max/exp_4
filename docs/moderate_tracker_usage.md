# ModerateClassificationTracker Usage Guide

## Overview

The `ModerateClassificationTracker` tracks consecutive moderate classifications per strad and generates warning notifications when a strad receives 3 consecutive moderate classifications within a 24-hour window.

## Requirements Addressed

- **Requirement 11.2**: Allow strads with moderate classifications to remain in eligible pool
- **Requirement 11.3**: Apply normal cooldown period (1 hour) to moderate classified strads
- **Requirement 11.5**: Track consecutive moderate classifications for trend analysis
- **Requirement 11.6**: Generate warning when 3 consecutive moderates within 24 hours

## Basic Usage

```python
from src.strad_monitoring.database import DatabaseInterface, ModerateClassificationTracker

# Initialize database interface
db = DatabaseInterface(
    connection_string="...",
    enable_fallback=True
)

# Initialize tracker
tracker = ModerateClassificationTracker(
    database_interface=db,
    time_window_hours=24
)

# Record a classification
tracker.record_classification(
    strad_id='SC042',
    classification='moderate',
    confidence=0.65
)
```

## How It Works

### 1. In-Memory Counter

The tracker maintains an in-memory counter for each strad that tracks consecutive moderate classifications:

```python
tracker._consecutive_counts = {
    'SC042': 2,  # 2 consecutive moderates
    'SC078': 1,  # 1 moderate
}
```

### 2. Database Query

When recording a classification, the tracker queries the database for recent classifications within the 24-hour window:

```sql
SELECT classification, created_at
FROM classification_results
WHERE strad_id = ?
  AND created_at >= ?  -- 24 hours ago
  AND created_at <= ?  -- current time
ORDER BY created_at DESC
```

### 3. Consecutive Count Logic

The tracker counts consecutive moderates by:

1. Starting with the current classification
2. Counting backwards through recent history
3. Stopping at the first non-moderate classification

**Example 1**: All moderates
```
Current: moderate (count = 1)
Recent:  moderate (count = 2)
         moderate (count = 3) ← Warning triggered!
```

**Example 2**: Mixed classifications
```
Current: moderate (count = 1)
Recent:  moderate (count = 2)
         none     (stop counting)
         moderate (not counted)
Final count: 2 (no warning)
```

### 4. Warning Notification

When a strad reaches **exactly 3** consecutive moderate classifications, a warning is generated:

```python
send_consecutive_moderate_alert(
    strad_id='SC042',
    consecutive_count=3,
    time_window_hours=24
)
```

The alert includes:
- Strad ID
- Consecutive count
- Time window (24 hours)
- Recommendation for manual inspection

### 5. Counter Reset

The counter resets when:
- A non-moderate classification (none or critical) occurs
- Manual reset is triggered

```python
# Automatic reset on non-moderate
tracker.record_classification('SC042', 'none', 0.95)
# Counter for SC042 is now 0

# Manual reset
tracker.reset_counter('SC042')
```

## Integration with Orchestrator

The tracker should be integrated into the orchestrator's `process_single_strad()` method:

```python
class MonitoringOrchestrator:
    def __init__(self, config):
        self.db = DatabaseInterface(...)
        self.moderate_tracker = ModerateClassificationTracker(self.db)
    
    def process_single_strad(self, strad_id):
        # ... capture snapshot, classify ...
        
        classification = dl_classifier.classify_snapshot(snapshot)
        
        # Store result in database
        self.db.store_classification_result(
            strad_id=strad_id,
            classification=classification.severity,
            confidence=classification.confidence,
            snapshot_path=snapshot_path if classification.severity == 'critical' else None
        )
        
        # Track moderate classifications
        self.moderate_tracker.record_classification(
            strad_id=strad_id,
            classification=classification.severity,
            confidence=classification.confidence
        )
        
        # Update check history
        self.db.update_check_history(strad_id)
```

## API Reference

### `ModerateClassificationTracker`

#### Constructor

```python
ModerateClassificationTracker(
    database_interface: DatabaseInterface,
    time_window_hours: int = 24
)
```

**Parameters:**
- `database_interface`: DatabaseInterface instance for querying classification history
- `time_window_hours`: Time window for tracking consecutive moderates (default: 24)

#### Methods

##### `record_classification()`

```python
record_classification(
    strad_id: str,
    classification: str,
    confidence: float,
    timestamp: Optional[datetime] = None
) -> None
```

Record a classification and update consecutive moderate tracking.

**Parameters:**
- `strad_id`: Strad CHE number (e.g., 'SC042')
- `classification`: Classification result ('none', 'moderate', 'critical')
- `confidence`: Classification confidence score (0.0-1.0)
- `timestamp`: Classification timestamp (default: current time)

**Behavior:**
- Queries database for recent classifications within time window
- Counts consecutive moderate classifications
- Updates in-memory counter
- Generates warning if threshold reached (3 consecutive moderates)
- Resets counter if non-moderate classification

##### `get_consecutive_count()`

```python
get_consecutive_count(strad_id: str) -> int
```

Get current consecutive moderate count for a strad.

**Returns:** Current consecutive moderate count (0 if none)

##### `reset_counter()`

```python
reset_counter(strad_id: str) -> None
```

Reset consecutive moderate counter for a strad. Called when non-moderate classification occurs or manual reset needed.

##### `get_all_counts()`

```python
get_all_counts() -> Dict[str, int]
```

Get all consecutive moderate counts.

**Returns:** Dictionary mapping strad_id to consecutive moderate count

##### `clear_all_counters()`

```python
clear_all_counters() -> None
```

Clear all consecutive moderate counters. Used for testing or system reset.

## Example Scenarios

### Scenario 1: Gradual Misalignment

A strad gradually develops misalignment over 12 hours:

```
Hour 0:  Classification: none      → Counter: 0
Hour 2:  Classification: moderate  → Counter: 1
Hour 4:  Classification: moderate  → Counter: 2
Hour 6:  Classification: moderate  → Counter: 3 → ⚠️ WARNING SENT
Hour 8:  Classification: moderate  → Counter: 4 (no new warning)
```

**Result:** Operators receive warning at hour 6, can schedule inspection

### Scenario 2: Transient Issue

A strad has temporary misalignment that self-corrects:

```
Hour 0:  Classification: none      → Counter: 0
Hour 2:  Classification: moderate  → Counter: 1
Hour 4:  Classification: none      → Counter: 0 (reset)
Hour 6:  Classification: moderate  → Counter: 1
```

**Result:** No warning generated, counter resets when issue resolves

### Scenario 3: Critical Escalation

Misalignment escalates from moderate to critical:

```
Hour 0:  Classification: moderate  → Counter: 1
Hour 2:  Classification: moderate  → Counter: 2
Hour 4:  Classification: critical  → Counter: 0 (reset)
                                    → Strad added to exclusion list
```

**Result:** Strad excluded from monitoring until adjustment confirmed

## Testing

The tracker includes comprehensive unit tests covering:

- Consecutive counting logic
- Counter reset on non-moderate classifications
- Warning notification at threshold
- Database query integration
- Error handling for database unavailability

Run tests:
```bash
python -m pytest tests/unit/test_moderate_tracker.py -v
```

## Fallback Behavior

When database is unavailable (local testing mode):
- `_query_recent_classifications()` returns empty list
- Consecutive count is based only on in-memory state
- No false warnings generated during local testing

## Performance Considerations

- **Database Queries**: One query per classification record (optimized with indexed columns)
- **Memory Usage**: Minimal - only stores integer counters per strad
- **Time Complexity**: O(n) where n is number of classifications within window (typically < 24)

## Future Enhancements

Potential improvements:
1. **Configurable threshold**: Allow different warning thresholds per strad
2. **Multiple time windows**: Track both 24h and 7-day trends
3. **Severity progression**: Detect escalation patterns (none → moderate → critical)
4. **Alert escalation**: Multiple notification levels for different consecutive counts
