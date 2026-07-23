"""
Audit Logger for the Camera-Location-Aware Hazard Rules engine.

Writes a JSON-lines audit trail of every rule evaluation decision, so the
system's reasoning can be reconstructed after incidents and so HSSE can
verify their stated rules are actually being applied as described
(requirements.md Requirement 10).

Requirements covered: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hazard_detection.diagnostics import get_logger
from hazard_detection.models import BBox

logger = get_logger("rule_engine.audit_logger")

DEFAULT_AUDIT_LOG_PATH = "logs/rule_audit.jsonl"

# Outcome values (Requirement 10.4). "suppressed" is intentionally distinct
# from "check_disabled" so a suppressed check (an EXPECTED, documented
# suppression per Requirement 4) is visibly different in the audit trail
# from a check that was simply never enabled for that location.
OUTCOME_HAZARD_EMITTED = "hazard_emitted"
OUTCOME_NO_HAZARD = "no_hazard"
OUTCOME_CHECK_DISABLED = "check_disabled"
OUTCOME_SUPPRESSED = "suppressed"
OUTCOME_POLICY_UNKNOWN = "policy_unknown"

VALID_OUTCOMES = {
    OUTCOME_HAZARD_EMITTED,
    OUTCOME_NO_HAZARD,
    OUTCOME_CHECK_DISABLED,
    OUTCOME_SUPPRESSED,
    OUTCOME_POLICY_UNKNOWN,
}


class AuditLogger:
    """
    Writes one JSON object per line to a dedicated audit log file,
    separate from the main operational log (Requirement 10.2, 10.3).

    Never raises: if writing fails (disk full, permission error, etc.),
    the failure is logged to the operational logger and processing
    continues — audit log failure SHALL NOT halt hazard detection
    (Requirement 10.5).
    """

    def __init__(self, audit_log_path: str = DEFAULT_AUDIT_LOG_PATH):
        self._audit_log_path = Path(audit_log_path)
        self._write_failed_once = False

        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(
                f"AuditLogger could not create audit log directory "
                f"'{self._audit_log_path.parent}': {e}. Audit logging will "
                f"be attempted anyway and failures will be logged per-entry."
            )

    def log_evaluation(
        self,
        camera_name: str,
        location_type: str,
        detection_class: str,
        confidence: float,
        bbox: BBox,
        rule_name: str,
        outcome: str,
        frame_index: int,
    ) -> None:
        """
        Write one audit log entry for a single detection's rule evaluation.

        Args:
            camera_name: The full Ocularis Camera_Name (or training folder
                name, when called from the dataset-correction path).
            location_type: The resolved Camera_Location_Type.
            detection_class: The raw YOLO/label class of the detection.
            confidence: The detection's confidence score.
            bbox: The detection's bounding box.
            rule_name: The matched rule name (e.g. "human_presence_prohibited",
                "container_open_doors_check").
            outcome: One of VALID_OUTCOMES.
            frame_index: 0-based index of the frame within the sequence.

        Never raises. Write failures are caught, logged to the operational
        logger, and swallowed (Requirement 10.5).
        """
        if outcome not in VALID_OUTCOMES:
            logger.warning(
                f"AuditLogger.log_evaluation() received an unrecognised "
                f"outcome '{outcome}'; logging it anyway for forensic value, "
                f"but this indicates a caller bug."
            )

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "camera_name": camera_name,
            "location_type": location_type,
            "detection_class": detection_class,
            "confidence": confidence,
            "bbox": {
                "x_center": bbox.x_center,
                "y_center": bbox.y_center,
                "width": bbox.width,
                "height": bbox.height,
            },
            "rule_name": rule_name,
            "outcome": outcome,
            "frame_index": frame_index,
        }

        self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self._audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str))
                f.write("\n")
        except OSError as e:
            # Only log the first failure loudly per logger instance to avoid
            # flooding the operational log if the disk stays unavailable;
            # subsequent failures are still logged, but at a lower level.
            if not self._write_failed_once:
                logger.error(
                    f"AuditLogger failed to write to '{self._audit_log_path}': "
                    f"{e}. Hazard detection will continue; audit trail for "
                    f"this entry (and possibly subsequent entries) is lost."
                )
                self._write_failed_once = True
            else:
                logger.debug(
                    f"AuditLogger write failure persists for "
                    f"'{self._audit_log_path}': {e}"
                )
