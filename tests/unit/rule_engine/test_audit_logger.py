"""
Unit tests for AuditLogger.

Requirements covered: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.models import BBox
from hazard_detection.rule_engine.audit_logger import (
    AuditLogger,
    OUTCOME_CHECK_DISABLED,
    OUTCOME_HAZARD_EMITTED,
    OUTCOME_SUPPRESSED,
)


@pytest.fixture
def tmp_log_path(tmp_path) -> str:
    return str(tmp_path / "rule_audit.jsonl")


def _log_one(logger: AuditLogger, outcome: str, rule_name: str = "human_presence_prohibited"):
    logger.log_evaluation(
        camera_name="A8 - SE PTZ - Block 1F",
        location_type="Block",
        detection_class="Human",
        confidence=0.9,
        bbox=BBox(x_center=0.5, y_center=0.5, width=0.1, height=0.2),
        rule_name=rule_name,
        outcome=outcome,
        frame_index=0,
    )


class TestJsonLinesOutput:
    def test_creates_log_directory_if_missing(self, tmp_path):
        nested_path = tmp_path / "nested" / "dir" / "rule_audit.jsonl"
        AuditLogger(audit_log_path=str(nested_path))
        assert nested_path.parent.is_dir()

    def test_writes_valid_json_per_line_with_required_fields(self, tmp_log_path):
        logger = AuditLogger(audit_log_path=tmp_log_path)
        _log_one(logger, OUTCOME_HAZARD_EMITTED)

        lines = Path(tmp_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        for field in (
            "timestamp", "camera_name", "location_type", "detection_class",
            "confidence", "bbox", "rule_name", "outcome", "frame_index",
        ):
            assert field in entry

        assert entry["camera_name"] == "A8 - SE PTZ - Block 1F"
        assert entry["location_type"] == "Block"
        assert entry["outcome"] == OUTCOME_HAZARD_EMITTED
        assert entry["bbox"]["x_center"] == 0.5

    def test_multiple_entries_are_one_json_object_per_line(self, tmp_log_path):
        logger = AuditLogger(audit_log_path=tmp_log_path)
        _log_one(logger, OUTCOME_HAZARD_EMITTED)
        _log_one(logger, OUTCOME_NO_HAZARD := "no_hazard")

        lines = Path(tmp_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # each line must independently parse as JSON


class TestSuppressedVsDisabled:
    def test_suppressed_and_check_disabled_are_distinct_values(self, tmp_log_path):
        logger = AuditLogger(audit_log_path=tmp_log_path)
        _log_one(logger, OUTCOME_SUPPRESSED, rule_name="container_open_doors_check")
        _log_one(logger, OUTCOME_CHECK_DISABLED, rule_name="container_open_doors_check")

        lines = Path(tmp_log_path).read_text(encoding="utf-8").strip().splitlines()
        outcomes = [json.loads(line)["outcome"] for line in lines]
        assert outcomes == [OUTCOME_SUPPRESSED, OUTCOME_CHECK_DISABLED]
        assert OUTCOME_SUPPRESSED != OUTCOME_CHECK_DISABLED


class TestWriteFailureResilience:
    def test_invalid_path_does_not_raise_on_log_evaluation(self):
        # Point the audit log at a path whose parent cannot be created
        # (a file used as if it were a directory) to force a write failure.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            blocking_file_path = f.name

        bad_path = str(Path(blocking_file_path) / "rule_audit.jsonl")
        logger = AuditLogger(audit_log_path=bad_path)

        # Must not raise despite the underlying write failing.
        _log_one(logger, OUTCOME_HAZARD_EMITTED)
        _log_one(logger, OUTCOME_HAZARD_EMITTED)  # second failure path (debug-logged)

    def test_unrecognised_outcome_does_not_raise(self, tmp_log_path):
        logger = AuditLogger(audit_log_path=tmp_log_path)
        # Should log a warning but still write the entry, not raise.
        _log_one(logger, outcome="not_a_real_outcome")
        lines = Path(tmp_log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
