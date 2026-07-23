"""
Unit tests for TrainingFolderLocationResolver.

Requirements covered: 11.2
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.rule_engine.training_folder_location_resolver import (
    TrainingFolderLocationResolver,
    UNKNOWN_LOCATION_TYPE,
)


@pytest.fixture
def resolver() -> TrainingFolderLocationResolver:
    return TrainingFolderLocationResolver()


class TestKnownFolders:
    def test_berth_401_resolves_to_berth(self, resolver):
        assert resolver.resolve_folder("berth_401") == "Berth"

    def test_berth_405_and_knuckle_resolves_to_berth(self, resolver):
        assert resolver.resolve_folder("berth_405_and_knuckle") == "Berth"


class TestUnmappedFolders:
    def test_unmapped_folder_resolves_unknown(self, resolver):
        assert resolver.resolve_folder("rail_yard_9") == UNKNOWN_LOCATION_TYPE

    def test_empty_string_resolves_unknown(self, resolver):
        assert resolver.resolve_folder("") == UNKNOWN_LOCATION_TYPE

    def test_none_input_resolves_unknown(self, resolver):
        assert resolver.resolve_folder(None) == UNKNOWN_LOCATION_TYPE  # type: ignore[arg-type]


class TestNotAKeywordScan:
    def test_folder_containing_berth_substring_but_unmapped_is_unknown(self, resolver):
        # Confirms this resolver does NOT do substring/keyword matching
        # like CameraLocationResolver — only exact table lookups.
        assert resolver.resolve_folder("berth_999_unlisted") == UNKNOWN_LOCATION_TYPE
