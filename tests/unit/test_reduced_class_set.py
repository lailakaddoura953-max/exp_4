"""
Unit tests for the shared Reduced_Class_Set taxonomy
(src/hazard_detection/rule_engine/class_taxonomy.py) and its consistent
use across scripts/generate_hazard_augmentations.py and
scripts/pretrain_hazard_sanity_check.py.

Requirements covered: 12.1, 12.2, 12.5
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.rule_engine.class_taxonomy import (
    DROPPED_CLASS_INDICES,
    FULL_CLASS_NAMES,
    FULL_TO_REDUCED_INDEX,
    REDUCED_CLASS_SET,
)

DROPPED_CLASS_NAMES = {
    "Boat - With Cargo",
    "Container - Reefer",
    "Container - Water Drop",
    "Container -Separate",
    "Yard - Dropoff zone",
}


class TestReducedClassSetShape:
    def test_has_exactly_twelve_entries(self):
        assert len(REDUCED_CLASS_SET) == 12

    def test_does_not_include_any_dropped_class(self):
        for dropped_name in DROPPED_CLASS_NAMES:
            assert dropped_name not in REDUCED_CLASS_SET

    def test_full_class_names_has_seventeen_entries(self):
        assert len(FULL_CLASS_NAMES) == 17

    def test_dropped_indices_correspond_to_dropped_names(self):
        dropped_from_full = {FULL_CLASS_NAMES[i] for i in DROPPED_CLASS_INDICES}
        assert dropped_from_full == DROPPED_CLASS_NAMES

    def test_reduced_class_set_preserves_relative_order(self):
        # REDUCED_CLASS_SET should be FULL_CLASS_NAMES with the dropped
        # entries removed, not independently reordered.
        expected = [
            name for i, name in enumerate(FULL_CLASS_NAMES)
            if i not in DROPPED_CLASS_INDICES
        ]
        assert REDUCED_CLASS_SET == expected

    def test_no_duplicate_class_names(self):
        assert len(REDUCED_CLASS_SET) == len(set(REDUCED_CLASS_SET))


class TestFullToReducedIndexMapping:
    def test_dropped_indices_absent_from_mapping(self):
        for dropped_index in DROPPED_CLASS_INDICES:
            assert dropped_index not in FULL_TO_REDUCED_INDEX

    def test_mapping_covers_all_kept_indices(self):
        kept_indices = [i for i in range(17) if i not in DROPPED_CLASS_INDICES]
        assert sorted(FULL_TO_REDUCED_INDEX.keys()) == kept_indices

    def test_mapping_values_are_valid_reduced_indices(self):
        for new_index in FULL_TO_REDUCED_INDEX.values():
            assert 0 <= new_index < 12

    def test_mapping_preserves_class_identity(self):
        for old_index, new_index in FULL_TO_REDUCED_INDEX.items():
            assert FULL_CLASS_NAMES[old_index] == REDUCED_CLASS_SET[new_index]


class TestScriptsReferenceSharedList:
    """
    Verify generate_hazard_augmentations.py and pretrain_hazard_sanity_check.py
    both reference the SAME shared class_taxonomy module, with no
    independently-maintained copy of the class list.
    """

    def test_generate_hazard_augmentations_uses_shared_full_class_names(self):
        scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import generate_hazard_augmentations as hazard_gen

        assert hazard_gen.CLASS_NAMES is FULL_CLASS_NAMES

    def test_pretrain_hazard_sanity_check_uses_shared_full_class_names(self):
        scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import pretrain_hazard_sanity_check as sanity_check

        assert sanity_check.FULL_CLASS_NAMES is FULL_CLASS_NAMES
