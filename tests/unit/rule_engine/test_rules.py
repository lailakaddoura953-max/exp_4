"""
Unit tests for rules.py: PPERequirement/LocationRuleSet dataclasses,
DEFAULT_RULES, and LocationRuleLoader's YAML override/validation logic.

Requirements covered: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 8.1-8.7
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.rule_engine.rules import (
    DEFAULT_RULES,
    LocationRuleLoader,
    LocationRuleSet,
    PPERequirement,
    UNKNOWN_LOCATION_TYPE,
)

ALL_LOCATION_TYPES = [
    "Berth", "Block", "TELs", "TEL", "Wall St", "Reefer Rack", "Rail",
    "Rail Storage", "Pedestal", "Flipline", "Plug Reefer", "Airlocks",
    "CEG", "AssetManagement", "Exit Fuel", "VACIS",
]


def _write_yaml(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(data, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# DEFAULT_RULES completeness (Requirement 2.1, 2.3)
# ---------------------------------------------------------------------------

class TestDefaultRulesCompleteness:
    @pytest.mark.parametrize("location_type", ALL_LOCATION_TYPES)
    def test_all_16_location_types_defined(self, location_type):
        assert location_type in DEFAULT_RULES
        rule_set = DEFAULT_RULES[location_type]
        assert isinstance(rule_set, LocationRuleSet)
        assert rule_set.location_type == location_type

    def test_unknown_failsafe_defined(self):
        assert UNKNOWN_LOCATION_TYPE in DEFAULT_RULES
        rule_set = DEFAULT_RULES[UNKNOWN_LOCATION_TYPE]
        assert rule_set.human_presence_policy == "prohibited"
        assert rule_set.vehicle_checks_enabled is True
        assert set(rule_set.container_checks_enabled) == {
            "misalignment", "open_doors", "flipped", "dangling",
        }
        assert rule_set.container_checks_suppressed == []


# ---------------------------------------------------------------------------
# Specific field values from the Location Rule Logic Table (Requirement 2.3)
# ---------------------------------------------------------------------------

class TestDefaultRulesFieldValues:
    def test_berth_permitted_vest_and_helmet(self):
        r = DEFAULT_RULES["Berth"]
        assert r.human_presence_policy == "permitted"
        assert r.ppe_requirement.vest_required is True
        assert r.ppe_requirement.helmet_required is True
        assert r.vehicle_checks_enabled is True

    def test_block_prohibited(self):
        r = DEFAULT_RULES["Block"]
        assert r.human_presence_policy == "prohibited"

    def test_tels_occupancy_limit_one_with_unenforced_exception(self):
        r = DEFAULT_RULES["TELs"]
        assert r.human_presence_policy == "permitted"
        assert r.occupancy_limit == 1
        assert r.occupancy_maintenance_exception is True
        assert r.trucker_spot_check_enabled is True
        assert r.vehicle_checks_enabled is True

    def test_tel_standalone_vehicle_checks_disabled(self):
        r = DEFAULT_RULES["TEL"]
        assert r.vehicle_checks_enabled is False
        assert r.trucker_spot_check_enabled is True

    def test_reefer_rack_permitted_updated(self):
        r = DEFAULT_RULES["Reefer Rack"]
        assert r.human_presence_policy == "permitted"
        assert r.ppe_requirement.helmet_required is False

    def test_rail_prohibited_and_pending_note(self):
        r = DEFAULT_RULES["Rail"]
        assert r.human_presence_policy == "prohibited"
        assert "PENDING" in r.notes.upper()

    def test_rail_storage_suppresses_open_doors_only(self):
        r = DEFAULT_RULES["Rail Storage"]
        assert r.human_presence_policy == "prohibited"
        assert r.container_checks_suppressed == ["open_doors"]
        assert set(r.container_checks_enabled) == {
            "misalignment", "open_doors", "flipped", "dangling",
        }

    def test_flipline_permitted_vest_and_helmet_open_doors_suppressed(self):
        r = DEFAULT_RULES["Flipline"]
        assert r.human_presence_policy == "permitted"
        assert r.ppe_requirement.helmet_required is True
        assert r.container_checks_suppressed == ["open_doors"]
        assert "open_doors" not in r.container_checks_enabled

    def test_plug_reefer_helmet_defaulted_true(self):
        r = DEFAULT_RULES["Plug Reefer"]
        assert r.ppe_requirement.helmet_required is True

    def test_airlocks_conditional_with_context_dependent_helmet(self):
        r = DEFAULT_RULES["Airlocks"]
        assert r.human_presence_policy == "conditional"
        assert r.ppe_requirement.helmet_type == "conditional_on_context"
        assert r.container_checks_enabled == []

    def test_ceg_conditional_no_helmet_exception(self):
        r = DEFAULT_RULES["CEG"]
        assert r.human_presence_policy == "conditional"
        assert r.ppe_requirement.helmet_required is False

    def test_asset_management_permitted_vest_and_helmet(self):
        r = DEFAULT_RULES["AssetManagement"]
        assert r.human_presence_policy == "permitted"
        assert r.ppe_requirement.helmet_required is True
        assert r.vehicle_checks_enabled is True

    def test_exit_fuel_permitted_vest_only(self):
        r = DEFAULT_RULES["Exit Fuel"]
        assert r.human_presence_policy == "permitted"
        assert r.ppe_requirement.helmet_required is False

    def test_vacis_unknown_policy_open_doors_suppressed(self):
        r = DEFAULT_RULES["VACIS"]
        assert r.human_presence_policy == "unknown"
        assert r.container_checks_suppressed == ["open_doors"]


# ---------------------------------------------------------------------------
# LocationRuleLoader: missing file (Requirement 8.3)
# ---------------------------------------------------------------------------

class TestLoaderMissingFile:
    def test_missing_yaml_falls_back_to_defaults(self):
        loader = LocationRuleLoader(config_path="does/not/exist.yaml")
        for location_type in ALL_LOCATION_TYPES:
            assert loader.get_rule_set(location_type) == DEFAULT_RULES[location_type]

    def test_none_config_path_uses_defaults(self):
        loader = LocationRuleLoader(config_path=None)
        assert loader.get_rule_set("Berth").human_presence_policy == "permitted"

    def test_unknown_type_falls_back_to_unknown_rule_set(self):
        loader = LocationRuleLoader(config_path=None)
        rs = loader.get_rule_set("Some Made Up Type")
        assert rs.location_type == "Unknown"


# ---------------------------------------------------------------------------
# LocationRuleLoader: field-level override merging (Requirement 8.2)
# ---------------------------------------------------------------------------

class TestLoaderFieldLevelMerging:
    def test_override_one_field_preserves_others(self):
        path = _write_yaml({
            "location_type_overrides": {
                "AssetManagement": {"vehicle_checks_enabled": False},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("AssetManagement")
        assert rs.vehicle_checks_enabled is False
        # Un-overridden fields retain their built-in defaults.
        assert rs.human_presence_policy == "permitted"
        assert rs.ppe_requirement.helmet_required is True

    def test_override_nested_ppe_field(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Plug Reefer": {"ppe_requirement": {"helmet_required": False}},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Plug Reefer")
        assert rs.ppe_requirement.helmet_required is False
        assert rs.ppe_requirement.vest_required is True  # untouched

    def test_override_notes_only(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Rail": {"notes": "Confirmed by HSSE on 2026-08-01."},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Rail")
        assert rs.notes == "Confirmed by HSSE on 2026-08-01."
        assert rs.human_presence_policy == "prohibited"  # untouched


# ---------------------------------------------------------------------------
# LocationRuleLoader: validation (Requirement 8.4, 8.5)
# ---------------------------------------------------------------------------

class TestLoaderValidation:
    def test_invalid_human_presence_policy_rejected_falls_back_to_default(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Block": {"human_presence_policy": "sometimes"},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Block")
        assert rs.human_presence_policy == "prohibited"  # default retained

    def test_unknown_check_name_rejected(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Berth": {"container_checks_enabled": ["misalignment", "not_a_real_check"]},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Berth")
        # Entire field rejected -> falls back to built-in default list.
        assert set(rs.container_checks_enabled) == {
            "misalignment", "open_doors", "flipped", "dangling",
        }

    def test_non_boolean_vehicle_checks_enabled_rejected(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Berth": {"vehicle_checks_enabled": "yes"},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Berth")
        assert rs.vehicle_checks_enabled is True  # default retained

    def test_non_positive_occupancy_limit_rejected(self):
        path = _write_yaml({
            "location_type_overrides": {
                "TEL": {"occupancy_limit": 0},
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("TEL")
        assert rs.occupancy_limit == 1  # default retained

    def test_unknown_location_type_key_ignored(self):
        path = _write_yaml({
            "location_type_overrides": {
                "NotARealLocation": {"human_presence_policy": "permitted"},
            }
        })
        # Should not raise; unrecognised key is simply skipped/logged.
        loader = LocationRuleLoader(config_path=path)
        assert loader.get_rule_set("Berth") == DEFAULT_RULES["Berth"]

    def test_one_bad_field_does_not_discard_other_valid_fields_in_same_entry(self):
        path = _write_yaml({
            "location_type_overrides": {
                "Berth": {
                    "vehicle_checks_enabled": "not-a-bool",  # invalid
                    "notes": "Updated note text.",  # valid
                },
            }
        })
        loader = LocationRuleLoader(config_path=path)
        rs = loader.get_rule_set("Berth")
        assert rs.vehicle_checks_enabled is True  # default retained
        assert rs.notes == "Updated note text."  # valid override still applied


# ---------------------------------------------------------------------------
# camera_name_overrides / camera_id_to_name loading (Requirement 8.6, 9.6)
# ---------------------------------------------------------------------------

class TestCameraMappingsLoaded:
    def test_camera_name_overrides_loaded(self):
        path = _write_yaml({
            "camera_name_overrides": {"ADM Parking": "AssetManagement"},
        })
        loader = LocationRuleLoader(config_path=path)
        assert loader.camera_name_overrides == {"ADM Parking": "AssetManagement"}

    def test_camera_id_to_name_loaded(self):
        path = _write_yaml({
            "camera_id_to_name": {"cam_01": "A8 - SE PTZ - Block 1F"},
        })
        loader = LocationRuleLoader(config_path=path)
        assert loader.camera_id_to_name == {"cam_01": "A8 - SE PTZ - Block 1F"}

    def test_missing_mappings_default_to_empty_dicts(self):
        loader = LocationRuleLoader(config_path=None)
        assert loader.camera_name_overrides == {}
        assert loader.camera_id_to_name == {}

    def test_non_string_camera_name_override_entry_ignored(self):
        path = _write_yaml({
            "camera_name_overrides": {"ADM Parking": 123},
        })
        loader = LocationRuleLoader(config_path=path)
        assert loader.camera_name_overrides == {}
