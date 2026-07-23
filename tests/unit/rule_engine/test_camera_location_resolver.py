"""
Unit tests for CameraLocationResolver.

Validates camera-name-to-location-type parsing against every example in
`Naming convention.JSON`, plus the HSSE-driven edge cases: Rail Storage
prefix precedence, TELs/TEL distinction, Flip Line/Flipline spelling
variants, VACIS substring matching, override precedence, determinism,
and the Unknown fallback.

Requirements covered: 1.1-1.8
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from hazard_detection.rule_engine.camera_location_resolver import (
    CameraLocationResolver,
    UNKNOWN_LOCATION_TYPE,
)


@pytest.fixture
def resolver() -> CameraLocationResolver:
    return CameraLocationResolver()


# ---------------------------------------------------------------------------
# Naming convention.JSON examples (Requirement 1.6)
# ---------------------------------------------------------------------------

class TestNamingConventionExamples:
    def test_berth_with_grid_and_direction(self, resolver):
        assert resolver.resolve("A10 - NE - Berth 404") == "Berth"

    def test_grid_to_grid_reference_is_unknown(self, resolver):
        assert resolver.resolve("A2 - B") == "Unknown"

    def test_block_with_ptz_suffix(self, resolver):
        assert resolver.resolve("A8 - SE PTZ - Block 1F") == "Block"

    def test_block_with_letter_ptz_variant(self, resolver):
        assert resolver.resolve("G13 - C PTZ - Block 4A") == "Block"

    def test_plug_reefer_with_direction_word(self, resolver):
        assert resolver.resolve("J10 - A PTZ North - Plug Reefer") == "Plug Reefer"

    def test_standalone_tel_camera(self, resolver):
        assert resolver.resolve("TEL 118") == "TEL"

    def test_exit_fuel_with_ptz(self, resolver):
        assert resolver.resolve("K11 - D PTZ - Exit Fuel") == "Exit Fuel"

    def test_tel_with_dash_number_suffix(self, resolver):
        # "TEL 01 - 144" from Naming convention.JSON's list of special cases
        assert resolver.resolve("TEL 01 - 144") == "TEL"

    @pytest.mark.parametrize(
        "camera_name",
        ["Causeway Sign #1", "ADM Parking", "FleetView Recording"],
    )
    def test_non_location_special_cases_resolve_unknown(self, resolver, camera_name):
        # These appear in Naming convention.JSON as special-purpose cameras
        # with no location keyword; they should resolve Unknown and rely on
        # camera_name_overrides (Requirement 7.4) for correct assignment.
        assert resolver.resolve(camera_name) == "Unknown"


# ---------------------------------------------------------------------------
# Rail Storage prefix precedence (Requirement 1.3, 1.4)
# ---------------------------------------------------------------------------

class TestRailStoragePrecedence:
    def test_rail_storage_prefix_resolves_to_rail_storage(self, resolver):
        assert resolver.resolve("Rail Storage 12 - North") == "Rail Storage"

    def test_rail_storage_case_insensitive(self, resolver):
        assert resolver.resolve("rail storage 3 - south") == "Rail Storage"

    def test_generic_rail_substring_resolves_to_rail(self, resolver):
        assert resolver.resolve("B4 - N PTZ - Rail 12") == "Rail"

    def test_rail_storage_never_misclassified_as_plain_rail(self, resolver):
        # A name that would match generic "Rail" but starts with the more
        # specific "Rail Storage" prefix must resolve to Rail Storage.
        result = resolver.resolve("Rail Storage - Rail Yard Overview")
        assert result == "Rail Storage"
        assert result != "Rail"

    def test_rail_storage_mid_string_falls_back_to_generic_rail(self, resolver):
        # "Rail Storage" appears mid-string here, not as a PREFIX, so per
        # Requirement 1.3 ("starts with") it should NOT resolve to
        # Rail Storage — it falls through to the generic Rail substring
        # match instead.
        assert resolver.resolve("C5 - Rail Storage Overview") == "Rail"


# ---------------------------------------------------------------------------
# TELs vs TEL distinction (Requirement 1.1, 1.2)
# ---------------------------------------------------------------------------

class TestTelVsTels:
    def test_tels_resolves_to_tels(self, resolver):
        assert resolver.resolve("D6 - N PTZ - TELs 3") == "TELs"

    def test_bare_tel_number_resolves_to_tel(self, resolver):
        assert resolver.resolve("TEL 118") == "TEL"

    def test_tels_and_tel_are_distinct(self, resolver):
        assert resolver.resolve("D6 - N PTZ - TELs 3") != resolver.resolve("TEL 118")


# ---------------------------------------------------------------------------
# Flip Line / Flipline spelling variants (Requirement 1.1)
# ---------------------------------------------------------------------------

class TestFliplineSpellingVariants:
    def test_flipline_no_space(self, resolver):
        assert resolver.resolve("E7 - S PTZ - Flipline") == "Flipline"

    def test_flip_line_with_space(self, resolver):
        assert resolver.resolve("E7 - S PTZ - Flip Line") == "Flipline"


# ---------------------------------------------------------------------------
# VACIS substring matching (Requirement 1.9)
# ---------------------------------------------------------------------------

class TestVacisMatching:
    def test_vacis_substring_anywhere_resolves_to_vacis(self, resolver):
        assert resolver.resolve("H2 - VACIS Inspection Area") == "VACIS"

    def test_vacis_case_insensitive(self, resolver):
        assert resolver.resolve("h2 - vacis area") == "VACIS"


# ---------------------------------------------------------------------------
# Override precedence (Requirement 7.4 / 8.6)
# ---------------------------------------------------------------------------

class TestOverridePrecedence:
    def test_override_takes_precedence_over_keyword_scan(self, resolver):
        overrides = {"ADM Parking": "AssetManagement"}
        assert (
            resolver.resolve_with_override("ADM Parking", overrides)
            == "AssetManagement"
        )

    def test_override_takes_precedence_even_when_keyword_would_match(self, resolver):
        # Even though "Berth 404" would normally resolve to "Berth", an
        # explicit override for the exact name must win.
        overrides = {"A10 - NE - Berth 404": "Wall St"}
        assert (
            resolver.resolve_with_override("A10 - NE - Berth 404", overrides)
            == "Wall St"
        )

    def test_no_override_falls_back_to_resolve(self, resolver):
        overrides = {"Some Other Camera": "Block"}
        assert (
            resolver.resolve_with_override("A10 - NE - Berth 404", overrides)
            == "Berth"
        )

    def test_none_overrides_falls_back_to_resolve(self, resolver):
        assert resolver.resolve_with_override("TEL 118", None) == "TEL"


# ---------------------------------------------------------------------------
# Determinism (Requirement 1.7)
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize(
        "camera_name",
        [
            "A10 - NE - Berth 404",
            "A8 - SE PTZ - Block 1F",
            "TEL 118",
            "Rail Storage 12 - North",
            "totally-unrecognized-camera-xyz",
        ],
    )
    def test_same_input_always_resolves_the_same(self, resolver, camera_name):
        results = {resolver.resolve(camera_name) for _ in range(10)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Unknown fallback (Requirement 1.5)
# ---------------------------------------------------------------------------

class TestUnknownFallback:
    def test_unrecognized_name_resolves_unknown(self, resolver):
        assert resolver.resolve("totally-unrecognized-camera-xyz") == UNKNOWN_LOCATION_TYPE

    def test_empty_string_resolves_unknown(self, resolver):
        assert resolver.resolve("") == UNKNOWN_LOCATION_TYPE

    def test_none_input_resolves_unknown(self, resolver):
        assert resolver.resolve(None) == UNKNOWN_LOCATION_TYPE  # type: ignore[arg-type]
