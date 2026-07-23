"""
Camera Location Resolver for the Camera-Location-Aware Hazard Rules engine.

Parses Ocularis camera display name strings into Camera_Location_Type values,
per requirements.md Requirement 1 and design.md's CameraLocationResolver design.

Deterministic: the same Camera_Name always resolves to the same
Camera_Location_Type. No external configuration file is required to perform
this parsing — the keyword list is embedded directly in this module, in a
single clearly-commented block, so it stays easy to alter (per the
Introduction's "easy, low-risk editing" goal).

Requirements covered: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
"""

import re
from typing import Dict, List, Optional, Tuple

from hazard_detection.diagnostics import get_logger

logger = get_logger("rule_engine.camera_location_resolver")


# Camera_Location_Type value returned when a Camera_Name matches none of the
# known patterns below (Requirement 1.5). Triggers fail-safe rules (Req 7).
UNKNOWN_LOCATION_TYPE = "Unknown"


class CameraLocationResolver:
    """
    Parses Ocularis camera name strings into Camera_Location_Type values.

    Deterministic: same input always returns same output. No external
    config required — keyword list is built in and commented with the
    HSSE/naming-convention rationale for each entry (Requirement 1.8).

    Matching priority (Requirement 1.3, 1.4):
      1. "Rail Storage" prefix match  (must run BEFORE generic "Rail")
      2. Exact "TEL <digits>" pattern (e.g. "TEL 118")
      3. Keyword search (see LOCATION_KEYWORDS, in priority order)
      4. Generic "Rail" substring match
      5. Unknown fallback (logs a warning)
    """

    # "Rail Storage" cameras must be identified BEFORE the generic "Rail"
    # substring check below, or they would always be misclassified as
    # plain "Rail" (requirements.md Requirement 1.3).
    RAIL_STORAGE_PREFIX = re.compile(r"^Rail Storage", re.IGNORECASE)

    # Standalone numbered TEL cameras, e.g. "TEL 118" (requirements.md 1.1).
    # Naming convention.JSON also shows "TEL 01 - 144" — allow an optional
    # " - <digits>" suffix after the initial number.
    TEL_NUMBER_PATTERN = re.compile(r"^TEL\s+\d{1,3}(\s*-\s*\d+)?\s*$", re.IGNORECASE)

    # Priority-ordered (keyword, Camera_Location_Type) pairs. Order matters:
    # more specific / collision-prone keywords are checked first so that,
    # e.g., "TELs" is matched before a bare "TEL" substring could ever be
    # considered, and "Flip Line" (HSSE doc spelling) is recognized
    # alongside "Flipline" (naming-convention spelling).
    LOCATION_KEYWORDS: List[Tuple[str, str]] = [
        # HSSE Q8 / naming convention: quay-adjacent zone, hard-hat area.
        ("Wall St", "Wall St"),
        # HSSE: mechanics work here, vest-only baseline.
        ("Reefer rack", "Reefer Rack"),
        # HSSE: vest + helmet required (helmet unconfirmed, defaulted true).
        ("Plug Reefer", "Plug Reefer"),
        # HSSE: PPE "possibly" required — treated as required (fail-safe).
        ("AssetManagement", "AssetManagement"),
        ("Asset Management", "AssetManagement"),  # naming convention spells it "AssetMangement"/no space variants
        # HSSE: still flagged as needing confirmation.
        ("Exit Fuel", "Exit Fuel"),
        # HSSE Q1: gate-state-conditional presence, vest+shoes baseline.
        ("Airlocks", "Airlocks"),
        # HSSE Q8: open container doors expected/normal here (suppressed).
        ("Flipline", "Flipline"),
        ("Flip Line", "Flipline"),  # HSSE follow-up doc spells it with a space
        # No humans; container checks only.
        ("Pedestal", "Pedestal"),
        # HSSE Q8: open doors expected/normal here too. Naming pattern
        # unconfirmed (Requirement 1.9) — match on substring until a real
        # example name is available.
        ("VACIS", "VACIS"),
        # Quay; hard-hat zone per HSSE baseline rule.
        ("Berth", "Berth"),
        # Inside the Automated Fenceline; humans prohibited.
        ("Block", "Block"),
        # Must be checked before a bare "TEL" substring could match, since
        # "TELs" contains "TEL" as a substring.
        ("TELs", "TELs"),
        # HSSE Q1: gate-state-conditional presence, same baseline as Airlocks.
        ("CEG", "CEG"),
        # "Rail" is intentionally NOT listed here — it's handled explicitly
        # in resolve() after the Rail Storage prefix check, so the ordering
        # between "Rail Storage" and generic "Rail" is unambiguous rather
        # than depending on keyword list position.
    ]

    def resolve(self, camera_name: str) -> str:
        """
        Return the Camera_Location_Type for a given Camera_Name string.

        Matching priority:
          1. "Rail Storage" prefix match
          2. Exact TEL number pattern (e.g. "TEL 118")
          3. Keyword search (LOCATION_KEYWORDS, in order)
          4. Generic "Rail" substring
          5. Unknown fallback (logs a warning with the full Camera_Name)

        Args:
            camera_name: The full Ocularis display name of the camera.

        Returns:
            The resolved Camera_Location_Type string, or "Unknown" if no
            pattern matched.
        """
        if not isinstance(camera_name, str) or not camera_name.strip():
            logger.warning(
                f"CameraLocationResolver received an empty/invalid camera_name "
                f"({camera_name!r}); resolving to '{UNKNOWN_LOCATION_TYPE}'"
            )
            return UNKNOWN_LOCATION_TYPE

        name = camera_name.strip()

        # 1. Rail Storage prefix — MUST run before the generic Rail check.
        if self.RAIL_STORAGE_PREFIX.search(name):
            return "Rail Storage"

        # 2. Standalone numbered TEL camera, e.g. "TEL 118".
        if self.TEL_NUMBER_PATTERN.match(name):
            return "TEL"

        # 3. Keyword search, in priority order.
        for keyword, location_type in self.LOCATION_KEYWORDS:
            if keyword.lower() in name.lower():
                return location_type

        # 4. Generic "Rail" substring (only reached if Rail Storage above
        #    did not match).
        if re.search(r"Rail", name, re.IGNORECASE):
            return "Rail"

        # 5. Unknown fallback.
        logger.warning(
            f"CameraLocationResolver could not resolve camera_name "
            f"'{camera_name}' to a known Camera_Location_Type; "
            f"resolving to '{UNKNOWN_LOCATION_TYPE}' (fail-safe rules apply)"
        )
        return UNKNOWN_LOCATION_TYPE

    def resolve_with_override(
        self, camera_name: str, overrides: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Resolve a Camera_Name, checking operator-defined overrides first.

        Requirement 7.4 / 8.6: operators can map specific Camera_Name
        strings to a Camera_Location_Type via the `camera_name_overrides`
        section of the location rules YAML, without any code change. This
        is the mechanism used to correct unconfirmed cases (VACIS naming,
        restroom-area cameras, `ADM Parking`, `FleetView Recording`, etc.).

        Args:
            camera_name: The full Ocularis display name of the camera.
            overrides: Mapping of exact Camera_Name -> Camera_Location_Type.
                       Matched by exact string equality (case-sensitive),
                       since these are meant to be precise, operator-curated
                       entries, not fuzzy keyword matches.

        Returns:
            The overridden Camera_Location_Type if camera_name is an exact
            key in overrides; otherwise the result of resolve(camera_name).
        """
        if overrides and camera_name in overrides:
            override_type = overrides[camera_name]
            logger.debug(
                f"CameraLocationResolver: camera_name_overrides match for "
                f"'{camera_name}' -> '{override_type}'"
            )
            return override_type

        return self.resolve(camera_name)
