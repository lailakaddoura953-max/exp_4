"""
Location Rule Data Model and Loader for the Camera-Location-Aware Hazard
Rules engine.

This file answers "what does each location require?" and nothing else —
per the module-structure split requested for this project (see
requirements.md Requirement 9.2 and design.md's Overview). It owns:

  - PPERequirement / LocationRuleSet dataclasses (the rule data shape)
  - DEFAULT_RULES: the built-in, HSSE-sourced rule set for every
    Camera_Location_Type (requirements.md Requirement 2.3)
  - LocationRuleLoader: YAML override loading, field-level merging, and
    validation (requirements.md Requirement 8)

Requirements covered: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 8.1, 8.2, 8.3, 8.4, 8.5,
8.6, 8.7
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazard_detection.diagnostics import get_logger

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = get_logger("rule_engine.rules")


# ============================================================================
# Validation constants (Requirement 8.4)
# ============================================================================

VALID_HUMAN_PRESENCE_POLICIES = {"prohibited", "permitted", "conditional", "unknown"}
VALID_CONTAINER_CHECKS = {"misalignment", "open_doors", "flipped", "dangling"}

# Fail-safe Camera_Location_Type for unrecognised cameras (Requirement 7).
UNKNOWN_LOCATION_TYPE = "Unknown"


# ============================================================================
# Data model (Requirement 2.2)
# ============================================================================


@dataclass
class PPERequirement:
    """
    Compositional PPE requirement.

    `vest_required` covers vest OR hi-vis coveralls/jacket (HSSE answer 2:
    coveralls/jacket are an accepted substitute). `shoes_required` and
    `helmet_required` are tracked even though today's Detection_Classes
    can't verify them from the Human/Human-No-Safety-Clothes classes alone
    (requirements.md Requirement 3.4) — that limitation lives in
    check_rules_from_object_label.py, not here.
    """

    vest_required: bool = True
    shoes_required: bool = True  # terminal-wide baseline (HSSE answer 2); currently unverifiable
    helmet_required: bool = False
    # "any" | "climbing_chinstrap" | "conditional_on_context"
    helmet_type: str = "any"


@dataclass
class LocationRuleSet:
    """
    A structured set of hazard conditions that apply to a specific
    Camera_Location_Type (requirements.md Requirement 2.2).
    """

    location_type: str
    human_presence_policy: str  # "prohibited" | "permitted" | "conditional" | "unknown"
    ppe_requirement: PPERequirement = field(default_factory=PPERequirement)
    occupancy_limit: Optional[int] = None
    # Documented but NOT enforced — requirements.md Requirement 6.7. There is
    # no detection signal today for "maintenance activity is occurring".
    occupancy_maintenance_exception: bool = False
    container_checks_enabled: List[str] = field(default_factory=list)
    container_checks_suppressed: List[str] = field(default_factory=list)
    vehicle_checks_enabled: bool = False
    trucker_spot_check_enabled: bool = False
    notes: str = ""


# ============================================================================
# Built-in default rule sets (Requirement 2.1, 2.3, 2.7)
#
# Every entry below carries an inline comment naming the camera-name
# pattern(s) it applies to, the HSSE answer it's sourced from, and whether
# it is still pending confirmation — so a supervisor reading this file
# without Python experience can still follow the reasoning (Req 2.7).
# ============================================================================

DEFAULT_RULES: Dict[str, LocationRuleSet] = {
    # --- Berth --------------------------------------------------------
    # Cameras whose name contains "Berth" (e.g. "A10 - NE - Berth 404").
    # HSSE answer 2: the quay is a hard-hat zone. Vest + helmet required.
    # Humans permitted (this is the working quay).
    "Berth": LocationRuleSet(
        location_type="Berth",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=True),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
        notes="No OTR trucks on quay, mostly (HSSE answer 7; soft rule, not enforceable with current vehicle classes).",
    ),
    # --- Block ----------------------------------------------------------
    # Cameras whose name contains "Block" (e.g. "A8 - SE PTZ - Block 1F").
    # HSSE answer 4: inside the Automated Fenceline. No humans, ever,
    # except via a Safe Access Gate (Airlock/CEG) — separate camera types.
    "Block": LocationRuleSet(
        location_type="Block",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),  # irrelevant — no humans permitted at all
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
        notes="Inside the Automated Fenceline; only sanctioned entry points are Airlocks/CEG (HSSE answer 4).",
    ),
    # --- TELs (general TEL-lane camera) ----------------------------------
    # Cameras whose name contains "TELs" (checked before bare "TEL").
    # HSSE answer 4: 1 person at a time unless maintenance (documented but
    # NOT enforced — see Requirement 6.7). Vest only. Trucker spot check applies.
    "TELs": LocationRuleSet(
        location_type="TELs",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=False),
        occupancy_limit=1,
        occupancy_maintenance_exception=True,
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
        trucker_spot_check_enabled=True,
        notes="1-person occupancy limit unless maintenance (HSSE answer 4); maintenance exception documented, not enforced (Req 6.7).",
    ),
    # --- TEL (standalone numbered camera, e.g. "TEL 118") ----------------
    # Same policy as TELs, but these are single-lane cameras, not general
    # yard cameras — vehicle checks off.
    "TEL": LocationRuleSet(
        location_type="TEL",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=False),
        occupancy_limit=1,
        occupancy_maintenance_exception=True,
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=False,
        trucker_spot_check_enabled=True,
        notes="Vests mandatory; humans only in the dedicated spot or transiting to/from the truck (naming convention doc).",
    ),
    # --- Wall St ----------------------------------------------------------
    # Cameras whose name contains "Wall St". No humans; container + vehicle
    # checks enabled.
    "Wall St": LocationRuleSet(
        location_type="Wall St",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
    ),
    # --- Reefer Rack --------------------------------------------------------
    # Cameras whose name contains "Reefer rack". UPDATED from the original
    # "no humans" draft: mechanics work here. Vest only, per terminal-wide
    # baseline (not on the confirmed hard-hat list).
    "Reefer Rack": LocationRuleSet(
        location_type="Reefer Rack",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=False),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=False,
        notes="Mechanics work here — updated from earlier 'no humans' draft.",
    ),
    # --- Rail (generic) --------------------------------------------------
    # Cameras whose name contains "Rail" (and did NOT match the Rail
    # Storage prefix). PENDING HSSE CONFIRMATION: a maintenance/supervisor
    # exception (requiring hard hat + climbing-style chin-strap helmet per
    # the dock-aloft rule) has been proposed but NOT confirmed. DO NOT
    # implement that exception — every human here is prohibited until
    # HSSE confirms otherwise (requirements.md Requirement 3.7).
    "Rail": LocationRuleSet(
        location_type="Rail",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=False,
        notes=(
            "PENDING HSSE CONFIRMATION: maintenance/supervisor personnel may be "
            "a permitted exception requiring hard hat + climbing-style chin-strap "
            "helmet (dock-aloft rule, HSSE answer 5). NOT implemented — all humans "
            "treated as prohibited until confirmed. No OTR trucks in rail yard "
            "(HSSE answer 7; not enforceable with current vehicle classes)."
        ),
    ),
    # --- Rail Storage (sub-type of Rail) ---------------------------------
    # Cameras whose name STARTS WITH "Rail Storage". Same as Rail in every
    # field except the open-door check is suppressed — HSSE answer 8: open
    # doors here (esp. top level) are normal/expected.
    "Rail Storage": LocationRuleSet(
        location_type="Rail Storage",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        container_checks_suppressed=["open_doors"],
        vehicle_checks_enabled=False,
        notes="Open container doors expected/normal here, esp. top level (HSSE answer 8) — suppressed, not a bug.",
    ),
    # --- Pedestal -----------------------------------------------------------
    # Cameras whose name contains "Pedestal". No humans; container checks only.
    "Pedestal": LocationRuleSet(
        location_type="Pedestal",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=False,
    ),
    # --- Flipline -------------------------------------------------------------
    # Cameras whose name contains "Flipline" or "Flip Line" (HSSE doc spells
    # it with a space). UPDATED from original draft: humans permitted (this
    # is a manual inspection point). Vest + helmet. Open doors expected/
    # normal here during inspection (HSSE answer 8) — suppressed.
    "Flipline": LocationRuleSet(
        location_type="Flipline",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=True),
        container_checks_enabled=["misalignment", "flipped", "dangling"],
        container_checks_suppressed=["open_doors"],
        vehicle_checks_enabled=False,
        notes="Open container doors expected/normal here during inspection (HSSE answer 8) — suppressed, not a bug.",
    ),
    # --- Plug Reefer -----------------------------------------------------------
    # Cameras whose name contains "Plug Reefer". Helmet requirement is NOT
    # explicitly confirmed by HSSE — defaulted to required (fail-safe)
    # until clarified.
    "Plug Reefer": LocationRuleSet(
        location_type="Plug Reefer",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=True),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
        notes="Helmet requirement not explicitly confirmed by HSSE — defaulting to required (fail-safe) until clarified.",
    ),
    # --- Airlocks -----------------------------------------------------------
    # Cameras whose name contains "Airlocks". HSSE answer 1: presence is
    # gate-state-dependent (unobservable by camera) — do not flag presence
    # itself. PPE IS observable — check it. Hard hat required only if
    # working under a straddle carrier inside the Airlock; that specific
    # context is not currently detectable.
    "Airlocks": LocationRuleSet(
        location_type="Airlocks",
        human_presence_policy="conditional",
        ppe_requirement=PPERequirement(
            vest_required=True,
            shoes_required=True,
            helmet_required=False,
            helmet_type="conditional_on_context",
        ),
        container_checks_enabled=[],
        vehicle_checks_enabled=False,
        notes=(
            "Presence conditional on automation gate being closed (HSSE answer 1) — "
            "not directly observable, so presence itself is not flagged. Hard hat "
            "required only if working under a straddle carrier inside the Airlock; "
            "that specific context is not currently detectable, so helmet is tracked "
            "but not enforced here."
        ),
    ),
    # --- CEG -----------------------------------------------------------
    # Cameras whose name contains "CEG". HSSE answer 1: presence conditional
    # on automation gate closed AND both manual gates open. No hard-hat
    # exception documented for CEG (unlike Airlocks).
    "CEG": LocationRuleSet(
        location_type="CEG",
        human_presence_policy="conditional",
        ppe_requirement=PPERequirement(vest_required=True, shoes_required=True, helmet_required=False),
        container_checks_enabled=[],
        vehicle_checks_enabled=False,
        notes="Presence conditional on automation gate closed AND both manual gates open (HSSE answer 1) — not directly observable.",
    ),
    # --- AssetManagement -----------------------------------------------------
    # Cameras whose name contains "AssetManagement"/"Asset Management".
    # UPDATED from original draft: vest + helmet, not vest-only — HSSE
    # flagged this as "possibly," treated conservatively (fail-safe).
    "AssetManagement": LocationRuleSet(
        location_type="AssetManagement",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=True),
        container_checks_enabled=[],
        vehicle_checks_enabled=True,
        notes="Human PPE rules still marked uncertain by HSSE ('possibly') — treated as required (fail-safe) until confirmed.",
    ),
    # --- Exit Fuel -----------------------------------------------------
    # Cameras whose name contains "Exit Fuel". Vest only, per the
    # terminal-wide baseline — Exit Fuel is not on the confirmed hard-hat
    # list. HSSE explicitly flagged this location as still needing
    # confirmation.
    "Exit Fuel": LocationRuleSet(
        location_type="Exit Fuel",
        human_presence_policy="permitted",
        ppe_requirement=PPERequirement(vest_required=True, helmet_required=False),
        container_checks_enabled=[],
        vehicle_checks_enabled=True,
        notes="HSSE explicitly flagged this as still needing confirmation — not final.",
    ),
    # --- VACIS -----------------------------------------------------
    # Cameras whose name contains "VACIS" (naming pattern unconfirmed,
    # requirements.md Requirement 1.9 — matches on substring until a real
    # example name is available). HSSE did not address human presence at
    # VACIS — only that open doors are expected/normal there (answer 8).
    "VACIS": LocationRuleSet(
        location_type="VACIS",
        human_presence_policy="unknown",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "flipped", "dangling"],
        container_checks_suppressed=["open_doors"],
        vehicle_checks_enabled=False,
        notes="Human presence policy not addressed by HSSE. Open doors expected/normal (HSSE answer 8) — suppressed.",
    ),
    # --- Unknown (fail-safe) -----------------------------------------------------
    # Any camera name that matches no known pattern (requirements.md
    # Requirement 7). Strictest possible rules applied.
    "Unknown": LocationRuleSet(
        location_type="Unknown",
        human_presence_policy="prohibited",
        ppe_requirement=PPERequirement(),
        container_checks_enabled=["misalignment", "open_doors", "flipped", "dangling"],
        vehicle_checks_enabled=True,
        notes="Fail-safe: strictest rules applied for unrecognised camera.",
    ),
}


# ============================================================================
# LocationRuleLoader (Requirement 2.4, 2.6, 8.1-8.7)
# ============================================================================

DEFAULT_RULES_CONFIG_PATH = "config/location_rules.yaml"


class LocationRuleLoader:
    """
    Owns DEFAULT_RULES and loads config/location_rules.yaml overrides on
    top of them.

    - Missing YAML file -> use DEFAULT_RULES entirely, log one warning
      (Requirement 8.3).
    - Field-level override merging: overriding one field of a
      LocationRuleSet does not reset the others to defaults (Requirement 8.2).
    - Invalid entries are rejected individually; that entry falls back to
      its built-in default while other entries still apply (Requirement 8.5).
    - Also loads `camera_name_overrides` and `camera_id_to_name` sections
      from the same YAML file (Requirement 8.6, 9.6).
    """

    def __init__(self, config_path: Optional[str] = DEFAULT_RULES_CONFIG_PATH):
        self._config_path = config_path
        self._rule_sets: Dict[str, LocationRuleSet] = dict(DEFAULT_RULES)
        self.camera_name_overrides: Dict[str, str] = {}
        self.camera_id_to_name: Dict[str, str] = {}

        if config_path and Path(config_path).is_file():
            self._load_yaml_overrides(config_path)
        else:
            logger.warning(
                f"Location rules config '{config_path}' not found; using "
                f"built-in DEFAULT_RULES with no overrides."
            )

    def get_rule_set(self, location_type: str) -> LocationRuleSet:
        """
        Return the LocationRuleSet for the given location type, falling
        back to the "Unknown" fail-safe rule set for unrecognised types.
        """
        return self._rule_sets.get(location_type, self._rule_sets[UNKNOWN_LOCATION_TYPE])

    # ------------------------------------------------------------------
    # YAML loading
    # ------------------------------------------------------------------

    def _load_yaml_overrides(self, config_path: str) -> None:
        if not YAML_AVAILABLE:
            logger.error(
                "PyYAML is not installed. Cannot read location_rules.yaml. "
                "Install with: pip install pyyaml. Using built-in DEFAULT_RULES."
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f.read())
        except Exception as e:
            logger.error(
                f"Failed to read/parse '{config_path}': {e}. Using built-in "
                f"DEFAULT_RULES with no overrides."
            )
            return

        if not isinstance(data, dict):
            logger.error(
                f"Location rules config '{config_path}' did not parse to a "
                f"mapping; using built-in DEFAULT_RULES with no overrides."
            )
            return

        self._apply_location_type_overrides(data.get("location_type_overrides"))
        self._load_camera_name_overrides(data.get("camera_name_overrides"))
        self._load_camera_id_to_name(data.get("camera_id_to_name"))

    def _apply_location_type_overrides(self, overrides: Any) -> None:
        if overrides is None:
            return
        if not isinstance(overrides, dict):
            logger.error(
                f"'location_type_overrides' must be a mapping, got "
                f"{type(overrides).__name__}; ignoring."
            )
            return

        for location_type, field_overrides in overrides.items():
            if location_type not in self._rule_sets:
                logger.error(
                    f"'location_type_overrides' references unknown "
                    f"Camera_Location_Type '{location_type}'; ignoring this entry."
                )
                continue

            if not isinstance(field_overrides, dict):
                logger.error(
                    f"Override for '{location_type}' must be a mapping, got "
                    f"{type(field_overrides).__name__}; ignoring this entry."
                )
                continue

            merged = self._merge_rule_set_fields(
                self._rule_sets[location_type], field_overrides, location_type
            )
            if merged is not None:
                self._rule_sets[location_type] = merged

    def _merge_rule_set_fields(
        self,
        base: LocationRuleSet,
        field_overrides: Dict[str, Any],
        location_type: str,
    ) -> Optional[LocationRuleSet]:
        """
        Merge field_overrides onto a copy of base, field by field, applying
        validation (Requirement 8.4). Any single invalid field is rejected
        (logged) and that field's built-in default value is kept — the
        entire entry is NOT discarded just because one field is bad.
        """
        candidate = LocationRuleSet(
            location_type=base.location_type,
            human_presence_policy=base.human_presence_policy,
            ppe_requirement=PPERequirement(
                vest_required=base.ppe_requirement.vest_required,
                shoes_required=base.ppe_requirement.shoes_required,
                helmet_required=base.ppe_requirement.helmet_required,
                helmet_type=base.ppe_requirement.helmet_type,
            ),
            occupancy_limit=base.occupancy_limit,
            occupancy_maintenance_exception=base.occupancy_maintenance_exception,
            container_checks_enabled=list(base.container_checks_enabled),
            container_checks_suppressed=list(base.container_checks_suppressed),
            vehicle_checks_enabled=base.vehicle_checks_enabled,
            trucker_spot_check_enabled=base.trucker_spot_check_enabled,
            notes=base.notes,
        )

        for key, value in field_overrides.items():
            if key == "human_presence_policy":
                if value not in VALID_HUMAN_PRESENCE_POLICIES:
                    logger.error(
                        f"Invalid human_presence_policy '{value}' for "
                        f"'{location_type}'; keeping default "
                        f"'{base.human_presence_policy}'."
                    )
                    continue
                candidate.human_presence_policy = value

            elif key == "ppe_requirement":
                if not isinstance(value, dict):
                    logger.error(
                        f"'ppe_requirement' override for '{location_type}' must "
                        f"be a mapping; ignoring."
                    )
                    continue
                for ppe_key, ppe_value in value.items():
                    if ppe_key in ("vest_required", "shoes_required", "helmet_required"):
                        if not isinstance(ppe_value, bool):
                            logger.error(
                                f"'ppe_requirement.{ppe_key}' for '{location_type}' "
                                f"must be boolean; ignoring this field."
                            )
                            continue
                        setattr(candidate.ppe_requirement, ppe_key, ppe_value)
                    elif ppe_key == "helmet_type":
                        setattr(candidate.ppe_requirement, ppe_key, ppe_value)
                    else:
                        logger.error(
                            f"Unknown ppe_requirement field '{ppe_key}' for "
                            f"'{location_type}'; ignoring."
                        )

            elif key == "occupancy_limit":
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                    logger.error(
                        f"'occupancy_limit' for '{location_type}' must be a "
                        f"positive integer; keeping default "
                        f"'{base.occupancy_limit}'."
                    )
                    continue
                candidate.occupancy_limit = value

            elif key == "occupancy_maintenance_exception":
                if not isinstance(value, bool):
                    logger.error(
                        f"'occupancy_maintenance_exception' for "
                        f"'{location_type}' must be boolean; ignoring."
                    )
                    continue
                candidate.occupancy_maintenance_exception = value

            elif key in ("container_checks_enabled", "container_checks_suppressed"):
                if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value
                ):
                    logger.error(
                        f"'{key}' for '{location_type}' must be a list of "
                        f"strings; ignoring."
                    )
                    continue
                unknown = [v for v in value if v not in VALID_CONTAINER_CHECKS]
                if unknown:
                    logger.error(
                        f"'{key}' for '{location_type}' contains unknown check "
                        f"name(s) {unknown}; ignoring this field."
                    )
                    continue
                setattr(candidate, key, list(value))

            elif key == "vehicle_checks_enabled":
                if not isinstance(value, bool):
                    logger.error(
                        f"'vehicle_checks_enabled' for '{location_type}' must "
                        f"be boolean; keeping default "
                        f"'{base.vehicle_checks_enabled}'."
                    )
                    continue
                candidate.vehicle_checks_enabled = value

            elif key == "trucker_spot_check_enabled":
                if not isinstance(value, bool):
                    logger.error(
                        f"'trucker_spot_check_enabled' for '{location_type}' "
                        f"must be boolean; ignoring."
                    )
                    continue
                candidate.trucker_spot_check_enabled = value

            elif key == "notes":
                if not isinstance(value, str):
                    logger.error(
                        f"'notes' for '{location_type}' must be a string; ignoring."
                    )
                    continue
                candidate.notes = value

            else:
                logger.error(
                    f"Unknown LocationRuleSet field '{key}' in override for "
                    f"'{location_type}'; ignoring."
                )

        return candidate

    def _load_camera_name_overrides(self, overrides: Any) -> None:
        if overrides is None:
            return
        if not isinstance(overrides, dict):
            logger.error(
                f"'camera_name_overrides' must be a mapping, got "
                f"{type(overrides).__name__}; ignoring."
            )
            return

        valid: Dict[str, str] = {}
        for camera_name, location_type in overrides.items():
            if not isinstance(camera_name, str) or not isinstance(location_type, str):
                logger.error(
                    f"camera_name_overrides entry ({camera_name!r} -> "
                    f"{location_type!r}) must be string -> string; ignoring."
                )
                continue
            valid[camera_name] = location_type

        self.camera_name_overrides = valid

    def _load_camera_id_to_name(self, mapping: Any) -> None:
        if mapping is None:
            return
        if not isinstance(mapping, dict):
            logger.error(
                f"'camera_id_to_name' must be a mapping, got "
                f"{type(mapping).__name__}; ignoring."
            )
            return

        valid: Dict[str, str] = {}
        for camera_id, camera_name in mapping.items():
            if not isinstance(camera_id, str) or not isinstance(camera_name, str):
                logger.error(
                    f"camera_id_to_name entry ({camera_id!r} -> "
                    f"{camera_name!r}) must be string -> string; ignoring."
                )
                continue
            valid[camera_id] = camera_name

        self.camera_id_to_name = valid
