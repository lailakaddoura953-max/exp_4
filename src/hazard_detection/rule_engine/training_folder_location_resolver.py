"""
Training Folder Location Resolver for the Camera-Location-Aware Hazard Rules engine.

Maps `image_data_with_synth/` location-folder names (e.g. "berth_401",
"berth_405_and_knuckle") to Camera_Location_Type values, per requirements.md
Requirement 11.2 and design.md's TrainingFolderLocationResolver design.

This is deliberately a SEPARATE resolver from CameraLocationResolver: the
synthetic-generation pipeline's folder-naming convention (lowercase,
underscored, no PTZ/grid-reference prefixes) does not follow the live
Ocularis naming convention, so reusing the same keyword-scan regex would be
incorrect, not just inconvenient. Folder names are controlled by whoever ran
the synthetic-image generation, so an explicit mapping table is safer than
pattern-guessing.

Requirements covered: 11.2
"""

from typing import Dict

from hazard_detection.diagnostics import get_logger

logger = get_logger("rule_engine.training_folder_location_resolver")


# Camera_Location_Type value returned when a training-data folder name is
# not present in FOLDER_TO_LOCATION_TYPE. Fail-safe rules (Requirement 7)
# apply to Unknown locations, same as at runtime.
UNKNOWN_LOCATION_TYPE = "Unknown"


class TrainingFolderLocationResolver:
    """
    Maps image_data_with_synth/ location-folder names to Camera_Location_Type
    values via an explicit lookup table (not a keyword scan).
    """

    # Explicit mapping table. Every entry is commented with its source.
    # Add new location folders here as they appear in future synthetic-
    # generation runs. Unmapped folders resolve to "Unknown" and are
    # treated fail-safe (Requirement 7), not silently guessed at.
    FOLDER_TO_LOCATION_TYPE: Dict[str, str] = {
        # Berth 401 camera group.
        "berth_401": "Berth",
        # Berth 405 and the adjoining knuckle — both are quay (Berth) zone.
        "berth_405_and_knuckle": "Berth",
    }

    def resolve_folder(self, folder_name: str) -> str:
        """
        Return the Camera_Location_Type for a given image_data_with_synth/
        location-folder name.

        Args:
            folder_name: The location-folder name (e.g. "berth_401").

        Returns:
            The mapped Camera_Location_Type, or "Unknown" (with a warning
            logged) if folder_name is not present in FOLDER_TO_LOCATION_TYPE.
        """
        if not isinstance(folder_name, str) or not folder_name.strip():
            logger.warning(
                f"TrainingFolderLocationResolver received an empty/invalid "
                f"folder_name ({folder_name!r}); resolving to "
                f"'{UNKNOWN_LOCATION_TYPE}'"
            )
            return UNKNOWN_LOCATION_TYPE

        location_type = self.FOLDER_TO_LOCATION_TYPE.get(folder_name)
        if location_type is None:
            logger.warning(
                f"TrainingFolderLocationResolver: unmapped training folder "
                f"'{folder_name}'; resolving to '{UNKNOWN_LOCATION_TYPE}' "
                f"(fail-safe rules apply). Add an entry to "
                f"FOLDER_TO_LOCATION_TYPE once its real-world location is known."
            )
            return UNKNOWN_LOCATION_TYPE

        return location_type
