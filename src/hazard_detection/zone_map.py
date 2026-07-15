"""
Zone Map for the Hazard Detection System.

Provides spatial zone management with polygon-based region detection,
runtime reload support, and YOLO-based zone initialization.

Uses ray-casting algorithm for point-in-polygon tests with normalized
coordinates (0.0 to 1.0 relative to camera resolution).

Requirements covered:
- 10.1: Configurable zone definitions via YAML/JSON with polygonal regions
- 10.2: Three zone types matching Roboflow annotations
- 10.3: Default entire FOV to no-people zone when no config exists
- 10.4: Associate zone definitions with specific camera identifiers
- 10.5: Runtime reload without system restart
- 10.6: Reject invalid config, retain previous definitions
- 10.7: Initialize from YOLO zone class predictions
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazard_detection.diagnostics import get_logger
from hazard_detection.models import Detection, ZonePolygon

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


logger = get_logger("zone_map")

# Valid zone types matching Roboflow annotations
VALID_ZONE_TYPES = {"no_people", "operation", "dropoff"}

# Mapping from Roboflow YOLO class labels to zone types
YOLO_CLASS_TO_ZONE_TYPE = {
    "Yard - No People": "no_people",
    "Yard - Operation Zone": "operation",
    "Yard - Dropoff zone": "dropoff",
}


class ZoneMap:
    """
    Spatial zone management with runtime reload support.

    Defines polygonal regions within each camera's field of view as
    no-people zones, operation zones, or dropoff zones. Uses ray-casting
    algorithm for point-in-polygon tests with normalized coordinates.

    If no zone config exists for a camera, the entire FOV defaults to
    a no-people zone (unsafe assumption).
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Load zone definitions from YAML/JSON config file.

        If no config provided, defaults to treating entire FOV as no-people zone
        for all cameras. Validates polygons: >=3 vertices, coordinates in [0.0, 1.0],
        known zone types.

        Args:
            config_path: Path to a YAML or JSON zone configuration file.
                        If None, no zones are loaded (all cameras default to no-people).

        Raises:
            No exceptions raised on invalid config — validation errors are logged
            and the invalid file is rejected while retaining previous definitions.
        """
        self._config_path: Optional[str] = config_path
        self._zones: Dict[str, List[ZonePolygon]] = {}

        if config_path:
            self._load_config(config_path)

    def get_zone_type(self, camera_id: str, point: Tuple[float, float]) -> str:
        """
        Returns the zone type for a normalized point in a camera's FOV.

        Uses ray-casting algorithm for point-in-polygon testing. If the point
        falls within multiple overlapping zones, the first matching zone is returned
        (zones are checked in definition order).

        Args:
            camera_id: Camera identifier to look up zones for.
            point: Normalized (x, y) coordinates in [0.0, 1.0].

        Returns:
            Zone type string: 'no_people', 'operation', or 'dropoff'.
            Returns 'no_people' if:
              - No zones are defined for the camera
              - The point is outside all defined zones
        """
        x, y = point

        # Get zones for this camera
        camera_zones = self._zones.get(camera_id, [])

        if not camera_zones:
            # No config for this camera — entire FOV is no-people zone
            logger.debug(
                f"No zones defined for camera '{camera_id}', "
                f"defaulting to 'no_people' for point ({x:.4f}, {y:.4f})"
            )
            return "no_people"

        # Check each zone polygon using ray-casting
        for zone in camera_zones:
            if self._point_in_polygon(x, y, zone.vertices):
                logger.debug(
                    f"Point ({x:.4f}, {y:.4f}) in camera '{camera_id}' "
                    f"falls within '{zone.zone_type}' zone"
                )
                return zone.zone_type

        # Point outside all defined zones — treat as no-people zone
        logger.debug(
            f"Point ({x:.4f}, {y:.4f}) in camera '{camera_id}' "
            f"outside all zones, defaulting to 'no_people'"
        )
        return "no_people"

    def reload(self) -> bool:
        """
        Reload zone definitions from the config file.

        Rejects invalid data and retains previous definitions on failure.
        New definitions take effect immediately for subsequent lookups.

        Returns:
            True if reload was successful, False if validation failed
            (previous definitions retained).
        """
        if not self._config_path:
            logger.warning("reload() called but no config_path was set")
            return False

        if not os.path.exists(self._config_path):
            logger.warning(
                f"Zone config file not found at '{self._config_path}', "
                f"retaining previous definitions"
            )
            return False

        logger.info(f"Reloading zone definitions from '{self._config_path}'")

        # Save previous zones in case validation fails
        previous_zones = dict(self._zones)

        try:
            # Attempt to load new config into a fresh state
            self._zones = {}
            success = self._load_config_with_status(self._config_path)

            if success:
                logger.info(
                    f"Zone reload successful: {sum(len(z) for z in self._zones.values())} "
                    f"zones loaded for {len(self._zones)} cameras"
                )
                return True
            else:
                # Validation failed — restore previous definitions
                self._zones = previous_zones
                logger.warning(
                    "Zone reload validation failed, retaining previous definitions"
                )
                return False

        except Exception as e:
            # Restore previous definitions on any failure
            self._zones = previous_zones
            logger.error(
                f"Zone reload failed with error: {e}. "
                f"Retaining previous definitions.",
                exc_info=True,
            )
            return False

    def initialize_from_detections(
        self,
        camera_id: str,
        zone_detections: List[Detection],
        confidence_threshold: float,
    ) -> None:
        """
        Initialize zone boundaries from YOLO zone class predictions.

        Used when no manual config exists for a camera. Creates zone polygons
        from detection bounding boxes that meet the confidence threshold.

        Args:
            camera_id: Camera identifier to initialize zones for.
            zone_detections: List of Detection objects with zone class labels
                           (e.g., "Yard - No People", "Yard - Operation Zone",
                            "Yard - Dropoff zone").
            confidence_threshold: Minimum confidence for zone predictions to be used.
        """
        logger.info(
            f"Initializing zones for camera '{camera_id}' from "
            f"{len(zone_detections)} YOLO detections "
            f"(threshold: {confidence_threshold})"
        )

        new_zones: List[ZonePolygon] = []

        for detection in zone_detections:
            # Filter by confidence
            if detection.confidence < confidence_threshold:
                logger.debug(
                    f"Skipping zone detection '{detection.class_label}' "
                    f"(confidence {detection.confidence:.3f} < {confidence_threshold})"
                )
                continue

            # Map YOLO class label to zone type
            zone_type = YOLO_CLASS_TO_ZONE_TYPE.get(detection.class_label)
            if zone_type is None:
                logger.debug(
                    f"Detection class '{detection.class_label}' is not a zone class, skipping"
                )
                continue

            # Convert bounding box to polygon vertices
            # BBox is (x_center, y_center, width, height) normalized
            bbox = detection.bbox
            half_w = bbox.width / 2.0
            half_h = bbox.height / 2.0

            # Clamp vertices to [0.0, 1.0]
            x_min = max(0.0, bbox.x_center - half_w)
            x_max = min(1.0, bbox.x_center + half_w)
            y_min = max(0.0, bbox.y_center - half_h)
            y_max = min(1.0, bbox.y_center + half_h)

            vertices = [
                (x_min, y_min),  # top-left
                (x_max, y_min),  # top-right
                (x_max, y_max),  # bottom-right
                (x_min, y_max),  # bottom-left
            ]

            try:
                zone_polygon = ZonePolygon(
                    vertices=vertices,
                    zone_type=zone_type,
                    camera_id=camera_id,
                )
                new_zones.append(zone_polygon)
                logger.debug(
                    f"Created '{zone_type}' zone from detection "
                    f"(confidence: {detection.confidence:.3f})"
                )
            except ValueError as e:
                logger.warning(
                    f"Failed to create zone polygon from detection: {e}"
                )

        if new_zones:
            self._zones[camera_id] = new_zones
            logger.info(
                f"Initialized {len(new_zones)} zones for camera '{camera_id}' "
                f"from YOLO detections"
            )
        else:
            # No valid predictions — default no-people behavior applies
            logger.info(
                f"No valid zone detections for camera '{camera_id}' "
                f"above threshold {confidence_threshold}. "
                f"Default no-people zone behavior will apply."
            )

    def get_zones_for_camera(self, camera_id: str) -> List[ZonePolygon]:
        """
        Get all zone polygons defined for a specific camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            List of ZonePolygon objects, or empty list if no zones defined.
        """
        return list(self._zones.get(camera_id, []))

    @property
    def camera_ids(self) -> List[str]:
        """Return list of camera IDs that have zone definitions."""
        return list(self._zones.keys())

    @property
    def total_zones(self) -> int:
        """Return total number of zone polygons across all cameras."""
        return sum(len(zones) for zones in self._zones.values())

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_config(self, config_path: str) -> None:
        """
        Load and validate zone configuration from a YAML or JSON file.

        Expected file format:
            zones:
              cam_01:
                - zone_type: "no_people"
                  vertices: [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
                - zone_type: "operation"
                  vertices: [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5]]
              cam_02:
                - zone_type: "dropoff"
                  vertices: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

        Args:
            config_path: Path to the configuration file.
        """
        self._load_config_with_status(config_path)

    def _load_config_with_status(self, config_path: str) -> bool:
        """
        Load and validate zone configuration, returning success status.

        Args:
            config_path: Path to the configuration file.

        Returns:
            True if loading and validation succeeded, False otherwise.
        """
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Zone config file not found: {config_path}")
            return False

        try:
            data = self._read_config_file(path)
        except Exception as e:
            logger.error(f"Failed to read zone config file '{config_path}': {e}")
            return False

        if data is None:
            logger.error(f"Zone config file '{config_path}' is empty or invalid")
            return False

        # Validate and load zones
        if not self._validate_and_load(data):
            logger.error(
                f"Zone config validation failed for '{config_path}'. "
                f"Rejecting entire file."
            )
            self._zones = {}
            return False

        return True

    def _read_config_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Read a YAML or JSON configuration file.

        Args:
            path: Path to the file.

        Returns:
            Parsed dictionary, or None on failure.
        """
        suffix = path.suffix.lower()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                logger.error(
                    "PyYAML is not installed. Cannot read YAML zone config. "
                    "Install with: pip install pyyaml"
                )
                return None
            return yaml.safe_load(content)
        elif suffix == ".json":
            return json.loads(content)
        else:
            # Try YAML first, then JSON
            if YAML_AVAILABLE:
                try:
                    return yaml.safe_load(content)
                except Exception:
                    pass
            try:
                return json.loads(content)
            except Exception:
                logger.error(
                    f"Could not parse zone config file '{path}' as YAML or JSON"
                )
                return None

    def _validate_and_load(self, data: Dict[str, Any]) -> bool:
        """
        Validate zone data structure and load into internal storage.

        Performs full validation before loading any zones. If any zone
        definition is invalid, the entire file is rejected.

        Args:
            data: Parsed configuration dictionary.

        Returns:
            True if all zones are valid and loaded, False otherwise.
        """
        # Extract zones section
        zones_data = data.get("zones")
        if zones_data is None:
            logger.error("Zone config missing 'zones' top-level key")
            return False

        if not isinstance(zones_data, dict):
            logger.error(
                f"Zone config 'zones' must be a dict, got {type(zones_data).__name__}"
            )
            return False

        # Validate all zones first before committing any
        validated_zones: Dict[str, List[ZonePolygon]] = {}

        for camera_id, zone_list in zones_data.items():
            if not isinstance(camera_id, str) or not camera_id.strip():
                logger.error(f"Invalid camera_id: '{camera_id}'")
                return False

            if not isinstance(zone_list, list):
                logger.error(
                    f"Zones for camera '{camera_id}' must be a list, "
                    f"got {type(zone_list).__name__}"
                )
                return False

            camera_zones: List[ZonePolygon] = []

            for idx, zone_def in enumerate(zone_list):
                if not isinstance(zone_def, dict):
                    logger.error(
                        f"Zone {idx} for camera '{camera_id}' must be a dict, "
                        f"got {type(zone_def).__name__}"
                    )
                    return False

                # Validate zone_type
                zone_type = zone_def.get("zone_type")
                if zone_type not in VALID_ZONE_TYPES:
                    logger.error(
                        f"Zone {idx} for camera '{camera_id}' has invalid zone_type "
                        f"'{zone_type}'. Must be one of {VALID_ZONE_TYPES}"
                    )
                    return False

                # Validate vertices
                vertices_raw = zone_def.get("vertices")
                if not isinstance(vertices_raw, list):
                    logger.error(
                        f"Zone {idx} for camera '{camera_id}' missing or invalid 'vertices'"
                    )
                    return False

                if len(vertices_raw) < 3:
                    logger.error(
                        f"Zone {idx} for camera '{camera_id}' has fewer than 3 vertices "
                        f"(got {len(vertices_raw)})"
                    )
                    return False

                # Validate each vertex coordinate
                vertices: List[Tuple[float, float]] = []
                for v_idx, vertex in enumerate(vertices_raw):
                    if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
                        logger.error(
                            f"Zone {idx}, vertex {v_idx} for camera '{camera_id}' "
                            f"must be a [x, y] pair, got {vertex}"
                        )
                        return False

                    try:
                        x = float(vertex[0])
                        y = float(vertex[1])
                    except (TypeError, ValueError):
                        logger.error(
                            f"Zone {idx}, vertex {v_idx} for camera '{camera_id}' "
                            f"has non-numeric coordinates: {vertex}"
                        )
                        return False

                    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                        logger.error(
                            f"Zone {idx}, vertex {v_idx} for camera '{camera_id}' "
                            f"coordinates ({x}, {y}) outside [0.0, 1.0] range"
                        )
                        return False

                    vertices.append((x, y))

                # Create ZonePolygon (validation already passed)
                try:
                    zone_polygon = ZonePolygon(
                        vertices=vertices,
                        zone_type=zone_type,
                        camera_id=camera_id,
                    )
                    camera_zones.append(zone_polygon)
                except ValueError as e:
                    logger.error(
                        f"Zone {idx} for camera '{camera_id}' failed validation: {e}"
                    )
                    return False

            validated_zones[camera_id] = camera_zones

        # All validation passed — commit zones
        self._zones = validated_zones
        logger.info(
            f"Loaded {sum(len(z) for z in self._zones.values())} zones "
            f"for {len(self._zones)} cameras from config"
        )
        return True

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, vertices: List[Tuple[float, float]]
    ) -> bool:
        """
        Determine if a point is inside a polygon using ray-casting algorithm.

        Casts a ray from the point to the right (+x direction) and counts
        how many polygon edges it crosses. An odd number of crossings means
        the point is inside.

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.
            vertices: List of (x, y) vertex coordinates defining the polygon.

        Returns:
            True if the point is inside the polygon, False otherwise.
        """
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            # Check if the ray crosses edge (i, j)
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            ):
                inside = not inside

            j = i

        return inside
