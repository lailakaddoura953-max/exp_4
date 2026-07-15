"""
Unit tests for ZoneMap validation and zone detection logic.

Property 18: Zone map validation
- Invalid polygons (<3 vertices) are rejected, previous definitions retained
- Out-of-range coordinates rejected
- Unknown zone types rejected
- Default no-people behavior when no config exists
- Runtime reload with valid and invalid configs

Validates: Requirements 10.6

Visual output: zone_map_overlay.png showing polygons on a sample frame.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Add tests/ directory to path for visual_helpers import
sys.path.insert(0, str(Path(__file__).parent.parent))

from hazard_detection.zone_map import ZoneMap
from hazard_detection.models import BBox, Detection, ZonePolygon
from visual_helpers import plot_annotated_frame


# ---------------------------------------------------------------------------
# Helper: write a temporary zone config file (JSON)
# ---------------------------------------------------------------------------

def _write_zone_config(config_data: dict, suffix: str = ".json") -> str:
    """Write zone config dict to a temporary file, return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    json.dump(config_data, tmp)
    tmp.close()
    return tmp.name


def _write_yaml_zone_config(config_data: dict) -> str:
    """Write zone config as YAML to a temporary file, return its path."""
    import yaml

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(config_data, tmp)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Valid config fixture
# ---------------------------------------------------------------------------

VALID_ZONE_CONFIG = {
    "zones": {
        "cam_01": [
            {
                "zone_type": "no_people",
                "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
            },
            {
                "zone_type": "operation",
                "vertices": [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5]],
            },
            {
                "zone_type": "dropoff",
                "vertices": [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]],
            },
        ]
    }
}


# ===========================================================================
# Test: Invalid polygons (<3 vertices) are rejected, previous defs retained
# ===========================================================================

class TestInvalidPolygonsRejected:
    """Property 18: Polygons with fewer than 3 vertices must be rejected."""

    def test_polygon_with_two_vertices_rejected(self, tmp_path):
        """A zone with only 2 vertices should cause the entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "no_people",
                        "vertices": [[0.0, 0.0], [1.0, 1.0]],  # Only 2
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            # Should have rejected - no zones loaded
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_polygon_with_one_vertex_rejected(self, tmp_path):
        """A zone with only 1 vertex should cause the entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "operation",
                        "vertices": [[0.5, 0.5]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_polygon_with_zero_vertices_rejected(self, tmp_path):
        """A zone with empty vertices list should cause rejection."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "no_people",
                        "vertices": [],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_previous_definitions_retained_on_reload_failure(self, tmp_path):
        """When reload fails due to invalid polygons, previous definitions remain."""
        # First load a valid config
        valid_path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=valid_path)
            assert zone_map.total_zones == 3

            # Now overwrite the file with invalid config (< 3 vertices)
            invalid_config = {
                "zones": {
                    "cam_01": [
                        {
                            "zone_type": "no_people",
                            "vertices": [[0.0, 0.0], [1.0, 1.0]],
                        }
                    ]
                }
            }
            with open(valid_path, "w") as f:
                json.dump(invalid_config, f)

            # Reload should fail, retain previous valid zones
            result = zone_map.reload()
            assert result is False
            assert zone_map.total_zones == 3
        finally:
            os.unlink(valid_path)


# ===========================================================================
# Test: Out-of-range coordinates rejected
# ===========================================================================

class TestOutOfRangeCoordinatesRejected:
    """Property 18: Coordinates outside [0.0, 1.0] must be rejected."""

    def test_coordinate_above_one_rejected(self, tmp_path):
        """Vertex coordinate > 1.0 causes entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "no_people",
                        "vertices": [[0.0, 0.0], [1.5, 0.0], [1.5, 1.0], [0.0, 1.0]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_negative_coordinate_rejected(self, tmp_path):
        """Vertex coordinate < 0.0 causes entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "operation",
                        "vertices": [[-0.1, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_y_coordinate_above_one_rejected(self, tmp_path):
        """Y coordinate > 1.0 causes entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "dropoff",
                        "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.2], [0.0, 1.0]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_previous_definitions_retained_on_coord_failure(self, tmp_path):
        """When reload fails due to out-of-range coords, previous defs retained."""
        valid_path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=valid_path)
            assert zone_map.total_zones == 3

            # Overwrite with invalid coordinates
            invalid_config = {
                "zones": {
                    "cam_01": [
                        {
                            "zone_type": "no_people",
                            "vertices": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                        }
                    ]
                }
            }
            with open(valid_path, "w") as f:
                json.dump(invalid_config, f)

            result = zone_map.reload()
            assert result is False
            assert zone_map.total_zones == 3
        finally:
            os.unlink(valid_path)


# ===========================================================================
# Test: Unknown zone types rejected
# ===========================================================================

class TestUnknownZoneTypesRejected:
    """Property 18: Unknown zone types must cause file rejection."""

    def test_unknown_zone_type_rejected(self, tmp_path):
        """An unknown zone_type causes the entire file to be rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "restricted_area",  # Not valid
                        "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_empty_zone_type_rejected(self, tmp_path):
        """Empty string zone_type causes rejection."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "",
                        "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_none_zone_type_rejected(self, tmp_path):
        """None zone_type causes rejection."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": None,
                        "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)

    def test_valid_file_with_one_bad_zone_type_rejects_all(self, tmp_path):
        """If one zone in a multi-zone file has bad type, entire file rejected."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "no_people",
                        "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                    },
                    {
                        "zone_type": "invalid_type",
                        "vertices": [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5]],
                    },
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 0
        finally:
            os.unlink(path)


# ===========================================================================
# Test: Default no-people behavior when no config exists
# ===========================================================================

class TestDefaultNoPeopleBehavior:
    """When no config exists for a camera, entire FOV is no-people zone."""

    def test_no_config_path_defaults_to_no_people(self):
        """ZoneMap with no config_path defaults all lookups to no_people."""
        zone_map = ZoneMap(config_path=None)
        assert zone_map.get_zone_type("cam_01", (0.5, 0.5)) == "no_people"
        assert zone_map.get_zone_type("cam_01", (0.0, 0.0)) == "no_people"
        assert zone_map.get_zone_type("cam_01", (1.0, 1.0)) == "no_people"

    def test_unknown_camera_defaults_to_no_people(self):
        """Camera not in config defaults to no_people for any point."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            # cam_99 not defined in config
            assert zone_map.get_zone_type("cam_99", (0.5, 0.5)) == "no_people"
        finally:
            os.unlink(path)

    def test_point_outside_all_zones_defaults_to_no_people(self):
        """A point that falls outside all defined polygons returns no_people."""
        # Config defines zones that don't cover the full FOV
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "operation",
                        "vertices": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            # Point at (0.9, 0.9) is outside the small operation zone
            assert zone_map.get_zone_type("cam_01", (0.9, 0.9)) == "no_people"
        finally:
            os.unlink(path)


# ===========================================================================
# Test: Runtime reload with valid and invalid configs
# ===========================================================================

class TestRuntimeReload:
    """Test reload() with valid configs succeeds and invalid configs retain previous."""

    def test_reload_with_valid_config_succeeds(self):
        """Reload with a valid config file returns True and updates zones."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 3

            # Update config to add a new camera
            updated_config = {
                "zones": {
                    "cam_01": [
                        {
                            "zone_type": "no_people",
                            "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        }
                    ],
                    "cam_02": [
                        {
                            "zone_type": "operation",
                            "vertices": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                        }
                    ],
                }
            }
            with open(path, "w") as f:
                json.dump(updated_config, f)

            result = zone_map.reload()
            assert result is True
            assert zone_map.total_zones == 2
            assert "cam_02" in zone_map.camera_ids
        finally:
            os.unlink(path)

    def test_reload_with_invalid_config_retains_previous(self):
        """Reload with invalid config returns False, retains previous zones."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 3

            # Overwrite with invalid config (unknown zone type)
            invalid_config = {
                "zones": {
                    "cam_01": [
                        {
                            "zone_type": "forbidden_zone",
                            "vertices": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
                        }
                    ]
                }
            }
            with open(path, "w") as f:
                json.dump(invalid_config, f)

            result = zone_map.reload()
            assert result is False
            assert zone_map.total_zones == 3  # Previous retained
        finally:
            os.unlink(path)

    def test_reload_with_missing_file_retains_previous(self):
        """Reload when file is deleted retains previous zones."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 3
        finally:
            os.unlink(path)

        # File no longer exists
        result = zone_map.reload()
        assert result is False
        assert zone_map.total_zones == 3

    def test_reload_with_malformed_json_retains_previous(self):
        """Reload with syntactically invalid JSON retains previous zones."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 3

            # Write malformed JSON
            with open(path, "w") as f:
                f.write("{invalid json content!!!}")

            result = zone_map.reload()
            assert result is False
            assert zone_map.total_zones == 3
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_reload_no_config_path_returns_false(self):
        """Reload when ZoneMap has no config_path returns False."""
        zone_map = ZoneMap(config_path=None)
        result = zone_map.reload()
        assert result is False


# ===========================================================================
# Test: Valid config loads correctly
# ===========================================================================

class TestValidConfigLoads:
    """Ensure valid configurations are loaded and zone lookups work."""

    def test_valid_config_loads_all_zones(self):
        """A fully valid config loads all zones correctly."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 3
            assert "cam_01" in zone_map.camera_ids
        finally:
            os.unlink(path)

    def test_point_in_no_people_zone(self):
        """Point within no_people polygon returns 'no_people'."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            # Left half is no_people: (0,0)→(0.5,0)→(0.5,1)→(0,1)
            result = zone_map.get_zone_type("cam_01", (0.25, 0.5))
            assert result == "no_people"
        finally:
            os.unlink(path)

    def test_point_in_operation_zone(self):
        """Point within operation polygon returns 'operation'."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            # Top-right is operation: (0.5,0)→(1.0,0)→(1.0,0.5)→(0.5,0.5)
            result = zone_map.get_zone_type("cam_01", (0.75, 0.25))
            assert result == "operation"
        finally:
            os.unlink(path)

    def test_point_in_dropoff_zone(self):
        """Point within dropoff polygon returns 'dropoff'."""
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            # Bottom-right is dropoff: (0.5,0.5)→(1.0,0.5)→(1.0,1.0)→(0.5,1.0)
            result = zone_map.get_zone_type("cam_01", (0.75, 0.75))
            assert result == "dropoff"
        finally:
            os.unlink(path)

    def test_three_vertex_polygon_accepted(self):
        """A triangle (exactly 3 vertices) is a valid polygon."""
        config = {
            "zones": {
                "cam_01": [
                    {
                        "zone_type": "no_people",
                        "vertices": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                    }
                ]
            }
        }
        path = _write_zone_config(config)
        try:
            zone_map = ZoneMap(config_path=path)
            assert zone_map.total_zones == 1
        finally:
            os.unlink(path)


# ===========================================================================
# Test: Initialize from detections
# ===========================================================================

class TestInitializeFromDetections:
    """Test YOLO-based zone initialization."""

    def test_initialize_from_valid_detections(self):
        """Valid zone detections above threshold create zone polygons."""
        zone_map = ZoneMap(config_path=None)

        detections = [
            Detection(
                bbox=BBox(x_center=0.25, y_center=0.25, width=0.4, height=0.4),
                class_label="Yard - No People",
                confidence=0.9,
            ),
            Detection(
                bbox=BBox(x_center=0.75, y_center=0.75, width=0.4, height=0.4),
                class_label="Yard - Operation Zone",
                confidence=0.85,
            ),
        ]

        zone_map.initialize_from_detections("cam_01", detections, confidence_threshold=0.5)
        assert len(zone_map.get_zones_for_camera("cam_01")) == 2

    def test_below_threshold_detections_ignored(self):
        """Detections below confidence threshold are not used for zone init."""
        zone_map = ZoneMap(config_path=None)

        detections = [
            Detection(
                bbox=BBox(x_center=0.5, y_center=0.5, width=0.5, height=0.5),
                class_label="Yard - No People",
                confidence=0.3,  # Below threshold
            ),
        ]

        zone_map.initialize_from_detections("cam_01", detections, confidence_threshold=0.5)
        assert len(zone_map.get_zones_for_camera("cam_01")) == 0

    def test_non_zone_class_detections_ignored(self):
        """Detections that aren't zone classes are skipped."""
        zone_map = ZoneMap(config_path=None)

        detections = [
            Detection(
                bbox=BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.3),
                class_label="Human",
                confidence=0.9,
            ),
        ]

        zone_map.initialize_from_detections("cam_01", detections, confidence_threshold=0.5)
        assert len(zone_map.get_zones_for_camera("cam_01")) == 0


# ===========================================================================
# Visual output: Zone overlay visualization
# ===========================================================================

class TestZoneMapVisualization:
    """Generate zone overlay visualization as a PNG diagnostic artifact."""

    def test_generate_zone_overlay_png(self, mock_frame, output_dir):
        """
        Generate zone_map_overlay.png showing polygons on a sample frame.

        Validates: Requirements 10.6
        """
        path = _write_zone_config(VALID_ZONE_CONFIG)
        try:
            zone_map = ZoneMap(config_path=path)
            zones = zone_map.get_zones_for_camera("cam_01")

            # Prepare zone data for plotting
            zone_dicts = [
                {"vertices": z.vertices, "zone_type": z.zone_type}
                for z in zones
            ]

            # Create sample detections to overlay on the frame
            sample_detections = [
                {
                    "bbox": {"x_center": 0.25, "y_center": 0.5, "width": 0.08, "height": 0.25},
                    "class_label": "Human",
                    "confidence": 0.85,
                },
                {
                    "bbox": {"x_center": 0.75, "y_center": 0.25, "width": 0.2, "height": 0.12},
                    "class_label": "Container - Stacked",
                    "confidence": 0.91,
                },
                {
                    "bbox": {"x_center": 0.75, "y_center": 0.75, "width": 0.15, "height": 0.1},
                    "class_label": "Truck - With Container",
                    "confidence": 0.78,
                },
            ]

            output_path = output_dir / "zone_map_overlay.png"
            plot_annotated_frame(
                frame=mock_frame,
                detections=sample_detections,
                zones=zone_dicts,
                output_path=output_path,
                title="Zone Map Overlay - cam_01",
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        finally:
            os.unlink(path)
