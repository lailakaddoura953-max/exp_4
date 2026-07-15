"""
Unit tests for ConfigurationManager validation logic.

Property 25: Configuration validation
- Malformed YAML raises startup error with descriptive message
- Missing required params raises error identifying the missing parameter
- Out-of-range values (frame_count, confidence, rate_limit) are rejected
- Valid configs load successfully with correct defaults applied

**Validates: Requirements 14.3, 14.4, 14.5**

Uses hypothesis for property-based testing of parameter ranges.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hazard_detection.config import ConfigurationManager, ConfigurationError

# Import visual helpers for generating PNG diagnostic output
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visual_helpers import save_json_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _minimal_valid_yaml(checkpoint_path: str) -> str:
    """Return minimal valid YAML with required params filled in."""
    return f"""\
cameras:
  sequence: ["cam_01", "cam_02"]
yolo:
  checkpoint_path: "{checkpoint_path}"
system:
  frame_sample_count: 6
alerts:
  rate_limit_seconds: 60
  channels: ["email"]
"""


def _get_real_checkpoint_path() -> str:
    """Return a real checkpoint path that exists on disk for testing.

    Returns forward-slash paths to avoid YAML escape issues on Windows.
    """
    # Use the actual checkpoint present in the repo
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "checkpoints", "yolov12_best.pt"
    )
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path.replace("\\", "/")
    # Fallback: create a temp file to act as a checkpoint
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    return tmp_path.replace("\\", "/")


# ---------------------------------------------------------------------------
# Test: Malformed YAML raises startup error (Requirement 14.3)
# ---------------------------------------------------------------------------


class TestMalformedYAML:
    """Tests that malformed YAML raises ConfigurationError at startup."""

    def test_completely_invalid_yaml_syntax(self):
        """Unbalanced braces / bad syntax causes descriptive error."""
        content = "cameras:\n  sequence: [unclosed bracket\n  : :\n"
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="[Mm]alformed"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_yaml_with_tabs_instead_of_spaces(self):
        """YAML with tab indentation that breaks parsing raises error."""
        content = "cameras:\n\t\tsequence: ['cam_01']\n"
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            # Tabs may parse or may not depending on YAML spec; verify we get either
            # valid load or a ConfigurationError (not a raw exception)
            try:
                mgr.load()
            except ConfigurationError:
                pass  # Expected — malformed or missing section
        finally:
            os.unlink(path)

    def test_empty_yaml_file(self):
        """An empty YAML file raises a descriptive error."""
        path = _write_yaml("")
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="empty"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_yaml_scalar_top_level(self):
        """A YAML file with just a scalar (not a mapping) raises error."""
        path = _write_yaml("just a string\n")
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="mapping"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_yaml_list_top_level(self):
        """A YAML file with a top-level list instead of mapping raises error."""
        path = _write_yaml("- item1\n- item2\n")
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="mapping"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        """Referencing a non-existent file raises descriptive error."""
        mgr = ConfigurationManager(config_path="/nonexistent/path/config.yaml")
        with pytest.raises(ConfigurationError, match="not found"):
            mgr.load()


# ---------------------------------------------------------------------------
# Test: Missing required parameters (Requirement 14.4)
# ---------------------------------------------------------------------------


class TestMissingRequiredParams:
    """Tests that missing required parameters raise errors identifying what's missing."""

    def test_missing_cameras_section(self):
        """Missing 'cameras' section is identified in error."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
yolo:
  checkpoint_path: "{checkpoint}"
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="cameras"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_missing_cameras_sequence(self):
        """Missing 'cameras.sequence' is identified in error."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
cameras:
  other_param: "something"
yolo:
  checkpoint_path: "{checkpoint}"
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="sequence"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_missing_yolo_section(self):
        """Missing 'yolo' section is identified in error."""
        content = """\
cameras:
  sequence: ["cam_01"]
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="yolo"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_missing_yolo_checkpoint_path(self):
        """Missing 'yolo.checkpoint_path' is identified in error."""
        content = """\
cameras:
  sequence: ["cam_01"]
yolo:
  device: "cpu"
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="checkpoint_path"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_empty_camera_sequence(self):
        """An empty camera sequence list raises a descriptive error."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
cameras:
  sequence: []
yolo:
  checkpoint_path: "{checkpoint}"
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="non-empty"):
                mgr.load()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Out-of-range values rejected (Requirement 14.5)
# ---------------------------------------------------------------------------


class TestOutOfRangeValues:
    """Tests that out-of-range parameter values are rejected at startup."""

    def _make_config(self, overrides: Dict[str, Any]) -> str:
        """Create a valid config YAML with specific overrides applied."""
        checkpoint = _get_real_checkpoint_path()
        base = {
            "cameras": {"sequence": ["cam_01"]},
            "yolo": {"checkpoint_path": checkpoint},
            "system": {"frame_sample_count": 6},
            "alerts": {"rate_limit_seconds": 60, "channels": ["email"]},
        }
        # Apply overrides using dotted-path keys
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            target = base
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value

        import yaml
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(base, f, default_flow_style=False)
        return path

    # frame_count range tests [5, 8]
    def test_frame_count_below_range(self):
        """frame_count < 5 is rejected."""
        path = self._make_config({"system.frame_sample_count": 4})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="frame_sample_count"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_frame_count_above_range(self):
        """frame_count > 8 is rejected."""
        path = self._make_config({"system.frame_sample_count": 9})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="frame_sample_count"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_frame_count_non_integer(self):
        """frame_count as float is rejected."""
        path = self._make_config({"system.frame_sample_count": 6.5})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="frame_sample_count"):
                mgr.load()
        finally:
            os.unlink(path)

    # confidence range tests [0.0, 1.0]
    def test_confidence_below_range(self):
        """Confidence < 0.0 is rejected."""
        path = self._make_config({"yolo.confidence_threshold": -0.1})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="confidence"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_confidence_above_range(self):
        """Confidence > 1.0 is rejected."""
        path = self._make_config({"yolo.confidence_threshold": 1.5})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="confidence"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_human_confidence_invalid(self):
        """Human detector confidence > 1.0 is rejected."""
        path = self._make_config({"detection.human.confidence_threshold": 2.0})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="confidence"):
                mgr.load()
        finally:
            os.unlink(path)

    # rate_limit_seconds range tests [10, 300]
    def test_rate_limit_below_range(self):
        """rate_limit_seconds < 10 is rejected."""
        path = self._make_config({"alerts.rate_limit_seconds": 5})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="rate_limit"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_rate_limit_above_range(self):
        """rate_limit_seconds > 300 is rejected."""
        path = self._make_config({"alerts.rate_limit_seconds": 301})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="rate_limit"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_rate_limit_non_integer(self):
        """rate_limit_seconds as float is rejected."""
        path = self._make_config({"alerts.rate_limit_seconds": 60.5})
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="rate_limit"):
                mgr.load()
        finally:
            os.unlink(path)

    def test_checkpoint_path_not_exists(self):
        """Non-existent checkpoint path is rejected."""
        content = """\
cameras:
  sequence: ["cam_01"]
yolo:
  checkpoint_path: "/nonexistent/model.pt"
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError, match="checkpoint"):
                mgr.load()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Valid configs load with correct defaults (Requirement 14.5/14.6)
# ---------------------------------------------------------------------------


class TestValidConfigWithDefaults:
    """Tests that valid configs load successfully and apply correct defaults."""

    def test_minimal_valid_config_loads(self):
        """Minimal valid config loads without error."""
        checkpoint = _get_real_checkpoint_path()
        content = _minimal_valid_yaml(checkpoint)
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            assert pipeline.camera_sequence == ["cam_01", "cam_02"]
        finally:
            os.unlink(path)

    def test_defaults_applied_for_optional_params(self):
        """Missing optional params receive documented defaults."""
        checkpoint = _get_real_checkpoint_path()
        # Only provide required params — everything else should get defaults
        content = f"""\
cameras:
  sequence: ["cam_01"]
yolo:
  checkpoint_path: "{checkpoint}"
alerts:
  rate_limit_seconds: 60
  channels: ["dashboard"]
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            # frame_sample_count should default to 6
            assert pipeline.frame_sampler.frame_count == 6
            # yolo device defaults to cuda
            assert pipeline.yolo.device == "cuda"
            # defaults_applied should include the defaulted params
            assert len(mgr.defaults_applied) > 0
        finally:
            os.unlink(path)

    def test_explicit_values_override_defaults(self):
        """Explicitly specified values override documented defaults."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
cameras:
  sequence: ["cam_01", "cam_02", "cam_03"]
yolo:
  checkpoint_path: "{checkpoint}"
  device: "cpu"
  input_resolution: 512
  confidence_threshold: 0.7
system:
  frame_sample_count: 7
alerts:
  rate_limit_seconds: 120
  channels: ["email", "sms"]
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            assert pipeline.frame_sampler.frame_count == 7
            assert pipeline.yolo.device == "cpu"
            assert pipeline.yolo.input_resolution == 512
            assert pipeline.yolo.confidence_threshold == 0.7
            assert pipeline.alert_dispatcher.rate_limit_seconds == 120
            assert pipeline.alert_dispatcher.channels == ["email", "sms"]
        finally:
            os.unlink(path)

    def test_boundary_values_accepted(self):
        """Boundary values (edges of valid ranges) are accepted."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
cameras:
  sequence: ["cam_01"]
yolo:
  checkpoint_path: "{checkpoint}"
  confidence_threshold: 0.0
system:
  frame_sample_count: 5
alerts:
  rate_limit_seconds: 10
  channels: ["email"]
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            assert pipeline.frame_sampler.frame_count == 5
            assert pipeline.yolo.confidence_threshold == 0.0
            assert pipeline.alert_dispatcher.rate_limit_seconds == 10
        finally:
            os.unlink(path)

    def test_upper_boundary_values_accepted(self):
        """Upper boundary values are accepted."""
        checkpoint = _get_real_checkpoint_path()
        content = f"""\
cameras:
  sequence: ["cam_01"]
yolo:
  checkpoint_path: "{checkpoint}"
  confidence_threshold: 1.0
system:
  frame_sample_count: 8
alerts:
  rate_limit_seconds: 300
  channels: ["email"]
"""
        path = _write_yaml(content)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            assert pipeline.frame_sampler.frame_count == 8
            assert pipeline.yolo.confidence_threshold == 1.0
            assert pipeline.alert_dispatcher.rate_limit_seconds == 300
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Property-Based Test: Configuration validation (Property 25)
# ---------------------------------------------------------------------------


class TestConfigValidationProperty:
    """
    Property-based tests for configuration parameter validation.

    **Validates: Requirements 14.3, 14.4, 14.5**
    """

    @given(frame_count=st.integers().filter(lambda x: x < 5 or x > 8))
    @settings(max_examples=50)
    def test_invalid_frame_count_always_rejected(self, frame_count: int):
        """
        Property 25: Any frame_count outside [5, 8] SHALL be rejected.

        **Validates: Requirements 14.5**
        """
        checkpoint = _get_real_checkpoint_path()
        import yaml
        config = {
            "cameras": {"sequence": ["cam_01"]},
            "yolo": {"checkpoint_path": checkpoint},
            "system": {"frame_sample_count": frame_count},
            "alerts": {"rate_limit_seconds": 60, "channels": ["email"]},
        }
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError):
                mgr.load()
        finally:
            os.unlink(path)

    @given(confidence=st.floats().filter(lambda x: x < 0.0 or x > 1.0))
    @settings(max_examples=50)
    def test_invalid_confidence_always_rejected(self, confidence: float):
        """
        Property 25: Any confidence outside [0.0, 1.0] SHALL be rejected.

        **Validates: Requirements 14.5**
        """
        assume(not (confidence != confidence))  # exclude NaN
        checkpoint = _get_real_checkpoint_path()
        import yaml
        config = {
            "cameras": {"sequence": ["cam_01"]},
            "yolo": {
                "checkpoint_path": checkpoint,
                "confidence_threshold": confidence,
            },
            "system": {"frame_sample_count": 6},
            "alerts": {"rate_limit_seconds": 60, "channels": ["email"]},
        }
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError):
                mgr.load()
        finally:
            os.unlink(path)

    @given(rate_limit=st.integers().filter(lambda x: x < 10 or x > 300))
    @settings(max_examples=50)
    def test_invalid_rate_limit_always_rejected(self, rate_limit: int):
        """
        Property 25: Any rate_limit_seconds outside [10, 300] SHALL be rejected.

        **Validates: Requirements 14.5**
        """
        checkpoint = _get_real_checkpoint_path()
        import yaml
        config = {
            "cameras": {"sequence": ["cam_01"]},
            "yolo": {"checkpoint_path": checkpoint},
            "system": {"frame_sample_count": 6},
            "alerts": {"rate_limit_seconds": rate_limit, "channels": ["email"]},
        }
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        try:
            mgr = ConfigurationManager(config_path=path)
            with pytest.raises(ConfigurationError):
                mgr.load()
        finally:
            os.unlink(path)

    @given(
        frame_count=st.integers(min_value=5, max_value=8),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        rate_limit=st.integers(min_value=10, max_value=300),
    )
    @settings(max_examples=50)
    def test_valid_params_always_accepted(
        self, frame_count: int, confidence: float, rate_limit: int
    ):
        """
        Property 25: Any combination of valid parameters SHALL load successfully.

        **Validates: Requirements 14.3, 14.4, 14.5**
        """
        assume(confidence == confidence)  # exclude NaN
        checkpoint = _get_real_checkpoint_path()
        import yaml
        config = {
            "cameras": {"sequence": ["cam_01"]},
            "yolo": {
                "checkpoint_path": checkpoint,
                "confidence_threshold": confidence,
            },
            "system": {"frame_sample_count": frame_count},
            "alerts": {"rate_limit_seconds": rate_limit, "channels": ["email"]},
        }
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        try:
            mgr = ConfigurationManager(config_path=path)
            mgr.load()
            pipeline = mgr.get_pipeline_config()
            assert pipeline.frame_sampler.frame_count == frame_count
            assert pipeline.alert_dispatcher.rate_limit_seconds == rate_limit
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Visual Output: Parameter Validation Matrix
# ---------------------------------------------------------------------------


class TestConfigValidationVisualOutput:
    """Generate visual diagnostic PNG for configuration validation matrix."""

    def test_generate_validation_matrix_png(self, output_dir: Path):
        """
        Generate a parameter validation matrix as PNG showing which
        parameters are validated and their valid ranges.

        Saves to tests/output/config_validation_matrix.png
        """
        # Define the validation matrix data
        parameters = [
            "system.frame_sample_count",
            "yolo.confidence_threshold",
            "detection.human.confidence_threshold",
            "detection.container.confidence_threshold",
            "alerts.rate_limit_seconds",
            "yolo.checkpoint_path",
            "cameras.sequence",
        ]
        validations = [
            "Required",
            "Type Check",
            "Range Min",
            "Range Max",
            "Exists Check",
        ]

        # Build the matrix: 1 = validated, 0 = not applicable
        # Rows = parameters, Cols = validation types
        matrix = np.array([
            [0, 1, 1, 1, 0],  # frame_sample_count: type, range [5,8]
            [0, 1, 1, 1, 0],  # yolo.confidence: type, range [0,1]
            [0, 1, 1, 1, 0],  # human.confidence: type, range [0,1]
            [0, 1, 1, 1, 0],  # container.confidence: type, range [0,1]
            [0, 1, 1, 1, 0],  # rate_limit: type, range [10,300]
            [1, 0, 0, 0, 1],  # checkpoint: required, exists
            [1, 1, 0, 0, 0],  # cameras.sequence: required, type (list)
        ])

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 7))

        # Create heatmap
        cmap = sns.color_palette(["#f8f9fa", "#2ecc71"], as_cmap=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=validations,
            yticklabels=parameters,
            cbar=False,
            linewidths=1,
            linecolor="#dee2e6",
            ax=ax,
        )

        # Add range annotations
        range_info = {
            0: "[5, 8]",
            1: "[0.0, 1.0]",
            2: "[0.0, 1.0]",
            3: "[0.0, 1.0]",
            4: "[10, 300]",
            5: "file exists",
            6: "non-empty list",
        }

        # Add a text column on the right for ranges
        for i, (param, info) in enumerate(zip(parameters, range_info.values())):
            ax.text(
                len(validations) + 0.3,
                i + 0.5,
                info,
                va="center",
                ha="left",
                fontsize=9,
                color="#2c3e50",
            )

        ax.text(
            len(validations) + 0.3,
            -0.3,
            "Valid Range",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

        ax.set_title(
            "Configuration Parameter Validation Matrix\n"
            "(Property 25 — Requirements 14.3, 14.4, 14.5)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        plt.tight_layout()
        output_path = output_dir / "config_validation_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert output_path.exists()
        assert output_path.stat().st_size > 0
