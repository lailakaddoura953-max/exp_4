"""
Configuration Manager for the Hazard Detection System.

This module handles loading, validating, and providing access to system
configuration from YAML files. It enforces startup-time validation of all
parameters and applies documented defaults for optional parameters.

Requirements covered:
- 14.1: Load configuration from YAML file via CLI argument or default path
- 14.2: Support configuring frame sample count, confidence thresholds,
         model checkpoint paths, camera sequence, alert settings, zone maps
- 14.3: Fail at startup for missing/malformed YAML
- 14.4: Fail at startup for missing required parameters
- 14.5: Validate ranges (frame_count [5,8], confidence [0.0,1.0],
         checkpoint paths exist, rate_limit_seconds [10,300])
- 14.6: Apply documented defaults for optional parameters and log which are in use
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hazard_detection.models import (
    AlertDispatcherConfig,
    CameraSwitcherConfig,
    ContainerAnalyzerConfig,
    FrameSamplerConfig,
    HumanDetectorConfig,
    PipelineConfig,
    YOLOConfig,
)

logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = "config/hazard_detection.yaml"

# Documented defaults for optional parameters
DOCUMENTED_DEFAULTS = {
    "system.frame_sample_count": 6,
    "system.per_camera_timeout_seconds": 30,
    "yolo.device": "cuda",
    "yolo.input_resolution": 640,
    "yolo.confidence_threshold": 0.5,
    "detection.human.confidence_threshold": 0.5,
    "detection.container.confidence_threshold": 0.5,
    "detection.container.flipped_aspect_ratio_threshold": 1.5,
    "detection.container.safe_overlap_threshold": 0.3,
    "detection.container.ground_level_threshold": 0.4,
    "detection.container.motion_threshold": 0.7,
    "detection.container.iou_threshold": 0.5,
    "detection.orientation.confidence_threshold": 0.5,
    "alerts.rate_limit_seconds": 60,
    "alerts.channels": ["email", "dashboard"],
}


class ConfigurationError(Exception):
    """Raised when configuration validation fails at startup."""

    pass


class ConfigurationManager:
    """
    Manages system configuration loading, validation, and access.

    Loads configuration from a YAML file, validates all parameters against
    their documented constraints, applies defaults for optional parameters,
    and provides typed access to configuration values.

    Usage:
        config_manager = ConfigurationManager()
        config_manager.load()  # Loads from CLI arg or default path
        pipeline_config = config_manager.get_pipeline_config()
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigurationManager.

        Args:
            config_path: Optional explicit path to YAML config file.
                         If not provided, will check CLI arguments or use default path.
        """
        self._config_path: Optional[str] = config_path
        self._raw_config: Dict[str, Any] = {}
        self._pipeline_config: Optional[PipelineConfig] = None
        self._defaults_applied: List[str] = []

    @property
    def config_path(self) -> str:
        """Return the resolved configuration file path."""
        if self._config_path:
            return self._config_path
        return DEFAULT_CONFIG_PATH

    @property
    def raw_config(self) -> Dict[str, Any]:
        """Return the raw parsed YAML configuration dictionary."""
        return self._raw_config

    @property
    def defaults_applied(self) -> List[str]:
        """Return list of parameter paths where defaults were applied."""
        return self._defaults_applied

    def load(self, config_path: Optional[str] = None) -> "ConfigurationManager":
        """
        Load and validate configuration from YAML file.

        Resolves the config path from (in priority order):
        1. Explicit config_path argument
        2. Path provided at initialization
        3. CLI --config argument
        4. Default path (config/hazard_detection.yaml)

        Args:
            config_path: Optional override for the config file path.

        Returns:
            self for method chaining.

        Raises:
            ConfigurationError: If YAML file is missing, malformed, or contains
                               invalid parameter values.
        """
        # Resolve the configuration file path
        resolved_path = self._resolve_config_path(config_path)
        self._config_path = resolved_path

        # Load the YAML file
        self._raw_config = self._load_yaml(resolved_path)

        # Validate and build configuration
        self._validate_and_build()

        # Log defaults that were applied
        self._log_defaults()

        return self

    def get_pipeline_config(self) -> PipelineConfig:
        """
        Get the validated PipelineConfig.

        Returns:
            PipelineConfig with all validated component configurations.

        Raises:
            ConfigurationError: If configuration has not been loaded yet.
        """
        if self._pipeline_config is None:
            raise ConfigurationError(
                "Configuration has not been loaded. Call load() first."
            )
        return self._pipeline_config

    def get_raw_value(self, dotted_path: str, default: Any = None) -> Any:
        """
        Get a raw configuration value by dotted path notation.

        Args:
            dotted_path: Dot-separated path (e.g., 'system.frame_sample_count')
            default: Default value if path not found.

        Returns:
            The configuration value or default.
        """
        keys = dotted_path.split(".")
        current = self._raw_config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def _resolve_config_path(self, explicit_path: Optional[str] = None) -> str:
        """
        Resolve the configuration file path from available sources.

        Priority: explicit_path > self._config_path > CLI arg > default.

        Returns:
            Resolved file path string.

        Raises:
            ConfigurationError: If the resolved path does not exist.
        """
        if explicit_path:
            path = explicit_path
        elif self._config_path:
            path = self._config_path
        else:
            path = self._parse_cli_config_path() or DEFAULT_CONFIG_PATH

        # Check file exists
        if not os.path.isfile(path):
            raise ConfigurationError(
                f"Configuration file not found: '{path}'. "
                f"Specify a valid path via --config CLI argument or ensure "
                f"the default config exists at '{DEFAULT_CONFIG_PATH}'."
            )

        return path

    def _parse_cli_config_path(self) -> Optional[str]:
        """
        Parse --config argument from command line if present.

        Returns:
            Config file path from CLI, or None if not specified.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        args, _ = parser.parse_known_args()
        return args.config

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load and parse YAML configuration file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            ConfigurationError: If the file cannot be read or contains malformed YAML.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            raise ConfigurationError(
                f"Failed to read configuration file '{path}': {e}"
            )

        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Malformed YAML in configuration file '{path}': {e}"
            )

        if config is None:
            raise ConfigurationError(
                f"Configuration file '{path}' is empty or contains only comments."
            )

        if not isinstance(config, dict):
            raise ConfigurationError(
                f"Configuration file '{path}' must contain a YAML mapping (dictionary) "
                f"at the top level, got {type(config).__name__}."
            )

        return config

    def _validate_and_build(self) -> None:
        """
        Validate all configuration parameters and build the PipelineConfig.

        Validates required parameters are present, values are within allowed
        ranges, and checkpoint paths exist. Applies defaults for optional
        parameters that are missing.

        Raises:
            ConfigurationError: If any required parameter is missing or
                               any value is outside its allowed range.
        """
        self._defaults_applied = []

        # Validate required sections
        self._require_section("cameras", "cameras")
        self._require_param("cameras.sequence", "cameras", "sequence")
        self._require_section("yolo", "yolo")
        self._require_param("yolo.checkpoint_path", "yolo", "checkpoint_path")

        # Build component configurations
        frame_sampler_config = self._build_frame_sampler_config()
        yolo_config = self._build_yolo_config()
        human_detector_config = self._build_human_detector_config()
        container_analyzer_config = self._build_container_analyzer_config()
        alert_dispatcher_config = self._build_alert_dispatcher_config()
        camera_switcher_config = self._build_camera_switcher_config()

        # Build pipeline config
        camera_sequence = self._raw_config["cameras"]["sequence"]
        if not isinstance(camera_sequence, list) or len(camera_sequence) == 0:
            raise ConfigurationError(
                "Required parameter 'cameras.sequence' must be a non-empty list "
                "of camera identifiers."
            )

        per_camera_timeout = self._get_with_default(
            "system.per_camera_timeout_seconds",
            ["system", "per_camera_timeout_seconds"],
            DOCUMENTED_DEFAULTS["system.per_camera_timeout_seconds"],
        )

        try:
            self._pipeline_config = PipelineConfig(
                camera_sequence=camera_sequence,
                per_camera_timeout_seconds=per_camera_timeout,
                frame_sampler=frame_sampler_config,
                yolo=yolo_config,
                human_detector=human_detector_config,
                container_analyzer=container_analyzer_config,
                alert_dispatcher=alert_dispatcher_config,
                camera_switcher=camera_switcher_config,
            )
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Configuration validation error: {e}"
            )

    def _build_frame_sampler_config(self) -> FrameSamplerConfig:
        """Build and validate FrameSamplerConfig from raw config."""
        frame_count = self._get_with_default(
            "system.frame_sample_count",
            ["system", "frame_sample_count"],
            DOCUMENTED_DEFAULTS["system.frame_sample_count"],
        )

        # Validate frame_count range
        if not isinstance(frame_count, int):
            raise ConfigurationError(
                f"Parameter 'system.frame_sample_count' must be an integer, "
                f"got {type(frame_count).__name__} with value '{frame_count}'."
            )
        if not (5 <= frame_count <= 8):
            raise ConfigurationError(
                f"Parameter 'system.frame_sample_count' must be between 5 and 8 inclusive, "
                f"got {frame_count}. Allowed range: [5, 8]."
            )

        try:
            return FrameSamplerConfig(frame_count=frame_count)
        except ValueError as e:
            raise ConfigurationError(f"Frame sampler configuration error: {e}")

    def _build_yolo_config(self) -> YOLOConfig:
        """Build and validate YOLOConfig from raw config."""
        yolo_section = self._raw_config.get("yolo", {})

        checkpoint_path = yolo_section.get("checkpoint_path")
        if not checkpoint_path:
            raise ConfigurationError(
                "Required parameter 'yolo.checkpoint_path' is missing. "
                "Provide the path to the YOLO model checkpoint file."
            )

        # Validate checkpoint path exists
        if not os.path.isfile(checkpoint_path):
            raise ConfigurationError(
                f"Parameter 'yolo.checkpoint_path' references a file that does not exist: "
                f"'{checkpoint_path}'. Ensure the checkpoint file is present at this path."
            )

        device = self._get_with_default(
            "yolo.device",
            ["yolo", "device"],
            DOCUMENTED_DEFAULTS["yolo.device"],
        )

        input_resolution = self._get_with_default(
            "yolo.input_resolution",
            ["yolo", "input_resolution"],
            DOCUMENTED_DEFAULTS["yolo.input_resolution"],
        )

        confidence_threshold = self._get_with_default(
            "yolo.confidence_threshold",
            ["yolo", "confidence_threshold"],
            DOCUMENTED_DEFAULTS["yolo.confidence_threshold"],
        )

        # Validate confidence threshold range
        self._validate_confidence("yolo.confidence_threshold", confidence_threshold)

        # Validate input resolution range
        if not isinstance(input_resolution, int):
            raise ConfigurationError(
                f"Parameter 'yolo.input_resolution' must be an integer, "
                f"got {type(input_resolution).__name__} with value '{input_resolution}'."
            )
        if not (320 <= input_resolution <= 750):
            raise ConfigurationError(
                f"Parameter 'yolo.input_resolution' must be between 320 and 750 inclusive, "
                f"got {input_resolution}. Allowed range: [320, 750]."
            )

        try:
            return YOLOConfig(
                checkpoint_path=checkpoint_path,
                device=device,
                input_resolution=input_resolution,
                confidence_threshold=confidence_threshold,
            )
        except ValueError as e:
            raise ConfigurationError(f"YOLO configuration error: {e}")

    def _build_human_detector_config(self) -> HumanDetectorConfig:
        """Build and validate HumanDetectorConfig from raw config."""
        confidence_threshold = self._get_with_default(
            "detection.human.confidence_threshold",
            ["detection", "human", "confidence_threshold"],
            DOCUMENTED_DEFAULTS["detection.human.confidence_threshold"],
        )

        self._validate_confidence(
            "detection.human.confidence_threshold", confidence_threshold
        )

        try:
            return HumanDetectorConfig(confidence_threshold=confidence_threshold)
        except ValueError as e:
            raise ConfigurationError(f"Human detector configuration error: {e}")

    def _build_container_analyzer_config(self) -> ContainerAnalyzerConfig:
        """Build and validate ContainerAnalyzerConfig from raw config."""
        confidence_threshold = self._get_with_default(
            "detection.container.confidence_threshold",
            ["detection", "container", "confidence_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.confidence_threshold"],
        )
        flipped_aspect_ratio_threshold = self._get_with_default(
            "detection.container.flipped_aspect_ratio_threshold",
            ["detection", "container", "flipped_aspect_ratio_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.flipped_aspect_ratio_threshold"],
        )
        safe_overlap_threshold = self._get_with_default(
            "detection.container.safe_overlap_threshold",
            ["detection", "container", "safe_overlap_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.safe_overlap_threshold"],
        )
        ground_level_threshold = self._get_with_default(
            "detection.container.ground_level_threshold",
            ["detection", "container", "ground_level_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.ground_level_threshold"],
        )
        motion_threshold = self._get_with_default(
            "detection.container.motion_threshold",
            ["detection", "container", "motion_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.motion_threshold"],
        )
        iou_threshold = self._get_with_default(
            "detection.container.iou_threshold",
            ["detection", "container", "iou_threshold"],
            DOCUMENTED_DEFAULTS["detection.container.iou_threshold"],
        )

        # Validate confidence thresholds
        self._validate_confidence(
            "detection.container.confidence_threshold", confidence_threshold
        )
        self._validate_confidence(
            "detection.container.safe_overlap_threshold", safe_overlap_threshold
        )
        self._validate_confidence(
            "detection.container.ground_level_threshold", ground_level_threshold
        )
        self._validate_confidence(
            "detection.container.iou_threshold", iou_threshold
        )

        try:
            return ContainerAnalyzerConfig(
                confidence_threshold=confidence_threshold,
                flipped_aspect_ratio_threshold=flipped_aspect_ratio_threshold,
                safe_overlap_threshold=safe_overlap_threshold,
                ground_level_threshold=ground_level_threshold,
                motion_threshold=motion_threshold,
                iou_threshold=iou_threshold,
            )
        except ValueError as e:
            raise ConfigurationError(f"Container analyzer configuration error: {e}")

    def _build_alert_dispatcher_config(self) -> AlertDispatcherConfig:
        """Build and validate AlertDispatcherConfig from raw config."""
        rate_limit_seconds = self._get_with_default(
            "alerts.rate_limit_seconds",
            ["alerts", "rate_limit_seconds"],
            DOCUMENTED_DEFAULTS["alerts.rate_limit_seconds"],
        )
        channels = self._get_with_default(
            "alerts.channels",
            ["alerts", "channels"],
            DOCUMENTED_DEFAULTS["alerts.channels"],
        )

        # Validate rate_limit_seconds range
        if not isinstance(rate_limit_seconds, int):
            raise ConfigurationError(
                f"Parameter 'alerts.rate_limit_seconds' must be an integer, "
                f"got {type(rate_limit_seconds).__name__} with value '{rate_limit_seconds}'."
            )
        if not (10 <= rate_limit_seconds <= 300):
            raise ConfigurationError(
                f"Parameter 'alerts.rate_limit_seconds' must be between 10 and 300 inclusive, "
                f"got {rate_limit_seconds}. Allowed range: [10, 300]."
            )

        if not isinstance(channels, list) or len(channels) == 0:
            raise ConfigurationError(
                "Parameter 'alerts.channels' must be a non-empty list of channel names."
            )

        try:
            return AlertDispatcherConfig(
                rate_limit_seconds=rate_limit_seconds,
                channels=channels,
            )
        except ValueError as e:
            raise ConfigurationError(f"Alert dispatcher configuration error: {e}")

    def _build_camera_switcher_config(self) -> CameraSwitcherConfig:
        """Build CameraSwitcherConfig from raw config."""
        camera_sequence = self._raw_config.get("cameras", {}).get("sequence", [])

        try:
            return CameraSwitcherConfig(camera_list=list(camera_sequence))
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Camera switcher configuration error: {e}")

    def _validate_confidence(self, param_path: str, value: Any) -> None:
        """
        Validate a confidence threshold is in [0.0, 1.0].

        Args:
            param_path: Dotted parameter path for error messages.
            value: The value to validate.

        Raises:
            ConfigurationError: If value is not a number in [0.0, 1.0].
        """
        if not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"Parameter '{param_path}' must be a number, "
                f"got {type(value).__name__} with value '{value}'."
            )
        if not (0.0 <= value <= 1.0):
            raise ConfigurationError(
                f"Parameter '{param_path}' must be between 0.0 and 1.0 inclusive, "
                f"got {value}. Allowed range: [0.0, 1.0]."
            )

    def _require_section(self, section_name: str, yaml_key: str) -> None:
        """
        Ensure a required top-level section exists in config.

        Raises:
            ConfigurationError: If the section is missing.
        """
        if yaml_key not in self._raw_config:
            raise ConfigurationError(
                f"Required configuration section '{section_name}' is missing. "
                f"Add a '{yaml_key}:' section to your configuration file."
            )

    def _require_param(
        self, param_path: str, *keys: str
    ) -> Any:
        """
        Ensure a required parameter exists in config.

        Args:
            param_path: Dotted path for error messages.
            *keys: Sequence of keys to traverse into the config dict.

        Returns:
            The parameter value.

        Raises:
            ConfigurationError: If the parameter is missing.
        """
        current = self._raw_config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise ConfigurationError(
                    f"Required parameter '{param_path}' is missing from the configuration. "
                    f"Expected location: {' > '.join(keys)} in the YAML file."
                )
            current = current[key]
        return current

    def _get_with_default(
        self, param_path: str, keys: List[str], default: Any
    ) -> Any:
        """
        Get a configuration value, applying default if missing.

        Tracks which defaults are applied for logging.

        Args:
            param_path: Dotted path for tracking/logging.
            keys: Sequence of keys to traverse into the config dict.
            default: Default value to apply if not found.

        Returns:
            The configuration value or the default.
        """
        current = self._raw_config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                self._defaults_applied.append(param_path)
                return default
            current = current[key]

        if current is None:
            self._defaults_applied.append(param_path)
            return default

        return current

    def _log_defaults(self) -> None:
        """Log which default values are in use."""
        if self._defaults_applied:
            logger.info(
                "Configuration defaults applied for the following parameters:"
            )
            for param_path in self._defaults_applied:
                default_value = DOCUMENTED_DEFAULTS.get(param_path, "N/A")
                logger.info(
                    f"  {param_path} = {default_value} (default)"
                )
        else:
            logger.info("All configuration parameters explicitly specified; no defaults applied.")


def load_config(config_path: Optional[str] = None) -> ConfigurationManager:
    """
    Convenience function to create and load a ConfigurationManager.

    Args:
        config_path: Optional explicit path to YAML config file.

    Returns:
        Loaded and validated ConfigurationManager instance.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    manager = ConfigurationManager(config_path=config_path)
    manager.load()
    return manager
