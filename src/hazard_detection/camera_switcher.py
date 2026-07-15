"""
Camera Switcher stub module for the Hazard Detection System.

Provides a well-defined interface for camera transitions. The actual hardware
implementation is deferred until camera hardware details are available. This stub
always returns success for recognized camera IDs, introduces no delay, and
performs no hardware operations.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

from datetime import datetime, timezone

from hazard_detection.diagnostics import get_logger
from hazard_detection.models import CameraSwitcherConfig

logger = get_logger("camera_switcher")


class CameraSwitcher:
    """
    Stub camera switcher that simulates camera transitions.

    This component provides the interface for transitioning between camera feeds.
    The stub implementation always returns True for recognized camera IDs and
    False for unrecognized ones, logging all transition attempts with structured
    metadata.

    Configuration supports up to 16 camera entries via CameraSwitcherConfig.
    """

    def __init__(self, config: CameraSwitcherConfig):
        """
        Initialize the Camera Switcher with the given configuration.

        Args:
            config: CameraSwitcherConfig containing camera_list (up to 16 entries),
                    connection_types, and transition_params (all placeholders).
        """
        self._config = config
        self._camera_set = set(config.camera_list)
        logger.info(
            "CameraSwitcher initialized",
            extra={
                "camera_count": len(config.camera_list),
                "cameras": config.camera_list,
            },
        )

    def transition(self, target_camera_id: str) -> bool:
        """
        Transition to the target camera.

        Stub implementation: always returns True for recognized camera IDs,
        introduces no delay, and performs no hardware operations. Returns False
        for unrecognized camera IDs not in the configured list.

        Args:
            target_camera_id: The identifier of the camera to transition to.

        Returns:
            True if the camera ID is recognized (transition successful),
            False if the camera ID is not in the configured camera list.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if target_camera_id not in self._camera_set:
            logger.error(
                "Camera transition failed: unrecognized camera ID",
                extra={
                    "target_camera_id": target_camera_id,
                    "timestamp": timestamp,
                    "success": False,
                },
            )
            return False

        logger.info(
            "Camera transition successful",
            extra={
                "target_camera_id": target_camera_id,
                "timestamp": timestamp,
                "success": True,
            },
        )
        return True

    @property
    def camera_list(self) -> list:
        """Return the configured camera list."""
        return self._config.camera_list

    @property
    def config(self) -> CameraSwitcherConfig:
        """Return the current configuration."""
        return self._config
