"""
Synthetic Data Generator for Container Hazard Scenarios.

Generates synthetic training data by superimposing cargo container images
onto background scenes in various safe and unsafe configurations. Produces
YOLO-format annotations with class IDs and normalized bounding box coordinates.

Uses the existing DataAugmenter from generate_synthetic_data.py for applying
random transformations (rotation, flip, brightness, contrast, noise, translation, blur).

Requirements covered:
- 12.1: Superimpose containers onto backgrounds at non-overlapping positions (1-10 per scene)
- 12.2: Produce labelled pairs for safe and unsafe configurations
- 12.3: Apply 2-4 random DataAugmenter transformations per sample
- 12.4: Configurable samples per class (50-10000) with <=5% class balance deviation
- 12.5: YOLO-format annotation output (class_id, normalized bbox)
- 12.6: Skip corrupt inputs, log error, continue with valid inputs
- 12.7: Output at consistent resolution matching background dimensions
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Use relative import for diagnostics to avoid triggering __init__.py
# which has transitive dependencies on optional modules
try:
    from hazard_detection.diagnostics import get_logger
    logger = get_logger("synthetic_generator")
except (ImportError, ModuleNotFoundError):
    # Fallback: create a basic logger if diagnostics can't be imported
    logger = logging.getLogger("hazard_detection.synthetic_generator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# =============================================================================
# Container States and Class Mapping
# =============================================================================

# Container states mapped to Roboflow class IDs from data.yaml
# The 17-class taxonomy:
# 0: Boat - With Cargo
# 1: Container - Misaligned
# 2: Container - Open
# 3: Container - Picked
# 4: Container - Reefer
# 5: Container - Water Drop
# 6: Container -Separate
# 7: Container -Stacked
# 8: Crane
# 9: Human
# 10: Human - No Safety Clothes
# 11: Truck - No Container
# 12: Truck - With Container
# 13: Vehicle
# 14: Yard - Dropoff zone
# 15: Yard - No People
# 16: Yard - Operation Zone

CONTAINER_STATE_TO_CLASS_ID = {
    # Safe states
    "aligned": 7,       # Container - Stacked (normal aligned state)
    "closed": 7,        # Container - Stacked (normal closed state)
    "upright": 7,       # Container - Stacked (normal upright state)
    # Unsafe states
    "misaligned": 1,    # Container - Misaligned
    "open_door": 2,     # Container - Open
    "flipped": 1,       # Container - Misaligned (flipped is a misalignment variant)
    "dangling": 3,      # Container - Picked (dangling from crane)
}

SAFE_STATES = ["aligned", "closed", "upright"]
UNSAFE_STATES = ["misaligned", "open_door", "flipped", "dangling"]
ALL_STATES = SAFE_STATES + UNSAFE_STATES


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class SyntheticGeneratorConfig:
    """
    Configuration for the SyntheticDataGenerator.

    Attributes:
        containers_per_scene: Number of containers to place per scene (1-10).
        samples_per_class: Number of samples to generate per class (50-10000).
        background_dir: Path to directory containing background scene images.
        container_assets_dir: Path to directory containing container asset images.
        output_dir: Path to output directory for generated images and annotations.
        container_states: List of container states to generate.
        seed: Random seed for reproducibility. None for non-deterministic.
    """

    containers_per_scene: int = 3
    samples_per_class: int = 100
    background_dir: str = "data/backgrounds"
    container_assets_dir: str = "data/container_assets"
    output_dir: str = "data/synthetic_output"
    container_states: List[str] = field(default_factory=lambda: ALL_STATES.copy())
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.containers_per_scene, int) or not (1 <= self.containers_per_scene <= 10):
            raise ValueError(
                f"containers_per_scene must be an integer in [1, 10], got {self.containers_per_scene}"
            )
        if not isinstance(self.samples_per_class, int) or not (50 <= self.samples_per_class <= 10000):
            raise ValueError(
                f"samples_per_class must be an integer in [50, 10000], got {self.samples_per_class}"
            )
        if not isinstance(self.background_dir, str) or not self.background_dir.strip():
            raise ValueError("background_dir must be a non-empty string")
        if not isinstance(self.container_assets_dir, str) or not self.container_assets_dir.strip():
            raise ValueError("container_assets_dir must be a non-empty string")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")
        if not isinstance(self.container_states, list) or len(self.container_states) == 0:
            raise ValueError("container_states must be a non-empty list")
        for state in self.container_states:
            if state not in ALL_STATES:
                raise ValueError(
                    f"Unknown container state '{state}'. Must be one of {ALL_STATES}"
                )


# =============================================================================
# DataAugmenter (replicates the interface from generate_synthetic_data.py)
# =============================================================================


class DataAugmenter:
    """
    Data augmentation for synthetic container imagery.

    Provides the same transformations as the existing DataAugmenter in
    generate_synthetic_data.py: rotation, flip, brightness, contrast,
    noise, translation, blur.
    """

    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Rotate image by random angle."""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    @staticmethod
    def random_flip(image: np.ndarray) -> np.ndarray:
        """Randomly flip image horizontally."""
        if random.random() > 0.5:
            return cv2.flip(image, 1)
        return image

    @staticmethod
    def random_brightness(image: np.ndarray, factor_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """Adjust brightness randomly."""
        factor = random.uniform(*factor_range)
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def random_contrast(image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Adjust contrast randomly."""
        factor = random.uniform(*factor_range)
        mean = image.mean()
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, sigma_range: Tuple[float, float] = (5.0, 15.0)) -> np.ndarray:
        """Add Gaussian noise."""
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def random_translation(image: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """Randomly translate image."""
        h, w = image.shape[:2]
        tx = int(random.uniform(-max_shift, max_shift) * w)
        ty = int(random.uniform(-max_shift, max_shift) * h)
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return translated

    @staticmethod
    def random_blur(image: np.ndarray, kernel_size_range: Tuple[int, int] = (3, 7)) -> np.ndarray:
        """Apply random Gaussian blur."""
        kernel_size = random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def augment(self, image: np.ndarray, num_augmentations: int = 3) -> np.ndarray:
        """
        Apply random augmentations to image.

        Args:
            image: Input image (BGR format, numpy array).
            num_augmentations: Number of augmentations to apply (2-4 for synthetic gen).

        Returns:
            Augmented image.
        """
        augmentations = [
            self.random_rotation,
            self.random_flip,
            self.random_brightness,
            self.random_contrast,
            self.add_gaussian_noise,
            self.random_translation,
            self.random_blur,
        ]

        selected = random.sample(augmentations, min(num_augmentations, len(augmentations)))

        augmented = image.copy()
        for aug_func in selected:
            augmented = aug_func(augmented)

        return augmented


# =============================================================================
# SyntheticDataGenerator
# =============================================================================


class SyntheticDataGenerator:
    """
    Generates synthetic container scenarios for training diversity.

    Superimposes cargo container images onto background scenes at random
    non-overlapping positions, generating both safe and unsafe configurations
    with YOLO-format annotations.

    Attributes:
        augmenter: DataAugmenter instance for applying transformations.
        config: SyntheticGeneratorConfig with generation parameters.
    """

    def __init__(self, augmenter: DataAugmenter, config: SyntheticGeneratorConfig):
        """
        Initialize the SyntheticDataGenerator.

        Args:
            augmenter: DataAugmenter instance providing transformation methods.
            config: Configuration with containers_per_scene, samples_per_class,
                   directories, and container states to generate.
        """
        self.augmenter = augmenter
        self.config = config
        self._rng = random.Random(config.seed)
        self._np_rng = np.random.RandomState(config.seed)

        # Load assets
        self._backgrounds: List[np.ndarray] = []
        self._container_assets: List[np.ndarray] = []
        self._background_paths: List[str] = []
        self._container_paths: List[str] = []

        logger.info(
            "SyntheticDataGenerator initialized",
            extra={
                "extra_data": {
                    "containers_per_scene": config.containers_per_scene,
                    "samples_per_class": config.samples_per_class,
                    "states": config.container_states,
                    "background_dir": config.background_dir,
                    "container_assets_dir": config.container_assets_dir,
                    "output_dir": config.output_dir,
                }
            },
        )

    def _load_images_from_directory(self, directory: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all valid images from a directory.

        Skips corrupt or unreadable files, logs errors, and continues.

        Args:
            directory: Path to directory containing images.

        Returns:
            Tuple of (list of loaded images, list of valid file paths).
        """
        dir_path = Path(directory)
        images = []
        paths = []

        if not dir_path.exists():
            logger.error(
                f"Directory does not exist: {directory}",
                extra={"extra_data": {"directory": directory}},
            )
            return images, paths

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in image_extensions:
                continue

            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    logger.error(
                        f"Failed to decode image: {file_path}",
                        extra={"extra_data": {"file_path": str(file_path), "reason": "cv2.imread returned None"}},
                    )
                    continue

                images.append(img)
                paths.append(str(file_path))
            except Exception as e:
                logger.error(
                    f"Error reading image {file_path}: {e}",
                    extra={"extra_data": {"file_path": str(file_path), "reason": str(e)}},
                )
                continue

        logger.info(
            f"Loaded {len(images)} images from {directory}",
            extra={"extra_data": {"directory": directory, "count": len(images)}},
        )
        return images, paths

    def _load_assets(self) -> bool:
        """
        Load background scenes and container asset images.

        Returns:
            True if sufficient assets were loaded, False otherwise.
        """
        self._backgrounds, self._background_paths = self._load_images_from_directory(
            self.config.background_dir
        )
        self._container_assets, self._container_paths = self._load_images_from_directory(
            self.config.container_assets_dir
        )

        if not self._backgrounds:
            logger.error("No valid background images found. Cannot generate synthetic data.")
            return False
        if not self._container_assets:
            logger.error("No valid container asset images found. Cannot generate synthetic data.")
            return False

        return True

    def _resize_container(
        self, container: np.ndarray, target_width: int, target_height: int
    ) -> np.ndarray:
        """
        Resize a container image to target dimensions while preserving aspect ratio.

        The container is fitted within the target dimensions, padded if needed.

        Args:
            container: Container image (BGR).
            target_width: Desired width.
            target_height: Desired height.

        Returns:
            Resized container image.
        """
        h, w = container.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(container, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    def _apply_container_state(
        self, container: np.ndarray, state: str
    ) -> np.ndarray:
        """
        Apply visual transformations to a container image to simulate a given state.

        Args:
            container: Original container image.
            state: Container state to simulate.

        Returns:
            Modified container image representing the given state.
        """
        if state == "aligned" or state == "closed" or state == "upright":
            # Safe states: no modification needed
            return container

        if state == "misaligned":
            # Simulate misalignment: slight rotation and shift
            angle = self._rng.uniform(5, 15) * self._rng.choice([-1, 1])
            h, w = container.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(container, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        if state == "open_door":
            # Simulate open door: add a dark vertical strip on one side
            h, w = container.shape[:2]
            result = container.copy()
            door_width = max(1, w // 6)
            side = self._rng.choice([0, 1])
            if side == 0:
                result[:, :door_width] = (result[:, :door_width].astype(np.float32) * 0.3).astype(np.uint8)
            else:
                result[:, w - door_width:] = (result[:, w - door_width:].astype(np.float32) * 0.3).astype(np.uint8)
            return result

        if state == "flipped":
            # Simulate flipped: rotate 180 degrees
            return cv2.flip(container, -1)

        if state == "dangling":
            # Simulate dangling: container in upper portion, slightly rotated
            angle = self._rng.uniform(2, 8) * self._rng.choice([-1, 1])
            h, w = container.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(container, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        return container

    def _find_non_overlapping_position(
        self,
        placed_boxes: List[Tuple[int, int, int, int]],
        container_w: int,
        container_h: int,
        scene_w: int,
        scene_h: int,
        max_attempts: int = 100,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a random non-overlapping position for a container in the scene.

        Args:
            placed_boxes: List of already placed bounding boxes as (x, y, w, h).
            container_w: Width of the container to place.
            container_h: Height of the container to place.
            scene_w: Width of the background scene.
            scene_h: Height of the background scene.
            max_attempts: Maximum placement attempts before giving up.

        Returns:
            (x, y) position or None if no valid position found.
        """
        for _ in range(max_attempts):
            x = self._rng.randint(0, max(0, scene_w - container_w))
            y = self._rng.randint(0, max(0, scene_h - container_h))

            # Check overlap with all placed boxes
            overlaps = False
            for px, py, pw, ph in placed_boxes:
                # Check if rectangles overlap
                if (x < px + pw and x + container_w > px and
                        y < py + ph and y + container_h > py):
                    overlaps = True
                    break

            if not overlaps:
                return (x, y)

        return None

    def _superimpose_container(
        self,
        scene: np.ndarray,
        container: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """
        Superimpose a container image onto a background scene at position (x, y).

        Args:
            scene: Background scene image (will be modified in-place).
            container: Container image to superimpose.
            x: X position (left edge) in the scene.
            y: Y position (top edge) in the scene.

        Returns:
            Modified scene with the container superimposed.
        """
        ch, cw = container.shape[:2]
        sh, sw = scene.shape[:2]

        # Clip container to scene bounds
        x_end = min(x + cw, sw)
        y_end = min(y + ch, sh)
        crop_w = x_end - x
        crop_h = y_end - y

        if crop_w <= 0 or crop_h <= 0:
            return scene

        scene[y:y_end, x:x_end] = container[:crop_h, :crop_w]
        return scene

    def _generate_single_sample(
        self,
        state: str,
        sample_index: int,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Generate a single synthetic sample for a given container state.

        Places containers_per_scene containers on a random background at
        non-overlapping positions, applies state transformations, and generates
        YOLO-format annotations.

        Args:
            state: Container state to generate.
            sample_index: Index of this sample within its class.

        Returns:
            Tuple of (image, annotation_lines) or None if generation fails.
            annotation_lines are in YOLO format: "class_id cx cy w h"
        """
        # Select random background
        bg_idx = self._rng.randint(0, len(self._backgrounds) - 1)
        background = self._backgrounds[bg_idx].copy()
        scene_h, scene_w = background.shape[:2]

        # Determine container size range (proportional to scene)
        min_container_scale = 0.08
        max_container_scale = 0.25
        min_cw = max(20, int(scene_w * min_container_scale))
        max_cw = int(scene_w * max_container_scale)
        min_ch = max(15, int(scene_h * min_container_scale))
        max_ch = int(scene_h * max_container_scale)

        placed_boxes: List[Tuple[int, int, int, int]] = []
        annotations: List[str] = []

        num_containers = self.config.containers_per_scene

        for i in range(num_containers):
            # Select random container asset
            asset_idx = self._rng.randint(0, len(self._container_assets) - 1)
            container = self._container_assets[asset_idx].copy()

            # Random size within range
            target_w = self._rng.randint(min_cw, max_cw)
            target_h = self._rng.randint(min_ch, max_ch)

            # Resize container
            container = self._resize_container(container, target_w, target_h)
            actual_h, actual_w = container.shape[:2]

            # Apply state transformation
            container = self._apply_container_state(container, state)

            # Find non-overlapping position
            position = self._find_non_overlapping_position(
                placed_boxes, actual_w, actual_h, scene_w, scene_h
            )

            if position is None:
                logger.debug(
                    f"Could not place container {i} for sample {sample_index} "
                    f"(state={state}). Skipping this container.",
                    extra={"extra_data": {"state": state, "sample_index": sample_index, "container_idx": i}},
                )
                continue

            x, y = position

            # Superimpose onto background
            background = self._superimpose_container(background, container, x, y)

            # Record placement
            placed_boxes.append((x, y, actual_w, actual_h))

            # Generate YOLO annotation (normalized)
            cx = (x + actual_w / 2) / scene_w
            cy = (y + actual_h / 2) / scene_h
            nw = actual_w / scene_w
            nh = actual_h / scene_h

            # Clamp to [0.0, 1.0]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            class_id = CONTAINER_STATE_TO_CLASS_ID[state]
            annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            logger.debug(
                f"Placed container {i}: state={state}, pos=({x},{y}), "
                f"size=({actual_w},{actual_h}), class_id={class_id}",
                extra={
                    "extra_data": {
                        "state": state,
                        "position": (x, y),
                        "size": (actual_w, actual_h),
                        "class_id": class_id,
                        "normalized_bbox": (cx, cy, nw, nh),
                    }
                },
            )

        if not annotations:
            logger.warning(
                f"No containers placed for sample {sample_index} (state={state})",
                extra={"extra_data": {"state": state, "sample_index": sample_index}},
            )
            return None

        # Apply 2-4 random augmentations to the entire scene
        num_augmentations = self._rng.randint(2, 4)
        augmented_scene = self.augmenter.augment(background, num_augmentations=num_augmentations)

        logger.debug(
            f"Applied {num_augmentations} augmentations to sample {sample_index}",
            extra={
                "extra_data": {
                    "state": state,
                    "sample_index": sample_index,
                    "num_augmentations": num_augmentations,
                }
            },
        )

        return augmented_scene, annotations

    def generate(self) -> Optional[Dict[str, any]]:
        """
        Generate a balanced synthetic dataset with all configured container states.

        Produces images and YOLO-format annotations in the output directory.
        Ensures class balance (<=5% deviation between largest and smallest class).

        Returns:
            Dictionary with generation statistics, or None on failure.
            Keys: total_images, per_class_counts, output_dir, balance_deviation
        """
        logger.info("Starting synthetic data generation")

        # Load assets
        if not self._load_assets():
            return None

        # Create output directories
        output_path = Path(self.config.output_dir)
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Generate samples per class with balancing
        states = self.config.container_states
        samples_per_class = self.config.samples_per_class
        per_class_counts: Dict[str, int] = {}
        total_images = 0

        for state in states:
            generated = 0
            attempts = 0
            max_attempts = samples_per_class * 3  # Allow extra attempts for failures

            while generated < samples_per_class and attempts < max_attempts:
                attempts += 1
                result = self._generate_single_sample(state, generated)

                if result is None:
                    continue

                image, annotations = result

                # Save image
                filename = f"{state}_{generated:05d}"
                image_path = images_dir / f"{filename}.jpg"
                label_path = labels_dir / f"{filename}.txt"

                try:
                    cv2.imwrite(str(image_path), image)
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(annotations) + "\n")

                    generated += 1
                    total_images += 1
                except Exception as e:
                    logger.error(
                        f"Error saving sample {filename}: {e}",
                        extra={"extra_data": {"filename": filename, "reason": str(e)}},
                    )
                    continue

            per_class_counts[state] = generated
            logger.info(
                f"Generated {generated} samples for state '{state}' "
                f"(target: {samples_per_class}, attempts: {attempts})",
                extra={
                    "extra_data": {
                        "state": state,
                        "generated": generated,
                        "target": samples_per_class,
                        "attempts": attempts,
                    }
                },
            )

        # Compute class balance statistics
        if per_class_counts:
            counts = list(per_class_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            if max_count > 0:
                balance_deviation = (max_count - min_count) / max_count
            else:
                balance_deviation = 0.0
        else:
            balance_deviation = 0.0

        logger.info(
            f"Synthetic generation complete: {total_images} total images, "
            f"balance deviation: {balance_deviation:.4f} ({balance_deviation*100:.2f}%)",
            extra={
                "extra_data": {
                    "total_images": total_images,
                    "per_class_counts": per_class_counts,
                    "balance_deviation": balance_deviation,
                    "output_dir": str(output_path),
                }
            },
        )

        return {
            "total_images": total_images,
            "per_class_counts": per_class_counts,
            "output_dir": str(output_path),
            "balance_deviation": balance_deviation,
        }
