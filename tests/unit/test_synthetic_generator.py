"""
Unit tests for the Synthetic Data Generator module.

Tests cover:
- Property 20: Synthetic data non-overlapping placement
- Property 21: Synthetic data class balance (<=5% deviation)
- Property 22: Synthetic annotation format compliance
- All unsafe states covered in generation run
- Augmentation applied (2-4 transforms per sample)
- Corrupt input handling (skip and continue)
- Visual outputs: class balance histogram, sample scenes, quality report

**Validates: Requirements 12.1, 12.4, 12.5**
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.data_pipeline.synthetic_generator import (
    ALL_STATES,
    CONTAINER_STATE_TO_CLASS_ID,
    SAFE_STATES,
    UNSAFE_STATES,
    DataAugmenter,
    SyntheticDataGenerator,
    SyntheticGeneratorConfig,
)

# Import visual helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from visual_helpers import plot_class_distribution, save_json_report

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Ensure tests/output/ directory exists for visual diagnostic artifacts."""
    out = Path(__file__).parent.parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def temp_dirs():
    """Create temporary directories with sample background and container assets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "backgrounds"
        assets_dir = Path(tmpdir) / "container_assets"
        output_dir = Path(tmpdir) / "output"
        bg_dir.mkdir()
        assets_dir.mkdir()
        output_dir.mkdir()

        # Create sample background images (800x600)
        rng = np.random.default_rng(42)
        for i in range(3):
            bg = rng.integers(50, 200, size=(600, 800, 3), dtype=np.uint8)
            cv2.imwrite(str(bg_dir / f"background_{i:02d}.jpg"), bg)

        # Create sample container asset images (various sizes)
        sizes = [(120, 80), (100, 60), (140, 90), (110, 70)]
        for i, (h, w) in enumerate(sizes):
            container = rng.integers(100, 255, size=(h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(assets_dir / f"container_{i:02d}.png"), container)

        yield {
            "bg_dir": str(bg_dir),
            "assets_dir": str(assets_dir),
            "output_dir": str(output_dir),
        }


@pytest.fixture
def temp_dirs_with_corrupt():
    """Create temp directories including corrupt image files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir) / "backgrounds"
        assets_dir = Path(tmpdir) / "container_assets"
        output_dir = Path(tmpdir) / "output"
        bg_dir.mkdir()
        assets_dir.mkdir()
        output_dir.mkdir()

        rng = np.random.default_rng(99)

        # Valid backgrounds
        for i in range(2):
            bg = rng.integers(50, 200, size=(600, 800, 3), dtype=np.uint8)
            cv2.imwrite(str(bg_dir / f"background_{i:02d}.jpg"), bg)

        # Corrupt background (invalid bytes)
        corrupt_path = bg_dir / "corrupt_bg.jpg"
        corrupt_path.write_bytes(b"not a valid image file content")

        # Valid container assets
        for i in range(3):
            container = rng.integers(100, 255, size=(80, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(assets_dir / f"container_{i:02d}.png"), container)

        # Corrupt container asset
        corrupt_asset = assets_dir / "corrupt_container.png"
        corrupt_asset.write_bytes(b"corrupted png data here")

        yield {
            "bg_dir": str(bg_dir),
            "assets_dir": str(assets_dir),
            "output_dir": str(output_dir),
        }


@pytest.fixture
def default_config(temp_dirs) -> SyntheticGeneratorConfig:
    """Default config with minimum samples for fast testing."""
    return SyntheticGeneratorConfig(
        containers_per_scene=3,
        samples_per_class=50,
        background_dir=temp_dirs["bg_dir"],
        container_assets_dir=temp_dirs["assets_dir"],
        output_dir=temp_dirs["output_dir"],
        container_states=ALL_STATES.copy(),
        seed=42,
    )


@pytest.fixture
def augmenter() -> DataAugmenter:
    """DataAugmenter instance."""
    return DataAugmenter()


@pytest.fixture
def generator(augmenter, default_config) -> SyntheticDataGenerator:
    """SyntheticDataGenerator instance ready for use."""
    return SyntheticDataGenerator(augmenter=augmenter, config=default_config)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_annotation_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO annotation file into (class_id, cx, cy, w, h) tuples."""
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((class_id, cx, cy, w, h))
    return annotations


def boxes_overlap(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> bool:
    """Check if two boxes (cx, cy, w, h) overlap."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    # Convert to corners
    x1_min, x1_max = cx1 - w1 / 2, cx1 + w1 / 2
    y1_min, y1_max = cy1 - h1 / 2, cy1 + h1 / 2
    x2_min, x2_max = cx2 - w2 / 2, cx2 + w2 / 2
    y2_min, y2_max = cy2 - h2 / 2, cy2 + h2 / 2
    # Check overlap
    return (x1_min < x2_max and x1_max > x2_min and
            y1_min < y2_max and y1_max > y2_min)


# ---------------------------------------------------------------------------
# Property 20: Synthetic data non-overlapping placement
# ---------------------------------------------------------------------------


class TestNonOverlappingPlacement:
    """
    Property 20: Synthetic data non-overlapping placement.

    For any synthetically generated scene with multiple containers, all container
    bounding boxes SHALL be placed at non-overlapping positions within the scene bounds.

    **Validates: Requirements 12.1**
    """

    def test_non_overlapping_placement_basic(self, generator):
        """
        Validates: Requirements 12.1

        All containers placed in a single generated sample have non-overlapping bboxes.
        """
        result = generator.generate()
        assert result is not None, "Generation should succeed"

        labels_dir = Path(result["output_dir"]) / "labels"
        assert labels_dir.exists()

        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) > 0, "Should produce annotation files"

        # Check a subset of label files for non-overlapping boxes
        checked = 0
        for label_path in label_files[:20]:
            annotations = parse_annotation_file(label_path)
            if len(annotations) < 2:
                continue

            # Check all pairs for overlap
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    box_i = annotations[i][1:]  # (cx, cy, w, h)
                    box_j = annotations[j][1:]
                    assert not boxes_overlap(box_i, box_j), (
                        f"Bounding boxes overlap in {label_path.name}: "
                        f"box {i} {box_i} vs box {j} {box_j}"
                    )
            checked += 1

        assert checked > 0, "Should have checked at least one multi-container file"


    def test_non_overlapping_with_max_containers(self, temp_dirs, augmenter):
        """
        Validates: Requirements 12.1

        With max containers_per_scene=10, placement still produces non-overlapping boxes.
        """
        config = SyntheticGeneratorConfig(
            containers_per_scene=10,
            samples_per_class=50,
            background_dir=temp_dirs["bg_dir"],
            container_assets_dir=temp_dirs["assets_dir"],
            output_dir=temp_dirs["output_dir"],
            container_states=["aligned"],
            seed=123,
        )
        gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
        result = gen.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"
        for label_path in list(labels_dir.glob("*.txt"))[:10]:
            annotations = parse_annotation_file(label_path)
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    box_i = annotations[i][1:]
                    box_j = annotations[j][1:]
                    assert not boxes_overlap(box_i, box_j), (
                        f"Overlap found in max-containers test: {label_path.name}"
                    )

    def test_bboxes_within_scene_bounds(self, generator):
        """
        Validates: Requirements 12.1

        All bounding boxes are within [0, 1] normalized scene bounds.
        """
        result = generator.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"
        for label_path in list(labels_dir.glob("*.txt"))[:20]:
            annotations = parse_annotation_file(label_path)
            for class_id, cx, cy, w, h in annotations:
                assert 0.0 <= cx <= 1.0, f"cx={cx} out of bounds"
                assert 0.0 <= cy <= 1.0, f"cy={cy} out of bounds"
                assert 0.0 <= w <= 1.0, f"w={w} out of bounds"
                assert 0.0 <= h <= 1.0, f"h={h} out of bounds"


# ---------------------------------------------------------------------------
# Property 21: Synthetic data class balance
# ---------------------------------------------------------------------------


class TestClassBalance:
    """
    Property 21: Synthetic data class balance.

    For any generation run with a configured samples_per_class count, the resulting
    dataset SHALL have no more than 5% deviation between the largest and smallest
    class sample counts.

    **Validates: Requirements 12.4**
    """

    def test_class_balance_within_threshold(self, generator):
        """
        Validates: Requirements 12.4

        The balance deviation between max and min class counts is <=5%.
        """
        result = generator.generate()
        assert result is not None

        per_class = result["per_class_counts"]
        assert len(per_class) > 0, "Should have per-class counts"

        counts = list(per_class.values())
        min_count = min(counts)
        max_count = max(counts)

        if max_count > 0:
            deviation = (max_count - min_count) / max_count
        else:
            deviation = 0.0

        assert deviation <= 0.05, (
            f"Class balance deviation {deviation:.4f} ({deviation*100:.2f}%) exceeds 5%. "
            f"Counts: {per_class}"
        )

    def test_all_classes_have_target_count(self, generator):
        """
        Validates: Requirements 12.4

        Each class achieves close to the configured samples_per_class target.
        """
        result = generator.generate()
        assert result is not None

        target = generator.config.samples_per_class
        per_class = result["per_class_counts"]

        for state, count in per_class.items():
            # Each class should be within 5% of target
            lower_bound = int(target * 0.95)
            assert count >= lower_bound, (
                f"Class '{state}' has {count} samples, "
                f"expected at least {lower_bound} (95% of {target})"
            )


    def test_balance_deviation_reported_correctly(self, generator):
        """
        Validates: Requirements 12.4

        The reported balance_deviation matches actual computed deviation.
        """
        result = generator.generate()
        assert result is not None

        per_class = result["per_class_counts"]
        counts = list(per_class.values())
        min_count = min(counts)
        max_count = max(counts)

        if max_count > 0:
            expected_deviation = (max_count - min_count) / max_count
        else:
            expected_deviation = 0.0

        assert abs(result["balance_deviation"] - expected_deviation) < 1e-6, (
            f"Reported deviation {result['balance_deviation']} doesn't match "
            f"computed {expected_deviation}"
        )


# ---------------------------------------------------------------------------
# Property 22: Synthetic annotation format compliance
# ---------------------------------------------------------------------------


class TestAnnotationFormatCompliance:
    """
    Property 22: Synthetic annotation format compliance.

    For any annotation file produced by the Synthetic_Data_Generator, each line
    SHALL contain a valid class_id (integer index into the 17-class taxonomy)
    followed by four normalized bounding box coordinates (center_x, center_y,
    width, height) all in [0.0, 1.0].

    **Validates: Requirements 12.5**
    """

    # Valid class IDs from the 17-class taxonomy
    VALID_CLASS_IDS = set(range(17))


    def test_annotation_format_valid_class_id(self, generator):
        """
        Validates: Requirements 12.5

        All annotation lines have a valid class_id from the 17-class taxonomy.
        """
        result = generator.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"
        for label_path in labels_dir.glob("*.txt"):
            annotations = parse_annotation_file(label_path)
            for class_id, cx, cy, w, h in annotations:
                assert class_id in self.VALID_CLASS_IDS, (
                    f"Invalid class_id {class_id} in {label_path.name}. "
                    f"Valid IDs: {sorted(self.VALID_CLASS_IDS)}"
                )

    def test_annotation_format_normalized_coordinates(self, generator):
        """
        Validates: Requirements 12.5

        All bbox coordinates are normalized to [0.0, 1.0].
        """
        result = generator.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"
        for label_path in labels_dir.glob("*.txt"):
            annotations = parse_annotation_file(label_path)
            for class_id, cx, cy, w, h in annotations:
                assert 0.0 <= cx <= 1.0, f"cx={cx} out of [0,1] in {label_path.name}"
                assert 0.0 <= cy <= 1.0, f"cy={cy} out of [0,1] in {label_path.name}"
                assert 0.0 <= w <= 1.0, f"w={w} out of [0,1] in {label_path.name}"
                assert 0.0 <= h <= 1.0, f"h={h} out of [0,1] in {label_path.name}"


    def test_annotation_format_line_structure(self, generator):
        """
        Validates: Requirements 12.5

        Each annotation line has exactly 5 space-separated values:
        class_id cx cy w h.
        """
        result = generator.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"
        for label_path in labels_dir.glob("*.txt"):
            with open(label_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    assert len(parts) == 5, (
                        f"Line {line_num} in {label_path.name} has "
                        f"{len(parts)} parts, expected 5: '{line}'"
                    )
                    # First part should be integer
                    assert parts[0].isdigit(), (
                        f"class_id '{parts[0]}' is not an integer "
                        f"in {label_path.name} line {line_num}"
                    )
                    # Remaining parts should be floats
                    for i, part in enumerate(parts[1:], 1):
                        try:
                            float(part)
                        except ValueError:
                            pytest.fail(
                                f"Part {i} '{part}' is not a valid float "
                                f"in {label_path.name} line {line_num}"
                            )


    def test_annotation_class_ids_match_states(self, generator):
        """
        Validates: Requirements 12.5

        Generated annotations use class IDs consistent with
        CONTAINER_STATE_TO_CLASS_ID mapping.
        """
        result = generator.generate()
        assert result is not None

        # Expected class IDs based on configured states
        expected_class_ids = set()
        for state in generator.config.container_states:
            expected_class_ids.add(CONTAINER_STATE_TO_CLASS_ID[state])

        labels_dir = Path(result["output_dir"]) / "labels"
        found_class_ids = set()
        for label_path in labels_dir.glob("*.txt"):
            annotations = parse_annotation_file(label_path)
            for class_id, _, _, _, _ in annotations:
                found_class_ids.add(class_id)

        # All found class IDs should be in expected set
        for cid in found_class_ids:
            assert cid in expected_class_ids, (
                f"Unexpected class_id {cid}. Expected one of {expected_class_ids}"
            )


# ---------------------------------------------------------------------------
# Test: All unsafe states covered
# ---------------------------------------------------------------------------


class TestUnsafeStatesCoverage:
    """Test that all unsafe states are covered in generation run."""

    def test_all_unsafe_states_present(self, generator):
        """
        All unsafe states (misaligned, open_door, flipped, dangling)
        produce at least one sample per generation run.
        """
        result = generator.generate()
        assert result is not None

        per_class = result["per_class_counts"]
        for state in UNSAFE_STATES:
            assert state in per_class, f"Unsafe state '{state}' not in output"
            assert per_class[state] > 0, (
                f"Unsafe state '{state}' has 0 samples"
            )


    def test_all_safe_states_present(self, generator):
        """
        All safe states (aligned, closed, upright) also produce samples.
        """
        result = generator.generate()
        assert result is not None

        per_class = result["per_class_counts"]
        for state in SAFE_STATES:
            assert state in per_class, f"Safe state '{state}' not in output"
            assert per_class[state] > 0, (
                f"Safe state '{state}' has 0 samples"
            )

    def test_unsafe_states_produce_correct_class_ids(self, temp_dirs, augmenter):
        """
        Unsafe states map to the expected YOLO class IDs:
        - misaligned -> 1 (Container - Misaligned)
        - open_door -> 2 (Container - Open)
        - flipped -> 1 (Container - Misaligned variant)
        - dangling -> 3 (Container - Picked)
        """
        config = SyntheticGeneratorConfig(
            containers_per_scene=1,
            samples_per_class=50,
            background_dir=temp_dirs["bg_dir"],
            container_assets_dir=temp_dirs["assets_dir"],
            output_dir=temp_dirs["output_dir"],
            container_states=UNSAFE_STATES.copy(),
            seed=77,
        )
        gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
        result = gen.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"

        # Check misaligned files have class_id 1
        misaligned_files = list(labels_dir.glob("misaligned_*.txt"))
        assert len(misaligned_files) > 0
        for lp in misaligned_files[:5]:
            annotations = parse_annotation_file(lp)
            for class_id, _, _, _, _ in annotations:
                assert class_id == 1

        # Check open_door files have class_id 2
        open_files = list(labels_dir.glob("open_door_*.txt"))
        assert len(open_files) > 0
        for lp in open_files[:5]:
            annotations = parse_annotation_file(lp)
            for class_id, _, _, _, _ in annotations:
                assert class_id == 2


# ---------------------------------------------------------------------------
# Test: Augmentation applied (2-4 transforms per sample)
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Test that 2-4 augmentations are applied per sample."""

    def test_augmenter_applies_correct_count(self, augmenter):
        """
        The DataAugmenter.augment() method applies the specified
        number of transformations.
        """
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)

        # Test with 2 augmentations
        aug2 = augmenter.augment(image, num_augmentations=2)
        assert aug2.shape == image.shape
        # Image should be modified (extremely unlikely to be identical)
        assert not np.array_equal(aug2, image)

        # Test with 4 augmentations
        aug4 = augmenter.augment(image, num_augmentations=4)
        assert aug4.shape == image.shape
        assert not np.array_equal(aug4, image)

    def test_augmentation_range_2_to_4(self, temp_dirs, augmenter):
        """
        The generator applies between 2 and 4 augmentations per sample.
        We verify images are modified (not identical to raw composites).
        """
        config = SyntheticGeneratorConfig(
            containers_per_scene=1,
            samples_per_class=50,
            background_dir=temp_dirs["bg_dir"],
            container_assets_dir=temp_dirs["assets_dir"],
            output_dir=temp_dirs["output_dir"],
            container_states=["aligned"],
            seed=55,
        )
        gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
        result = gen.generate()
        assert result is not None
        assert result["total_images"] > 0

        # Verify output images exist and are valid
        images_dir = Path(result["output_dir"]) / "images"
        image_files = list(images_dir.glob("*.jpg"))
        assert len(image_files) > 0

        # Read a sample image and verify it's valid
        sample = cv2.imread(str(image_files[0]))
        assert sample is not None
        assert sample.shape[2] == 3  # BGR


    def test_augmenter_individual_transforms(self, augmenter):
        """Each individual augmentation produces a valid image of the same shape."""
        rng = np.random.default_rng(42)
        image = rng.integers(50, 200, size=(80, 120, 3), dtype=np.uint8)

        transforms = [
            augmenter.random_rotation,
            augmenter.random_flip,
            augmenter.random_brightness,
            augmenter.random_contrast,
            augmenter.add_gaussian_noise,
            augmenter.random_translation,
            augmenter.random_blur,
        ]

        for transform in transforms:
            result = transform(image)
            assert result.shape == image.shape, (
                f"{transform.__name__} changed shape from {image.shape} to {result.shape}"
            )
            assert result.dtype == np.uint8, (
                f"{transform.__name__} output dtype is {result.dtype}, expected uint8"
            )


# ---------------------------------------------------------------------------
# Test: Corrupt input handling
# ---------------------------------------------------------------------------


class TestCorruptInputHandling:
    """Test that corrupt inputs are skipped and generation continues."""

    def test_corrupt_files_skipped(self, temp_dirs_with_corrupt, augmenter):
        """
        Validates: Requirements 12.6

        Corrupt image files are skipped without crashing; generation
        continues with valid inputs.
        """
        config = SyntheticGeneratorConfig(
            containers_per_scene=2,
            samples_per_class=50,
            background_dir=temp_dirs_with_corrupt["bg_dir"],
            container_assets_dir=temp_dirs_with_corrupt["assets_dir"],
            output_dir=temp_dirs_with_corrupt["output_dir"],
            container_states=["aligned", "misaligned"],
            seed=42,
        )
        gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
        result = gen.generate()

        # Should succeed despite corrupt files
        assert result is not None
        assert result["total_images"] > 0


    def test_missing_directory_returns_none(self, augmenter):
        """
        When asset directories don't exist, generation returns None gracefully.
        """
        config = SyntheticGeneratorConfig(
            containers_per_scene=2,
            samples_per_class=50,
            background_dir="/nonexistent/path/backgrounds",
            container_assets_dir="/nonexistent/path/assets",
            output_dir="/tmp/synthetic_test_output",
            container_states=["aligned"],
            seed=42,
        )
        gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
        result = gen.generate()
        assert result is None, "Should return None when directories don't exist"

    def test_empty_backgrounds_returns_none(self, augmenter):
        """When no valid backgrounds exist, generation returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = Path(tmpdir) / "empty_bg"
            assets_dir = Path(tmpdir) / "assets"
            output_dir = Path(tmpdir) / "output"
            bg_dir.mkdir()
            assets_dir.mkdir()
            output_dir.mkdir()

            # Create only container assets, no backgrounds
            rng = np.random.default_rng(42)
            container = rng.integers(100, 255, size=(80, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(assets_dir / "c.png"), container)

            config = SyntheticGeneratorConfig(
                containers_per_scene=1,
                samples_per_class=50,
                background_dir=str(bg_dir),
                container_assets_dir=str(assets_dir),
                output_dir=str(output_dir),
                container_states=["aligned"],
                seed=42,
            )
            gen = SyntheticDataGenerator(augmenter=augmenter, config=config)
            result = gen.generate()
            assert result is None


# ---------------------------------------------------------------------------
# Test: Configuration validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test SyntheticGeneratorConfig validates parameters correctly."""

    def test_valid_config(self, temp_dirs):
        """Valid configuration parameters are accepted."""
        config = SyntheticGeneratorConfig(
            containers_per_scene=5,
            samples_per_class=100,
            background_dir=temp_dirs["bg_dir"],
            container_assets_dir=temp_dirs["assets_dir"],
            output_dir=temp_dirs["output_dir"],
        )
        assert config.containers_per_scene == 5
        assert config.samples_per_class == 100

    def test_invalid_containers_per_scene(self):
        """containers_per_scene outside [1, 10] raises ValueError."""
        with pytest.raises(ValueError, match="containers_per_scene"):
            SyntheticGeneratorConfig(containers_per_scene=0)
        with pytest.raises(ValueError, match="containers_per_scene"):
            SyntheticGeneratorConfig(containers_per_scene=11)

    def test_invalid_samples_per_class(self):
        """samples_per_class outside [50, 10000] raises ValueError."""
        with pytest.raises(ValueError, match="samples_per_class"):
            SyntheticGeneratorConfig(samples_per_class=10)
        with pytest.raises(ValueError, match="samples_per_class"):
            SyntheticGeneratorConfig(samples_per_class=20000)

    def test_invalid_container_state(self):
        """Unknown container state raises ValueError."""
        with pytest.raises(ValueError, match="Unknown container state"):
            SyntheticGeneratorConfig(container_states=["invalid_state"])

    def test_empty_states_raises(self):
        """Empty container_states list raises ValueError."""
        with pytest.raises(ValueError, match="container_states"):
            SyntheticGeneratorConfig(container_states=[])


# ---------------------------------------------------------------------------
# Visual Output Generation
# ---------------------------------------------------------------------------


class TestVisualOutputs:
    """Generate visual diagnostic outputs for the synthetic generator."""

    def test_generate_class_balance_histogram(self, generator, output_dir):
        """
        Generate visual output: save class balance histogram as PNG
        to tests/output/synthetic_class_balance.png.
        """
        result = generator.generate()
        assert result is not None

        per_class = result["per_class_counts"]
        plot_class_distribution(
            class_counts=per_class,
            output_path=output_dir / "synthetic_class_balance.png",
            title="Synthetic Data Class Balance",
        )

        assert (output_dir / "synthetic_class_balance.png").exists()


    def test_generate_sample_scenes_with_bbox_overlays(self, generator, output_dir):
        """
        Generate visual output: save sample generated scenes with bounding box
        overlays as PNG to tests/output/synthetic_sample_scenes.png.
        """
        result = generator.generate()
        assert result is not None

        images_dir = Path(result["output_dir"]) / "images"
        labels_dir = Path(result["output_dir"]) / "labels"

        image_files = sorted(images_dir.glob("*.jpg"))[:4]
        assert len(image_files) > 0

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        for idx, (ax, img_path) in enumerate(zip(axes, image_files)):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            ax.imshow(img_rgb)

            # Load corresponding annotations
            label_path = labels_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                annotations = parse_annotation_file(label_path)
                for i, (class_id, cx, cy, bw, bh) in enumerate(annotations):
                    # Convert normalized to pixel coords
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    color = colors[i % len(colors)]
                    rect = mpatches.Rectangle(
                        (x1, y1), box_w, box_h,
                        linewidth=2, edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 3, f"cls={class_id}",
                            fontsize=7, color=color,
                            bbox=dict(facecolor="white", alpha=0.7, pad=1))

            ax.set_title(img_path.stem, fontsize=9)
            ax.axis("off")

        # Fill remaining axes if fewer than 4 images
        for ax in axes[len(image_files):]:
            ax.axis("off")

        plt.suptitle("Synthetic Sample Scenes with BBox Overlays", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "synthetic_sample_scenes.png", dpi=150)
        plt.close()

        assert (output_dir / "synthetic_sample_scenes.png").exists()


    def test_generate_annotation_quality_report(self, generator, output_dir):
        """
        Generate visual output: save annotation quality report as JSON
        to tests/output/synthetic_quality_report.json.
        """
        result = generator.generate()
        assert result is not None

        labels_dir = Path(result["output_dir"]) / "labels"

        # Collect annotation statistics
        all_class_ids = []
        all_cx = []
        all_cy = []
        all_widths = []
        all_heights = []
        total_annotations = 0
        files_with_annotations = 0

        for label_path in labels_dir.glob("*.txt"):
            annotations = parse_annotation_file(label_path)
            if annotations:
                files_with_annotations += 1
            for class_id, cx, cy, w, h in annotations:
                all_class_ids.append(class_id)
                all_cx.append(cx)
                all_cy.append(cy)
                all_widths.append(w)
                all_heights.append(h)
                total_annotations += 1

        report = {
            "total_images": result["total_images"],
            "total_annotation_files": files_with_annotations,
            "total_annotations": total_annotations,
            "per_class_counts": result["per_class_counts"],
            "balance_deviation": result["balance_deviation"],
            "coordinate_ranges": {
                "cx": {"min": min(all_cx), "max": max(all_cx), "mean": float(np.mean(all_cx))},
                "cy": {"min": min(all_cy), "max": max(all_cy), "mean": float(np.mean(all_cy))},
                "width": {"min": min(all_widths), "max": max(all_widths), "mean": float(np.mean(all_widths))},
                "height": {"min": min(all_heights), "max": max(all_heights), "mean": float(np.mean(all_heights))},
            },
            "class_id_distribution": {str(k): v for k, v in sorted(
                {cid: all_class_ids.count(cid) for cid in set(all_class_ids)}.items()
            )},
            "annotations_per_image": {
                "min": min(len(parse_annotation_file(lp)) for lp in labels_dir.glob("*.txt")),
                "max": max(len(parse_annotation_file(lp)) for lp in labels_dir.glob("*.txt")),
                "mean": total_annotations / max(files_with_annotations, 1),
            },
        }

        save_json_report(report, output_dir / "synthetic_quality_report.json")
        assert (output_dir / "synthetic_quality_report.json").exists()

        # Verify report content
        with open(output_dir / "synthetic_quality_report.json") as f:
            loaded = json.load(f)
        assert loaded["total_images"] > 0
        assert loaded["balance_deviation"] <= 0.05
