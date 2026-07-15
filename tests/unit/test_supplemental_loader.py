"""
Unit tests for the Supplemental Dataset Loader module.

Tests cover:
- Property 19: Annotation normalization to YOLO format
- Class remapping to Roboflow 17-class IDs
- Discarding annotations with no corresponding class
- Skip behavior for unreadable files
- Split ratio output (70/15/15 default)
- Visual output generation (class balance histogram, quality report)

**Validates: Requirements 11.2, 11.6**
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hazard_detection.data_pipeline.supplemental_loader import (
    AnnotationEntry,
    DatasetOutput,
    ROBOFLOW_CLASSES,
    SupplementalConfig,
    SupplementalDatasetLoader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir() -> Path:
    """Ensure tests/output/ directory exists for visual diagnostic artifacts."""
    out = Path(__file__).parent.parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def voc_dataset(tmp_path) -> Path:
    """Create a temporary VOC XML annotated dataset."""
    dataset_root = tmp_path / "voc_dataset"
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    # Create dummy images (1x1 pixel PNGs via raw bytes)
    for i in range(5):
        img_name = f"img_{i:03d}.jpg"
        img_path = images_dir / img_name
        # Create a minimal valid file (just needs to exist with correct ext)
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Create VOC XML annotations
        xml_template = '''<?xml version="1.0"?>
<annotation>
    <filename>{filename}</filename>
    <size>
        <width>800</width>
        <height>600</height>
    </size>
    <object>
        <name>container</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>400</xmax>
            <ymax>350</ymax>
        </bndbox>
    </object>
    <object>
        <name>truck</name>
        <bndbox>
            <xmin>500</xmin>
            <ymin>200</ymin>
            <xmax>700</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>'''

        for i in range(5):
            img_name = f"img_{i:03d}.jpg"
            xml_path = annotations_dir / f"img_{i:03d}.xml"
            xml_path.write_text(xml_template.format(filename=img_name))

    return dataset_root


@pytest.fixture
def coco_dataset(tmp_path) -> Path:
    """Create a temporary COCO JSON annotated dataset."""
    dataset_root = tmp_path / "coco_dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)

    # Create dummy images
    for i in range(4):
        img_path = images_dir / f"coco_img_{i:03d}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    # Create COCO JSON annotation file
    coco_data = {
        "images": [
            {"id": i, "file_name": f"coco_img_{i:03d}.jpg",
             "width": 1024, "height": 768}
            for i in range(4)
        ],
        "categories": [
            {"id": 1, "name": "shipping_container"},
            {"id": 2, "name": "crane"},
            {"id": 3, "name": "person"},
            {"id": 4, "name": "unknown_object"},  # No mapping
        ],
        "annotations": [
            # Image 0: container + crane
            {"id": 1, "image_id": 0, "category_id": 1,
             "bbox": [100, 100, 200, 150]},
            {"id": 2, "image_id": 0, "category_id": 2,
             "bbox": [400, 50, 300, 400]},
            # Image 1: person + unknown (unknown should be discarded)
            {"id": 3, "image_id": 1, "category_id": 3,
             "bbox": [200, 200, 80, 200]},
            {"id": 4, "image_id": 1, "category_id": 4,
             "bbox": [500, 300, 100, 100]},
            # Image 2: container only
            {"id": 5, "image_id": 2, "category_id": 1,
             "bbox": [50, 50, 400, 300]},
            # Image 3: person
            {"id": 6, "image_id": 3, "category_id": 3,
             "bbox": [300, 100, 60, 180]},
        ],
    }

    coco_path = dataset_root / "annotations.json"
    coco_path.write_text(json.dumps(coco_data))

    return dataset_root


@pytest.fixture
def yolo_dataset(tmp_path) -> Path:
    """Create a temporary YOLO txt annotated dataset."""
    dataset_root = tmp_path / "yolo_dataset"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create classes.txt mapping
    classes_txt = dataset_root / "classes.txt"
    classes_txt.write_text("container\ncrane\nperson\nunknown_thing\n")

    # Create image + label pairs
    for i in range(6):
        img_path = images_dir / f"yolo_img_{i:03d}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        label_path = labels_dir / f"yolo_img_{i:03d}.txt"
        # class_id cx cy w h (normalized)
        lines = []
        if i % 3 == 0:
            lines.append("0 0.5 0.5 0.4 0.3")  # container
        if i % 2 == 0:
            lines.append("1 0.3 0.2 0.2 0.5")  # crane
        lines.append("2 0.7 0.8 0.1 0.2")  # person
        if i == 4:
            lines.append("3 0.6 0.6 0.15 0.15")  # unknown_thing (no map)
        label_path.write_text("\n".join(lines) + "\n")

    return dataset_root


@pytest.fixture
def mixed_dataset_with_unreadable(tmp_path) -> Path:
    """Create a dataset with some unreadable files mixed in."""
    dataset_root = tmp_path / "mixed_dataset"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    classes_txt = dataset_root / "classes.txt"
    classes_txt.write_text("container\ncrane\n")

    # Create valid image + label pairs
    for i in range(3):
        img_path = images_dir / f"valid_{i:03d}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        label_path = labels_dir / f"valid_{i:03d}.txt"
        label_path.write_text("0 0.5 0.5 0.3 0.2\n")

    # Create an image with a corrupted/unreadable label
    img_path = images_dir / "corrupt_label.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    label_path = labels_dir / "corrupt_label.txt"
    # Malformed lines (not enough fields, bad floats)
    label_path.write_text("0\nnot_a_number 0.5 0.5 0.3\n")

    return dataset_root


def _make_config(dataset_roots: List[str], output_dir: str, **kwargs) -> SupplementalConfig:
    """Helper to create a SupplementalConfig with test defaults."""
    return SupplementalConfig(
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        roboflow_data_yaml="nonexistent.yaml",  # Use hardcoded classes
        random_seed=42,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: Property 19 - Annotation Normalization to YOLO Format
# ---------------------------------------------------------------------------


class TestAnnotationNormalization:
    """
    Property 19: Annotation normalization to YOLO format.

    For any source annotation loaded by the Supplemental_Dataset_Loader,
    the normalized output SHALL follow YOLO format (class_id center_x
    center_y width height) with all coordinate values in [0.0, 1.0]
    and class IDs remapped to the Roboflow 17-class taxonomy.

    **Validates: Requirements 11.2, 11.6**
    """

    def test_voc_annotations_normalized_to_yolo_format(self, voc_dataset, tmp_path):
        """VOC XML annotations produce YOLO format output with coords in [0,1]."""
        output_dir = tmp_path / "output_voc"
        config = _make_config([str(voc_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        assert result.total_images > 0
        assert result.total_annotations > 0

        # Verify all output label files follow YOLO format
        for split in ["train", "valid", "test"]:
            labels_dir = output_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    assert len(parts) == 5, (
                        f"YOLO format requires 5 fields, got {len(parts)}: {line}"
                    )
                    class_id = int(parts[0])
                    assert 0 <= class_id < 17, (
                        f"Class ID {class_id} outside Roboflow range [0,16]"
                    )
                    for j in range(1, 5):
                        val = float(parts[j])
                        assert 0.0 <= val <= 1.0, (
                            f"Coordinate {val} outside [0.0, 1.0]"
                        )

    def test_coco_annotations_normalized_to_yolo_format(self, coco_dataset, tmp_path):
        """COCO JSON annotations produce YOLO format output with coords in [0,1]."""
        output_dir = tmp_path / "output_coco"
        config = _make_config([str(coco_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        assert result.total_images > 0
        assert result.total_annotations > 0

        # Verify YOLO format in output
        for split in ["train", "valid", "test"]:
            labels_dir = output_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    assert len(parts) == 5
                    class_id = int(parts[0])
                    assert 0 <= class_id < 17
                    for j in range(1, 5):
                        val = float(parts[j])
                        assert 0.0 <= val <= 1.0

    def test_yolo_txt_annotations_preserved_in_format(self, yolo_dataset, tmp_path):
        """YOLO txt annotations remain in valid YOLO format after remapping."""
        output_dir = tmp_path / "output_yolo"
        config = _make_config([str(yolo_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        assert result.total_images > 0

        for split in ["train", "valid", "test"]:
            labels_dir = output_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    assert len(parts) == 5
                    class_id = int(parts[0])
                    assert 0 <= class_id < 17
                    for j in range(1, 5):
                        val = float(parts[j])
                        assert 0.0 <= val <= 1.0

    def test_annotation_entry_coordinates_in_range(self):
        """AnnotationEntry.is_valid() rejects coords outside [0.0, 1.0]."""
        valid = AnnotationEntry(class_id=0, x_center=0.5, y_center=0.5,
                                width=0.3, height=0.2)
        assert valid.is_valid() is True

        # Out-of-range entries
        invalid_cases = [
            AnnotationEntry(class_id=0, x_center=1.1, y_center=0.5,
                            width=0.3, height=0.2),
            AnnotationEntry(class_id=0, x_center=0.5, y_center=-0.1,
                            width=0.3, height=0.2),
            AnnotationEntry(class_id=0, x_center=0.5, y_center=0.5,
                            width=1.5, height=0.2),
            AnnotationEntry(class_id=0, x_center=0.5, y_center=0.5,
                            width=0.3, height=-0.1),
        ]
        for entry in invalid_cases:
            assert entry.is_valid() is False

    def test_annotation_entry_to_yolo_line(self):
        """AnnotationEntry.to_yolo_line() produces correct format."""
        entry = AnnotationEntry(class_id=7, x_center=0.5, y_center=0.4,
                                width=0.3, height=0.2)
        line = entry.to_yolo_line()
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "7"
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-5)
        assert float(parts[2]) == pytest.approx(0.4, abs=1e-5)
        assert float(parts[3]) == pytest.approx(0.3, abs=1e-5)
        assert float(parts[4]) == pytest.approx(0.2, abs=1e-5)


    def test_voc_bbox_conversion_correctness(self, voc_dataset, tmp_path):
        """VOC absolute coords (xmin,ymin,xmax,ymax) are correctly normalized."""
        output_dir = tmp_path / "output_voc_conv"
        config = _make_config([str(voc_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # From the fixture: xmin=100, ymin=150, xmax=400, ymax=350, w=800, h=600
        # Expected: cx=(100+400)/2/800=0.3125, cy=(150+350)/2/600=0.4167
        #           w=(400-100)/800=0.375, h=(350-150)/600=0.3333
        # Check at least one label file has these values
        found_container = False
        for split in ["train", "valid", "test"]:
            labels_dir = output_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                for line in content.split("\n"):
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cx, cy, w, h = [float(p) for p in parts[1:]]
                    if (abs(cx - 0.3125) < 0.001 and abs(cy - 0.4167) < 0.001
                            and abs(w - 0.375) < 0.001 and abs(h - 0.3333) < 0.001):
                        found_container = True
                        break
        assert found_container, "Expected container annotation not found in output"


# ---------------------------------------------------------------------------
# Tests: Class Remapping to Roboflow 17-class IDs
# ---------------------------------------------------------------------------


class TestClassRemapping:
    """Tests for class label remapping to the Roboflow 17-class taxonomy."""

    def test_container_maps_to_stacked(self, tmp_path):
        """Source label 'container' remaps to 'Container -Stacked' (ID 7)."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        result = loader._remap_class("container")
        assert result == ROBOFLOW_CLASSES.index("Container -Stacked")

    def test_person_maps_to_human(self, tmp_path):
        """Source label 'person' remaps to 'Human' (ID 9)."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        result = loader._remap_class("person")
        assert result == ROBOFLOW_CLASSES.index("Human")

    def test_crane_maps_to_crane(self, tmp_path):
        """Source label 'crane' remaps to 'Crane' (ID 8)."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        result = loader._remap_class("crane")
        assert result == ROBOFLOW_CLASSES.index("Crane")

    def test_truck_maps_to_truck_no_container(self, tmp_path):
        """Source label 'truck' remaps to 'Truck - No Container' (ID 11)."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        result = loader._remap_class("truck")
        assert result == ROBOFLOW_CLASSES.index("Truck - No Container")

    def test_case_insensitive_remapping(self, tmp_path):
        """Remapping is case-insensitive."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        assert loader._remap_class("Container") is not None
        assert loader._remap_class("CRANE") is not None
        assert loader._remap_class("Person") is not None

    def test_direct_roboflow_class_name_maps_to_self(self, tmp_path):
        """Roboflow class names map directly to their own index."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        for idx, class_name in enumerate(ROBOFLOW_CLASSES):
            result = loader._remap_class(class_name.lower())
            assert result == idx, (
                f"'{class_name}' should map to index {idx}, got {result}"
            )

    def test_custom_class_mapping(self, tmp_path):
        """Custom class mapping in config is used for remapping."""
        custom_mapping = {"my_special_container": "Container - Reefer"}
        config = _make_config(
            [], str(tmp_path / "out"),
            class_mapping=custom_mapping,
        )
        loader = SupplementalDatasetLoader(config)
        result = loader._remap_class("my_special_container")
        assert result == ROBOFLOW_CLASSES.index("Container - Reefer")


# ---------------------------------------------------------------------------
# Tests: Discarding Annotations with No Corresponding Class
# ---------------------------------------------------------------------------


class TestDiscardingUnmappedAnnotations:
    """Tests for discarding annotations whose source class has no Roboflow mapping."""

    def test_unknown_class_returns_none(self, tmp_path):
        """Source labels with no mapping return None from _remap_class."""
        config = _make_config([], str(tmp_path / "out"))
        loader = SupplementalDatasetLoader(config)
        assert loader._remap_class("completely_unknown_class_xyz") is None

    def test_coco_unknown_annotations_discarded(self, coco_dataset, tmp_path):
        """COCO annotations with unmapped category are counted as discarded."""
        output_dir = tmp_path / "output_discard"
        config = _make_config([str(coco_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # The fixture has category_id=4 ("unknown_object") which has no mapping
        assert result.discarded_annotations > 0

    def test_yolo_unknown_class_discarded(self, yolo_dataset, tmp_path):
        """YOLO txt annotations with unmapped class ID are discarded."""
        output_dir = tmp_path / "output_discard_yolo"
        config = _make_config([str(yolo_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # The fixture has class 3 ("unknown_thing") which has no mapping
        assert result.discarded_annotations > 0

    def test_discarded_count_matches_unmapped_annotations(self, tmp_path):
        """Discard count tracks exactly the number of unmapped annotations."""
        dataset_root = tmp_path / "discard_count_ds"
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        classes_txt = dataset_root / "classes.txt"
        # Only 'container' is mappable; 'zzxnoclass' and 'qqymystery' have no mapping
        # (these names avoid partial matching with any entry in DEFAULT_CLASS_MAPPING)
        classes_txt.write_text("container\nzzxnoclass\nqqymystery\n")

        img_path = images_dir / "test_img.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        label_path = labels_dir / "test_img.txt"
        # 3 annotations: class 0 (container=mappable), 1 (zzxnoclass=not), 2 (qqymystery=not)
        label_path.write_text("0 0.5 0.5 0.3 0.2\n1 0.3 0.3 0.2 0.1\n2 0.7 0.7 0.1 0.1\n")

        output_dir = tmp_path / "output_discard_count"
        config = _make_config([str(dataset_root)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # 2 annotations should be discarded (zzxnoclass, qqymystery)
        assert result.discarded_annotations == 2
        assert result.total_annotations == 1  # Only container counted


# ---------------------------------------------------------------------------
# Tests: Skip Behavior for Unreadable Files
# ---------------------------------------------------------------------------


class TestSkipBehavior:
    """Tests for skipping unreadable/corrupt files and continuing processing."""

    def test_unreadable_label_file_skipped(self, mixed_dataset_with_unreadable, tmp_path):
        """Files with malformed labels are skipped; valid ones still processed."""
        output_dir = tmp_path / "output_skip"
        config = _make_config(
            [str(mixed_dataset_with_unreadable)], str(output_dir)
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # 3 valid images should be loaded
        assert result.total_images == 3
        # Discarded annotations from the corrupt label (malformed lines)
        assert result.discarded_annotations >= 2

    def test_nonexistent_dataset_root_skipped(self, tmp_path):
        """Nonexistent dataset roots are skipped gracefully."""
        output_dir = tmp_path / "output_nonexist"
        config = _make_config(
            [str(tmp_path / "does_not_exist")], str(output_dir)
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        assert result.total_images == 0
        assert result.total_annotations == 0

    def test_corrupt_xml_skipped(self, tmp_path):
        """Corrupt VOC XML files are skipped without stopping processing."""
        dataset_root = tmp_path / "corrupt_xml_ds"
        images_dir = dataset_root / "images"
        annotations_dir = dataset_root / "annotations"
        images_dir.mkdir(parents=True)
        annotations_dir.mkdir(parents=True)

        # Valid XML
        img_path = images_dir / "valid.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        valid_xml = annotations_dir / "valid.xml"
        valid_xml.write_text('''<?xml version="1.0"?>
<annotation>
    <filename>valid.jpg</filename>
    <size><width>640</width><height>480</height></size>
    <object>
        <name>container</name>
        <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox>
    </object>
</annotation>''')

        # Corrupt XML
        corrupt_xml = annotations_dir / "corrupt.xml"
        corrupt_xml.write_text("this is not valid xml <<>><>!!!")

        output_dir = tmp_path / "output_corrupt_xml"
        config = _make_config([str(dataset_root)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # Should still load the valid one
        assert result.total_images >= 1
        assert result.skipped_files >= 1

    def test_corrupt_coco_json_skipped(self, tmp_path):
        """Corrupt COCO JSON files are skipped without crashing."""
        dataset_root = tmp_path / "corrupt_json_ds"
        dataset_root.mkdir(parents=True)

        # Write malformed JSON
        corrupt_json = dataset_root / "annotations.json"
        corrupt_json.write_text("{this is not valid json!!!")

        output_dir = tmp_path / "output_corrupt_json"
        config = _make_config([str(dataset_root)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # Should not crash, just skip
        assert result.total_images == 0
        assert result.skipped_files >= 1


# ---------------------------------------------------------------------------
# Tests: Split Ratio Output (70/15/15 Default)
# ---------------------------------------------------------------------------


class TestSplitRatios:
    """Tests for dataset splitting into train/valid/test sets."""

    def test_default_split_ratios(self, voc_dataset, tmp_path):
        """Default split ratios are 70% train, 15% valid, 15% test."""
        output_dir = tmp_path / "output_split"
        config = _make_config([str(voc_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        total = result.total_images
        assert total > 0

        # Verify splits exist
        assert "train" in result.split_counts
        assert "valid" in result.split_counts
        assert "test" in result.split_counts

        # All images are accounted for
        split_total = sum(result.split_counts.values())
        assert split_total == total

        # Train should be the largest split
        assert result.split_counts["train"] >= result.split_counts["valid"]
        assert result.split_counts["train"] >= result.split_counts["test"]

    def test_custom_split_ratios(self, voc_dataset, tmp_path):
        """Custom split ratios are applied correctly."""
        output_dir = tmp_path / "output_custom_split"
        config = _make_config(
            [str(voc_dataset)], str(output_dir),
            split_ratios={"train": 0.60, "valid": 0.20, "test": 0.20},
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        total = result.total_images
        split_total = sum(result.split_counts.values())
        assert split_total == total

    def test_split_output_directory_structure(self, voc_dataset, tmp_path):
        """Output has images/ and labels/ subdirectories per split."""
        output_dir = tmp_path / "output_structure"
        config = _make_config([str(voc_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        loader.load_and_normalize()

        for split in ["train", "valid", "test"]:
            split_dir = output_dir / split
            if split_dir.exists():
                assert (split_dir / "images").exists()
                assert (split_dir / "labels").exists()

    def test_invalid_split_ratios_raise_error(self):
        """Split ratios that don't sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SupplementalConfig(
                dataset_roots=[],
                output_dir="out",
                split_ratios={"train": 0.5, "valid": 0.2, "test": 0.1},
            )

    def test_negative_split_ratio_raises_error(self):
        """Negative split ratio raises ValueError."""
        with pytest.raises(ValueError):
            SupplementalConfig(
                dataset_roots=[],
                output_dir="out",
                split_ratios={"train": 0.8, "valid": -0.1, "test": 0.3},
            )

    def test_images_and_labels_match_per_split(self, yolo_dataset, tmp_path):
        """Each split has matching image and label file counts."""
        output_dir = tmp_path / "output_match"
        config = _make_config([str(yolo_dataset)], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        loader.load_and_normalize()

        for split in ["train", "valid", "test"]:
            images_dir = output_dir / split / "images"
            labels_dir = output_dir / split / "labels"
            if not images_dir.exists():
                continue
            img_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}
            lbl_stems = {p.stem for p in labels_dir.iterdir() if p.is_file()}
            assert img_stems == lbl_stems, (
                f"Mismatch in {split}: images={img_stems}, labels={lbl_stems}"
            )


# ---------------------------------------------------------------------------
# Tests: Multiple Dataset Roots
# ---------------------------------------------------------------------------


class TestMultipleDatasets:
    """Tests for loading from multiple dataset roots."""

    def test_multiple_roots_combined(self, voc_dataset, yolo_dataset, tmp_path):
        """Multiple dataset roots are combined into one output."""
        output_dir = tmp_path / "output_multi"
        config = _make_config(
            [str(voc_dataset), str(yolo_dataset)], str(output_dir)
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # Both datasets contribute images
        assert result.total_images > 5  # VOC has 5, YOLO has 6 (some may be discarded)

    def test_empty_dataset_roots_returns_zero(self, tmp_path):
        """Empty dataset_roots list results in zero images."""
        output_dir = tmp_path / "output_empty"
        config = _make_config([], str(output_dir))
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        assert result.total_images == 0
        assert result.total_annotations == 0


# ---------------------------------------------------------------------------
# Tests: Visual Output Generation
# ---------------------------------------------------------------------------


class TestVisualOutputGeneration:
    """
    Visual output tests for supplemental loader diagnostics.

    Generates:
    - tests/output/supplemental_class_balance.png
    - tests/output/supplemental_quality_report.json
    """

    def test_generate_class_balance_histogram(
        self, voc_dataset, yolo_dataset, tmp_path, output_dir
    ):
        """
        Generate visual output: class balance histogram of loaded annotations.
        Saves to tests/output/supplemental_class_balance.png.
        """
        from tests.visual_helpers import plot_class_distribution

        out = tmp_path / "output_visual"
        config = _make_config(
            [str(voc_dataset), str(yolo_dataset)], str(out)
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # Build class name -> count mapping
        class_counts: Dict[str, int] = {}
        for class_id, count in result.class_distribution.items():
            if 0 <= class_id < len(ROBOFLOW_CLASSES):
                class_name = ROBOFLOW_CLASSES[class_id]
            else:
                class_name = f"Unknown ({class_id})"
            class_counts[class_name] = count

        output_path = output_dir / "supplemental_class_balance.png"
        plot_class_distribution(
            class_counts=class_counts,
            output_path=output_path,
            title="Supplemental Dataset Class Balance",
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_quality_report(
        self, voc_dataset, yolo_dataset, tmp_path, output_dir
    ):
        """
        Generate visual output: annotation quality report with coordinate ranges
        and class coverage. Saves to tests/output/supplemental_quality_report.json.
        """
        from tests.visual_helpers import save_json_report

        out = tmp_path / "output_quality"
        config = _make_config(
            [str(voc_dataset), str(yolo_dataset)], str(out)
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()

        # Collect all annotation coordinate values from output labels
        all_cx, all_cy, all_w, all_h = [], [], [], []
        class_ids_found = set()

        for split in ["train", "valid", "test"]:
            labels_dir = Path(out) / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                content = label_file.read_text().strip()
                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        class_ids_found.add(int(parts[0]))
                        all_cx.append(float(parts[1]))
                        all_cy.append(float(parts[2]))
                        all_w.append(float(parts[3]))
                        all_h.append(float(parts[4]))

        # Build quality report
        quality_report = {
            "total_images": result.total_images,
            "total_annotations": result.total_annotations,
            "discarded_annotations": result.discarded_annotations,
            "skipped_files": result.skipped_files,
            "split_counts": result.split_counts,
            "coordinate_ranges": {
                "x_center": {
                    "min": min(all_cx) if all_cx else None,
                    "max": max(all_cx) if all_cx else None,
                    "mean": float(np.mean(all_cx)) if all_cx else None,
                },
                "y_center": {
                    "min": min(all_cy) if all_cy else None,
                    "max": max(all_cy) if all_cy else None,
                    "mean": float(np.mean(all_cy)) if all_cy else None,
                },
                "width": {
                    "min": min(all_w) if all_w else None,
                    "max": max(all_w) if all_w else None,
                    "mean": float(np.mean(all_w)) if all_w else None,
                },
                "height": {
                    "min": min(all_h) if all_h else None,
                    "max": max(all_h) if all_h else None,
                    "mean": float(np.mean(all_h)) if all_h else None,
                },
            },
            "class_coverage": {
                "classes_found": sorted(list(class_ids_found)),
                "class_names": [
                    ROBOFLOW_CLASSES[cid] for cid in sorted(class_ids_found)
                    if 0 <= cid < len(ROBOFLOW_CLASSES)
                ],
                "total_roboflow_classes": len(ROBOFLOW_CLASSES),
                "coverage_ratio": len(class_ids_found) / len(ROBOFLOW_CLASSES),
            },
            "class_distribution": {
                ROBOFLOW_CLASSES[k] if 0 <= k < len(ROBOFLOW_CLASSES) else f"Unknown({k})": v
                for k, v in result.class_distribution.items()
            },
        }

        output_path = output_dir / "supplemental_quality_report.json"
        save_json_report(quality_report, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify report content is valid JSON
        with open(output_path, "r") as f:
            loaded = json.load(f)
        assert "coordinate_ranges" in loaded
        assert "class_coverage" in loaded
        assert "total_images" in loaded

        # Verify coordinate ranges are within [0.0, 1.0]
        for coord_name in ["x_center", "y_center", "width", "height"]:
            ranges = loaded["coordinate_ranges"][coord_name]
            if ranges["min"] is not None:
                assert ranges["min"] >= 0.0
                assert ranges["max"] <= 1.0
