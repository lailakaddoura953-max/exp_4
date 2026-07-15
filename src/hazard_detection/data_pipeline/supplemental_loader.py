"""
Supplemental Dataset Loader for the Hazard Detection System.

Loads container imagery from external open-source datasets, normalizes
annotations to YOLO format, and remaps class labels to the Roboflow
17-class taxonomy defined in `roboflow data/data.yaml`.

Supports multiple source annotation formats:
- VOC XML (Pascal VOC bounding box annotations)
- COCO JSON (MS COCO format with bounding boxes)
- Plain text (YOLO-style txt with class_id x y w h per line)

Requirements covered:
- 11.1: Load container imagery with bounding box annotations (min 50 images)
- 11.2: Normalize to YOLO format, remap to Roboflow class IDs
- 11.3: Output in images/ + labels/ directory structure per split
- 11.4: Configurable dataset root paths and split ratios (default 70/15/15)
- 11.5: Skip unreadable files, log warning, continue processing
- 11.6: Discard annotations with no Roboflow class, log count
"""

import json
import logging
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hazard_detection.diagnostics import get_logger

logger = get_logger("supplemental_loader")


# Roboflow 17-class taxonomy (from roboflow data/data.yaml)
ROBOFLOW_CLASSES = [
    "Boat - With Cargo",
    "Container - Misaligned",
    "Container - Open",
    "Container - Picked",
    "Container - Reefer",
    "Container - Water Drop",
    "Container -Separate",
    "Container -Stacked",
    "Crane",
    "Human",
    "Human - No Safety Clothes",
    "Truck - No Container",
    "Truck - With Container",
    "Vehicle",
    "Yard - Dropoff zone",
    "Yard - No People",
    "Yard - Operation Zone",
]

# Default class mapping from common external dataset labels to Roboflow classes
DEFAULT_CLASS_MAPPING = {
    # Container-related mappings
    "container": "Container -Stacked",
    "shipping_container": "Container -Stacked",
    "shipping container": "Container -Stacked",
    "cargo_container": "Container -Stacked",
    "cargo container": "Container -Stacked",
    "freight_container": "Container -Stacked",
    "freight container": "Container -Stacked",
    "container_stacked": "Container -Stacked",
    "stacked_container": "Container -Stacked",
    "container_separate": "Container -Separate",
    "separate_container": "Container -Separate",
    "reefer": "Container - Reefer",
    "reefer_container": "Container - Reefer",
    "refrigerated_container": "Container - Reefer",
    "open_container": "Container - Open",
    "container_open": "Container - Open",
    "misaligned_container": "Container - Misaligned",
    "container_misaligned": "Container - Misaligned",
    "picked_container": "Container - Picked",
    "container_picked": "Container - Picked",
    # Vehicle mappings
    "truck": "Truck - No Container",
    "truck_with_container": "Truck - With Container",
    "vehicle": "Vehicle",
    "car": "Vehicle",
    # Human mappings
    "person": "Human",
    "human": "Human",
    "worker": "Human",
    "pedestrian": "Human",
    # Equipment mappings
    "crane": "Crane",
    "gantry_crane": "Crane",
    "boat": "Boat - With Cargo",
    "ship": "Boat - With Cargo",
    "vessel": "Boat - With Cargo",
}


@dataclass
class SupplementalConfig:
    """Configuration for the Supplemental Dataset Loader."""

    dataset_roots: List[str] = field(default_factory=list)
    output_dir: str = "supplemental_output"
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "valid": 0.15, "test": 0.15}
    )
    class_mapping: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_MAPPING))
    roboflow_data_yaml: str = "roboflow data/data.yaml"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    )
    random_seed: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration."""
        # Validate split ratios
        total = sum(self.split_ratios.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(ratios: {self.split_ratios})"
            )
        for split_name, ratio in self.split_ratios.items():
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(
                    f"Split ratio for '{split_name}' must be in [0.0, 1.0], "
                    f"got {ratio}"
                )


@dataclass
class AnnotationEntry:
    """A single annotation for one object in an image."""

    class_id: int  # Roboflow class index (0-16)
    x_center: float  # Normalized [0.0, 1.0]
    y_center: float  # Normalized [0.0, 1.0]
    width: float  # Normalized [0.0, 1.0]
    height: float  # Normalized [0.0, 1.0]

    def to_yolo_line(self) -> str:
        """Format as YOLO annotation line: class_id cx cy w h"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def is_valid(self) -> bool:
        """Check all coordinate values are in [0.0, 1.0]."""
        return all(
            0.0 <= v <= 1.0
            for v in [self.x_center, self.y_center, self.width, self.height]
        )


@dataclass
class DatasetOutput:
    """Result of loading and normalizing a supplemental dataset."""

    output_dir: str
    total_images: int
    total_annotations: int
    discarded_annotations: int
    split_counts: Dict[str, int]
    class_distribution: Dict[int, int]
    skipped_files: int


class SupplementalDatasetLoader:
    """
    Loads and normalizes external container datasets to YOLO format.

    Supports loading from multiple source annotation formats (VOC XML,
    COCO JSON, plain text YOLO-style) and normalizes them all to YOLO
    txt format with class IDs remapped to the Roboflow 17-class taxonomy.

    Usage:
        config = SupplementalConfig(
            dataset_roots=["path/to/dataset1", "path/to/dataset2"],
            output_dir="supplemental_output",
        )
        loader = SupplementalDatasetLoader(config)
        result = loader.load_and_normalize()
    """

    def __init__(self, config: SupplementalConfig):
        """
        Initialize the loader.

        Args:
            config: SupplementalConfig with dataset roots, output dir,
                    split ratios, and class mapping.
        """
        self.config = config
        self._roboflow_classes = self._load_roboflow_classes()
        self._class_mapping = self._build_class_mapping()
        self._discarded_count = 0
        self._skipped_files = 0
        self._total_annotations = 0

        if config.random_seed is not None:
            random.seed(config.random_seed)

        logger.info(
            "SupplementalDatasetLoader initialized",
            extra={
                "extra_data": {
                    "dataset_roots": config.dataset_roots,
                    "output_dir": config.output_dir,
                    "split_ratios": config.split_ratios,
                    "roboflow_classes_count": len(self._roboflow_classes),
                }
            },
        )

    def _load_roboflow_classes(self) -> List[str]:
        """
        Load Roboflow class names from data.yaml.

        Falls back to the hardcoded ROBOFLOW_CLASSES if file not available.
        """
        yaml_path = Path(self.config.roboflow_data_yaml)
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "names" in data:
                    classes = data["names"]
                    logger.info(
                        f"Loaded {len(classes)} classes from {yaml_path}",
                        extra={"extra_data": {"classes": classes}},
                    )
                    return classes
            except (yaml.YAMLError, OSError) as e:
                logger.warning(
                    f"Failed to load roboflow data.yaml from {yaml_path}: {e}. "
                    "Using hardcoded class list.",
                )
        else:
            logger.info(
                f"Roboflow data.yaml not found at {yaml_path}, "
                "using hardcoded class list.",
            )
        return list(ROBOFLOW_CLASSES)

    def _build_class_mapping(self) -> Dict[str, int]:
        """
        Build a mapping from source class labels to Roboflow class IDs.

        Returns:
            Dict mapping lowercase source label -> Roboflow class index.
        """
        mapping = {}
        for source_label, roboflow_label in self.config.class_mapping.items():
            if roboflow_label in self._roboflow_classes:
                class_id = self._roboflow_classes.index(roboflow_label)
                mapping[source_label.lower()] = class_id
            else:
                logger.warning(
                    f"Class mapping target '{roboflow_label}' not found in "
                    f"Roboflow classes. Skipping mapping for '{source_label}'.",
                )
        # Also add direct Roboflow class names mapped to themselves
        for idx, class_name in enumerate(self._roboflow_classes):
            mapping[class_name.lower()] = idx
        return mapping

    def load_and_normalize(self) -> DatasetOutput:
        """
        Load all configured datasets, normalize annotations, and output
        in YOLO format with images/ + labels/ directory structure per split.

        Returns:
            DatasetOutput with statistics about the loaded data.
        """
        self._discarded_count = 0
        self._skipped_files = 0
        self._total_annotations = 0

        # Collect all valid (image_path, annotations) pairs
        all_samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        for dataset_root in self.config.dataset_roots:
            root_path = Path(dataset_root)
            if not root_path.exists():
                logger.warning(
                    f"Dataset root does not exist: {dataset_root}. Skipping.",
                    extra={"extra_data": {"dataset_root": dataset_root}},
                )
                continue

            samples = self._load_dataset(root_path)
            all_samples.extend(samples)
            logger.info(
                f"Loaded {len(samples)} samples from {dataset_root}",
                extra={
                    "extra_data": {
                        "dataset_root": dataset_root,
                        "sample_count": len(samples),
                    }
                },
            )

        if not all_samples:
            logger.warning(
                "No valid samples found across all dataset roots.",
                extra={"extra_data": {"dataset_roots": self.config.dataset_roots}},
            )
            return DatasetOutput(
                output_dir=self.config.output_dir,
                total_images=0,
                total_annotations=0,
                discarded_annotations=self._discarded_count,
                split_counts={},
                class_distribution={},
                skipped_files=self._skipped_files,
            )

        # Shuffle and split
        random.shuffle(all_samples)
        splits = self._split_samples(all_samples)

        # Write output
        output_dir = Path(self.config.output_dir)
        class_distribution: Dict[int, int] = {}
        split_counts: Dict[str, int] = {}

        for split_name, split_samples in splits.items():
            count, dist = self._write_split(output_dir, split_name, split_samples)
            split_counts[split_name] = count
            for cls_id, cls_count in dist.items():
                class_distribution[cls_id] = class_distribution.get(cls_id, 0) + cls_count

        # Log final statistics
        logger.info(
            "Supplemental dataset loading complete",
            extra={
                "extra_data": {
                    "total_images": len(all_samples),
                    "total_annotations": self._total_annotations,
                    "discarded_annotations": self._discarded_count,
                    "skipped_files": self._skipped_files,
                    "split_counts": split_counts,
                    "class_distribution": class_distribution,
                }
            },
        )

        if self._discarded_count > 0:
            logger.info(
                f"Discarded {self._discarded_count} annotations with no "
                f"corresponding Roboflow class.",
            )

        return DatasetOutput(
            output_dir=self.config.output_dir,
            total_images=len(all_samples),
            total_annotations=self._total_annotations,
            discarded_annotations=self._discarded_count,
            split_counts=split_counts,
            class_distribution=class_distribution,
            skipped_files=self._skipped_files,
        )

    def _load_dataset(self, root_path: Path) -> List[Tuple[Path, List[AnnotationEntry]]]:
        """
        Load a single dataset from root path, detecting annotation format.

        Supports:
        - COCO JSON (annotations/*.json or *.json in root with 'images' and 'annotations' keys)
        - VOC XML (annotations/*.xml or alongside images)
        - YOLO txt (labels/*.txt alongside images/*.ext)

        Returns:
            List of (image_path, annotations) tuples.
        """
        samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        # Try COCO JSON format first
        coco_samples = self._try_load_coco(root_path)
        if coco_samples:
            samples.extend(coco_samples)
            return samples

        # Try VOC XML format
        voc_samples = self._try_load_voc(root_path)
        if voc_samples:
            samples.extend(voc_samples)
            return samples

        # Try YOLO txt format
        yolo_samples = self._try_load_yolo_txt(root_path)
        if yolo_samples:
            samples.extend(yolo_samples)
            return samples

        # If no format detected, try scanning for image/annotation pairs
        logger.warning(
            f"Could not detect annotation format in {root_path}. "
            "Trying recursive scan for image-annotation pairs.",
        )
        scan_samples = self._scan_for_pairs(root_path)
        samples.extend(scan_samples)

        return samples

    def _try_load_coco(self, root_path: Path) -> List[Tuple[Path, List[AnnotationEntry]]]:
        """
        Try to load dataset as COCO JSON format.

        Looks for JSON files with 'images' and 'annotations' keys.
        """
        samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        # Look for COCO annotation files
        json_candidates = list(root_path.glob("*.json")) + list(
            root_path.glob("annotations/*.json")
        )

        for json_path in json_candidates:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to read JSON file {json_path}: {e}",
                    extra={"extra_data": {"file": str(json_path)}},
                )
                self._skipped_files += 1
                continue

            # Validate COCO structure
            if not isinstance(coco_data, dict):
                continue
            if "images" not in coco_data or "annotations" not in coco_data:
                continue

            # Build category mapping
            categories = {}
            if "categories" in coco_data:
                for cat in coco_data["categories"]:
                    categories[cat["id"]] = cat.get("name", "")

            # Build image ID -> image info mapping
            image_map: Dict[int, Dict[str, Any]] = {}
            for img_info in coco_data["images"]:
                image_map[img_info["id"]] = img_info

            # Group annotations by image
            ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in ann_by_image:
                    ann_by_image[img_id] = []
                ann_by_image[img_id].append(ann)

            # Process each image
            for img_id, img_info in image_map.items():
                file_name = img_info.get("file_name", "")
                img_width = img_info.get("width", 0)
                img_height = img_info.get("height", 0)

                if img_width <= 0 or img_height <= 0:
                    logger.warning(
                        f"Invalid image dimensions for {file_name}: "
                        f"{img_width}x{img_height}. Skipping.",
                    )
                    self._skipped_files += 1
                    continue

                # Find image file
                img_path = self._find_image_file(root_path, file_name)
                if img_path is None:
                    self._skipped_files += 1
                    continue

                # Parse annotations for this image
                annotations = []
                image_anns = ann_by_image.get(img_id, [])
                for ann in image_anns:
                    bbox = ann.get("bbox")  # COCO format: [x, y, width, height] (absolute)
                    cat_id = ann.get("category_id")

                    if bbox is None or cat_id is None:
                        continue

                    # Get source class name
                    source_class = categories.get(cat_id, "").lower()

                    # Remap to Roboflow class ID
                    roboflow_id = self._remap_class(source_class)
                    if roboflow_id is None:
                        self._discarded_count += 1
                        continue

                    # Convert COCO bbox to YOLO normalized format
                    x, y, w, h = bbox
                    x_center = (x + w / 2.0) / img_width
                    y_center = (y + h / 2.0) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height

                    entry = AnnotationEntry(
                        class_id=roboflow_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=norm_w,
                        height=norm_h,
                    )

                    if entry.is_valid():
                        annotations.append(entry)
                        self._total_annotations += 1
                    else:
                        self._discarded_count += 1

                if annotations:
                    samples.append((img_path, annotations))

            logger.info(
                f"Loaded {len(samples)} samples from COCO JSON: {json_path.name}",
                extra={"extra_data": {"json_file": str(json_path)}},
            )

        return samples

    def _try_load_voc(self, root_path: Path) -> List[Tuple[Path, List[AnnotationEntry]]]:
        """
        Try to load dataset as Pascal VOC XML format.

        Looks for XML annotation files in annotations/, Annotations/, or alongside images.
        """
        samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        # Find XML files (deduplicate for case-insensitive filesystems)
        xml_dirs = [
            root_path / "annotations",
            root_path / "Annotations",
            root_path,
        ]

        xml_files: List[Path] = []
        seen_paths: set = set()
        for xml_dir in xml_dirs:
            if xml_dir.exists():
                for xml_file in xml_dir.glob("*.xml"):
                    resolved = xml_file.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        xml_files.append(xml_file)

        if not xml_files:
            return samples

        for xml_path in xml_files:
            try:
                tree = ET.parse(str(xml_path))
                root = tree.getroot()
            except (ET.ParseError, OSError) as e:
                logger.warning(
                    f"Failed to parse VOC XML {xml_path}: {e}",
                    extra={"extra_data": {"file": str(xml_path)}},
                )
                self._skipped_files += 1
                continue

            # Get image filename and dimensions
            filename_elem = root.find("filename")
            size_elem = root.find("size")

            if filename_elem is None or size_elem is None:
                self._skipped_files += 1
                continue

            file_name = filename_elem.text or ""
            width_elem = size_elem.find("width")
            height_elem = size_elem.find("height")

            if width_elem is None or height_elem is None:
                self._skipped_files += 1
                continue

            try:
                img_width = int(width_elem.text or "0")
                img_height = int(height_elem.text or "0")
            except ValueError:
                self._skipped_files += 1
                continue

            if img_width <= 0 or img_height <= 0:
                self._skipped_files += 1
                continue

            # Find the image file
            img_path = self._find_image_file(root_path, file_name)
            if img_path is None:
                self._skipped_files += 1
                continue

            # Parse object annotations
            annotations = []
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                bndbox = obj.find("bndbox")

                if name_elem is None or bndbox is None:
                    continue

                source_class = (name_elem.text or "").lower()

                # Remap to Roboflow class ID
                roboflow_id = self._remap_class(source_class)
                if roboflow_id is None:
                    self._discarded_count += 1
                    continue

                # Parse bounding box (VOC format: xmin, ymin, xmax, ymax absolute)
                try:
                    xmin = float(bndbox.find("xmin").text or "0")
                    ymin = float(bndbox.find("ymin").text or "0")
                    xmax = float(bndbox.find("xmax").text or "0")
                    ymax = float(bndbox.find("ymax").text or "0")
                except (ValueError, AttributeError):
                    self._discarded_count += 1
                    continue

                # Convert to YOLO normalized format
                x_center = ((xmin + xmax) / 2.0) / img_width
                y_center = ((ymin + ymax) / 2.0) / img_height
                norm_w = (xmax - xmin) / img_width
                norm_h = (ymax - ymin) / img_height

                entry = AnnotationEntry(
                    class_id=roboflow_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=norm_w,
                    height=norm_h,
                )

                if entry.is_valid():
                    annotations.append(entry)
                    self._total_annotations += 1
                else:
                    self._discarded_count += 1

            if annotations:
                samples.append((img_path, annotations))

        if samples:
            logger.info(
                f"Loaded {len(samples)} samples from VOC XML in {root_path}",
            )

        return samples

    def _try_load_yolo_txt(self, root_path: Path) -> List[Tuple[Path, List[AnnotationEntry]]]:
        """
        Try to load dataset in YOLO txt format.

        Expects images/ and labels/ directories with matching filenames,
        or images and .txt labels side-by-side.
        """
        samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        # Look for images/ and labels/ directories
        images_dir = None
        labels_dir = None

        for candidate_img in ["images", "Images", "img"]:
            candidate = root_path / candidate_img
            if candidate.exists():
                images_dir = candidate
                break

        for candidate_lbl in ["labels", "Labels", "label"]:
            candidate = root_path / candidate_lbl
            if candidate.exists():
                labels_dir = candidate
                break

        if images_dir is None or labels_dir is None:
            return samples

        # Also check for a classes.txt or data.yaml for class name mapping
        source_classes = self._load_source_class_names(root_path)

        # Find image files (deduplicate for case-insensitive filesystems)
        image_files = []
        seen_img_paths: set = set()
        for ext in self.config.image_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for img_file in images_dir.glob(pattern):
                    resolved = img_file.resolve()
                    if resolved not in seen_img_paths:
                        seen_img_paths.add(resolved)
                        image_files.append(img_file)

        for img_path in image_files:
            # Find corresponding label file
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError as e:
                logger.warning(
                    f"Failed to read label file {label_path}: {e}",
                    extra={"extra_data": {"file": str(label_path)}},
                )
                self._skipped_files += 1
                continue

            annotations = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    self._discarded_count += 1
                    continue

                try:
                    source_class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except (ValueError, IndexError):
                    self._discarded_count += 1
                    continue

                # Remap class: use source class names if available
                if source_classes and source_class_id < len(source_classes):
                    source_class_name = source_classes[source_class_id].lower()
                    roboflow_id = self._remap_class(source_class_name)
                else:
                    # Try direct ID mapping (if source uses same IDs)
                    if 0 <= source_class_id < len(self._roboflow_classes):
                        roboflow_id = source_class_id
                    else:
                        roboflow_id = None

                if roboflow_id is None:
                    self._discarded_count += 1
                    continue

                entry = AnnotationEntry(
                    class_id=roboflow_id,
                    x_center=cx,
                    y_center=cy,
                    width=w,
                    height=h,
                )

                if entry.is_valid():
                    annotations.append(entry)
                    self._total_annotations += 1
                else:
                    self._discarded_count += 1

            if annotations:
                samples.append((img_path, annotations))

        if samples:
            logger.info(
                f"Loaded {len(samples)} samples from YOLO txt in {root_path}",
            )

        return samples

    def _scan_for_pairs(self, root_path: Path) -> List[Tuple[Path, List[AnnotationEntry]]]:
        """
        Recursively scan for image-annotation pairs when format is not detected.

        Looks for image files with corresponding .txt or .xml annotation files.
        """
        samples: List[Tuple[Path, List[AnnotationEntry]]] = []

        image_files = []
        seen_paths: set = set()
        for ext in self.config.image_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for img_file in root_path.rglob(pattern):
                    resolved = img_file.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        image_files.append(img_file)

        source_classes = self._load_source_class_names(root_path)

        for img_path in image_files:
            # Look for .txt label file
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                annotations = self._parse_yolo_txt_file(txt_path, source_classes)
                if annotations:
                    samples.append((img_path, annotations))
                continue

            # Look for .xml annotation file
            xml_path = img_path.with_suffix(".xml")
            if xml_path.exists():
                annotations = self._parse_single_voc_xml(xml_path)
                if annotations:
                    samples.append((img_path, annotations))

        return samples

    def _parse_yolo_txt_file(
        self, label_path: Path, source_classes: List[str]
    ) -> List[AnnotationEntry]:
        """Parse a single YOLO txt label file."""
        annotations = []
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            self._skipped_files += 1
            return annotations

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                self._discarded_count += 1
                continue
            try:
                source_class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except (ValueError, IndexError):
                self._discarded_count += 1
                continue

            if source_classes and source_class_id < len(source_classes):
                source_class_name = source_classes[source_class_id].lower()
                roboflow_id = self._remap_class(source_class_name)
            else:
                if 0 <= source_class_id < len(self._roboflow_classes):
                    roboflow_id = source_class_id
                else:
                    roboflow_id = None

            if roboflow_id is None:
                self._discarded_count += 1
                continue

            entry = AnnotationEntry(
                class_id=roboflow_id, x_center=cx, y_center=cy, width=w, height=h
            )
            if entry.is_valid():
                annotations.append(entry)
                self._total_annotations += 1
            else:
                self._discarded_count += 1

        return annotations

    def _parse_single_voc_xml(self, xml_path: Path) -> List[AnnotationEntry]:
        """Parse a single VOC XML file for annotations."""
        annotations = []
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except (ET.ParseError, OSError):
            self._skipped_files += 1
            return annotations

        size_elem = root.find("size")
        if size_elem is None:
            return annotations

        width_elem = size_elem.find("width")
        height_elem = size_elem.find("height")
        if width_elem is None or height_elem is None:
            return annotations

        try:
            img_width = int(width_elem.text or "0")
            img_height = int(height_elem.text or "0")
        except ValueError:
            return annotations

        if img_width <= 0 or img_height <= 0:
            return annotations

        for obj in root.findall("object"):
            name_elem = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_elem is None or bndbox is None:
                continue

            source_class = (name_elem.text or "").lower()
            roboflow_id = self._remap_class(source_class)
            if roboflow_id is None:
                self._discarded_count += 1
                continue

            try:
                xmin = float(bndbox.find("xmin").text or "0")
                ymin = float(bndbox.find("ymin").text or "0")
                xmax = float(bndbox.find("xmax").text or "0")
                ymax = float(bndbox.find("ymax").text or "0")
            except (ValueError, AttributeError):
                self._discarded_count += 1
                continue

            x_center = ((xmin + xmax) / 2.0) / img_width
            y_center = ((ymin + ymax) / 2.0) / img_height
            norm_w = (xmax - xmin) / img_width
            norm_h = (ymax - ymin) / img_height

            entry = AnnotationEntry(
                class_id=roboflow_id,
                x_center=x_center,
                y_center=y_center,
                width=norm_w,
                height=norm_h,
            )
            if entry.is_valid():
                annotations.append(entry)
                self._total_annotations += 1
            else:
                self._discarded_count += 1

        return annotations

    def _remap_class(self, source_class: str) -> Optional[int]:
        """
        Remap a source class label to a Roboflow class ID.

        Args:
            source_class: Lowercase source class name.

        Returns:
            Roboflow class ID (0-16) or None if no mapping exists.
        """
        source_class = source_class.strip().lower()

        # Direct match in mapping
        if source_class in self._class_mapping:
            roboflow_id = self._class_mapping[source_class]
            logger.debug(
                f"Class remapped: '{source_class}' -> {roboflow_id} "
                f"({self._roboflow_classes[roboflow_id]})",
            )
            return roboflow_id

        # Try partial matching (source class contains a known term)
        for key, class_id in self._class_mapping.items():
            if key in source_class or source_class in key:
                logger.debug(
                    f"Class remapped (partial): '{source_class}' -> {class_id} "
                    f"({self._roboflow_classes[class_id]}) via key '{key}'",
                )
                return class_id

        logger.debug(
            f"No class mapping found for '{source_class}'. Discarding.",
        )
        return None

    def _find_image_file(self, root_path: Path, file_name: str) -> Optional[Path]:
        """
        Find an image file relative to root path.

        Searches in common image directories.
        """
        # Try direct path
        candidates = [
            root_path / file_name,
            root_path / "images" / file_name,
            root_path / "Images" / file_name,
            root_path / "img" / file_name,
            root_path / "JPEGImages" / file_name,
        ]

        for candidate in candidates:
            if candidate.exists():
                # Validate it's a readable image by extension
                if candidate.suffix.lower() in self.config.image_extensions:
                    return candidate

        logger.debug(
            f"Image file not found: {file_name} in {root_path}",
        )
        return None

    def _load_source_class_names(self, root_path: Path) -> List[str]:
        """
        Load source class names from a classes.txt or data.yaml file.

        Returns:
            List of class names indexed by class ID, or empty list if not found.
        """
        # Try classes.txt
        classes_txt = root_path / "classes.txt"
        if classes_txt.exists():
            try:
                with open(classes_txt, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                return classes
            except OSError:
                pass

        # Try data.yaml
        data_yaml = root_path / "data.yaml"
        if data_yaml.exists():
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "names" in data:
                    return data["names"]
            except (yaml.YAMLError, OSError):
                pass

        return []

    def _split_samples(
        self, samples: List[Tuple[Path, List[AnnotationEntry]]]
    ) -> Dict[str, List[Tuple[Path, List[AnnotationEntry]]]]:
        """
        Split samples into train/valid/test sets based on configured ratios.

        Args:
            samples: List of (image_path, annotations) tuples (already shuffled).

        Returns:
            Dict mapping split names to their sample lists.
        """
        total = len(samples)
        splits: Dict[str, List[Tuple[Path, List[AnnotationEntry]]]] = {}
        offset = 0

        split_names = list(self.config.split_ratios.keys())
        for i, split_name in enumerate(split_names):
            ratio = self.config.split_ratios[split_name]
            if i == len(split_names) - 1:
                # Last split gets remaining samples to handle rounding
                split_count = total - offset
            else:
                split_count = int(total * ratio)

            splits[split_name] = samples[offset : offset + split_count]
            offset += split_count

        return splits

    def _write_split(
        self,
        output_dir: Path,
        split_name: str,
        samples: List[Tuple[Path, List[AnnotationEntry]]],
    ) -> Tuple[int, Dict[int, int]]:
        """
        Write a split's images and labels to the output directory.

        Creates:
            output_dir/split_name/images/
            output_dir/split_name/labels/

        Args:
            output_dir: Base output directory.
            split_name: Name of the split (train, valid, test).
            samples: List of (image_path, annotations) tuples.

        Returns:
            Tuple of (number of images written, class distribution dict).
        """
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        class_dist: Dict[int, int] = {}
        written_count = 0

        for img_path, annotations in samples:
            # Copy image
            dest_img = images_dir / img_path.name

            # Handle duplicate filenames by appending a counter
            if dest_img.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while dest_img.exists():
                    dest_img = images_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                shutil.copy2(str(img_path), str(dest_img))
            except (OSError, shutil.Error) as e:
                logger.warning(
                    f"Failed to copy image {img_path} to {dest_img}: {e}",
                    extra={"extra_data": {"source": str(img_path), "dest": str(dest_img)}},
                )
                self._skipped_files += 1
                continue

            # Write label file
            label_file = labels_dir / (dest_img.stem + ".txt")
            try:
                with open(label_file, "w", encoding="utf-8") as f:
                    for ann in annotations:
                        f.write(ann.to_yolo_line() + "\n")
                        class_dist[ann.class_id] = class_dist.get(ann.class_id, 0) + 1
            except OSError as e:
                logger.warning(
                    f"Failed to write label file {label_file}: {e}",
                    extra={"extra_data": {"file": str(label_file)}},
                )
                # Remove the copied image since label couldn't be written
                try:
                    dest_img.unlink()
                except OSError:
                    pass
                self._skipped_files += 1
                continue

            written_count += 1

            logger.debug(
                f"Wrote sample: {dest_img.name} with {len(annotations)} annotations",
                extra={
                    "extra_data": {
                        "split": split_name,
                        "image": dest_img.name,
                        "annotation_count": len(annotations),
                    }
                },
            )

        logger.info(
            f"Split '{split_name}': {written_count} images, "
            f"{sum(class_dist.values())} annotations",
            extra={
                "extra_data": {
                    "split": split_name,
                    "image_count": written_count,
                    "class_distribution": class_dist,
                }
            },
        )

        return written_count, class_dist
