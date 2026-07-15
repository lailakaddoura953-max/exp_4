"""
Data Pipeline package for the Hazard Detection System.

This package contains components for loading, normalizing, and generating
training data for the YOLOv12 model:
- SupplementalDatasetLoader: Loads external container datasets and normalizes to YOLO format
- SyntheticDataGenerator: Generates synthetic container scenarios for training diversity
- YOLOTrainingPipeline: Training script wrapper for YOLOv12
"""

from hazard_detection.data_pipeline.supplemental_loader import (  # noqa: F401
    SupplementalDatasetLoader,
    SupplementalConfig,
    AnnotationEntry,
    DatasetOutput,
    ROBOFLOW_CLASSES,
)

__all__ = [
    "SupplementalDatasetLoader",
    "SupplementalConfig",
    "AnnotationEntry",
    "DatasetOutput",
    "ROBOFLOW_CLASSES",
]
