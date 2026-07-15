"""Step 2 - Inspect the Roboflow dataset splits."""
import os
import sys

dataset_root = "roboflow data"

for split in ["train", "valid", "test"]:
    images_dir = os.path.join(dataset_root, split, "images")
    labels_dir = os.path.join(dataset_root, split, "labels")
    if os.path.exists(images_dir):
        imgs = len([f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        lbls = len([f for f in os.listdir(labels_dir)
                    if f.endswith('.txt')]) if os.path.exists(labels_dir) else 0
        print(f"{split:6s}: {imgs:4d} images,  {lbls:4d} labels")
    else:
        print(f"{split:6s}: NOT FOUND at {images_dir}")
