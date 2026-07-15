# Hazard Detection System — User Guide

**Project:** Yard Safety CCTV — Hazard Detection  
**Status:** Development / POC Stage  
**Last Updated:** 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Dependencies](#prerequisites--dependencies)
3. [Project Layout](#project-layout)
4. [Step 1 — Verify Installation](#step-1--verify-installation)
5. [Step 2 — Understand the Roboflow Dataset](#step-2--understand-the-roboflow-dataset)
6. [Step 3 — Generate Synthetic Training Data](#step-3--generate-synthetic-training-data)
7. [Step 4 — Prepare Supplemental Data (Optional)](#step-4--prepare-supplemental-data-optional)
8. [Step 5 — Train the YOLOv12 Model](#step-5--train-the-yolov12-model)
9. [Step 6 — Evaluate the Trained Model](#step-6--evaluate-the-trained-model)
10. [Step 7 — Configure the System](#step-7--configure-the-system)
11. [Step 8 — Run Hazard Detection (Live or Fallback)](#step-8--run-hazard-detection-live-or-fallback)
12. [Model Checkpoint Management](#model-checkpoint-management)
13. [Component Interaction Map](#component-interaction-map)
14. [Understanding the Output](#understanding-the-output)
15. [Troubleshooting](#troubleshooting)
16. [Road Map: One-Click Desktop App](#road-map-one-click-desktop-app)

---

## Overview

This system monitors industrial yard environments via multi-camera footage and
detects three categories of hazards using a YOLOv12 object detection model:

| Category | Examples |
|---|---|
| **Human zone violations** | People in no-go areas, PPE violations |
| **Container misalignment** | Shifted, open-door, or misaligned containers |
| **Unsafe container orientation** | Flipped containers, dangling loads |

Classification is **binary**: every detection is either a **hazard** (alert
dispatched) or **not a hazard** (logged only). There are no severity tiers.

### What this guide walks you through

```
generate_synthetic_data.py   ←  Create extra training images
         │
         ▼
  Training Pipeline          ←  Train YOLOv12 on Roboflow + synthetic data
         │
         ▼
  evaluate_model.py          ←  Assess model quality with visual charts
         │
         ▼
  config/hazard_detection.yaml ← Point system at the best checkpoint
         │
         ▼
  src/hazard_detection/main.py ← Run live detection (or fallback simulation)
```

---

## Prerequisites & Dependencies

### Hardware

| Task | Minimum | Recommended |
|---|---|---|
| Training | CPU (slow) | NVIDIA GPU 6 GB+ VRAM |
| Inference | CPU | NVIDIA GPU 4 GB+ VRAM |
| RAM | 8 GB | 16 GB |

The system includes CPU fallback. If no GPU is detected it logs a warning and
continues on CPU — inference will be slower but fully functional.

### Software

```cmd
REM Python 3.10 or 3.11 required
python --version

REM Verify virtual environment is active
.venv\Scripts\activate

REM Install all dependencies
pip install -r requirements.txt
```

Key packages installed by `requirements.txt`:

| Package | Used for |
|---|---|
| `ultralytics` | YOLOv12 training and inference |
| `torch` / `torchvision` | Deep learning backend |
| `opencv-python` | Frame capture and image manipulation |
| `numpy` | Array operations |
| `matplotlib` / `seaborn` | Evaluation charts |
| `pyyaml` | Config file loading |
| `hypothesis` | Property-based tests |
| `pytest` | Test runner |

---

## Project Layout

```
exp_3/
├── config/
│   ├── hazard_detection.yaml      ← Main system config (edit this)
│   └── zones/
│       ├── cam_01_zones.yaml      ← Per-camera zone polygons
│       └── cam_02_zones.yaml
│
├── checkpoints/
│   └── yolov12_best.pt            ← Active model checkpoint (update path in config)
│
├── roboflow data/                 ← Base training dataset (17 classes, ~100 images)
│   ├── data.yaml
│   ├── train/images/ + labels/
│   ├── valid/images/ + labels/
│   └── test/images/ + labels/
│
├── generate_synthetic_data.py     ← Root-level script (from exp_2 pattern)
├── evaluate_model.py              ← Root-level evaluation script
│
├── src/hazard_detection/
│   ├── main.py                    ← Entry point — run this to start detection
│   ├── config.py                  ← Loads + validates hazard_detection.yaml
│   ├── models.py                  ← All dataclasses (HazardEvent, Detection, etc.)
│   ├── yolo_detector.py           ← Wraps Ultralytics YOLO inference
│   ├── frame_sampler.py           ← Captures 5-8 frames per camera
│   ├── human_detector.py          ← Zone-violation and PPE logic
│   ├── container_analyzer.py      ← Misalignment, door, flip, dangle logic
│   ├── alert_dispatcher.py        ← Routes hazard events to channels
│   ├── zone_map.py                ← Polygon zone definitions per camera
│   ├── camera_switcher.py         ← Camera transition stub
│   ├── detection_pipeline.py      ← Orchestrates all stages per camera
│   ├── diagnostics.py             ← Structured logging, timers, state dumps
│   ├── evaluation.py              ← ModelEvaluator with visual chart output
│   └── data_pipeline/
│       ├── supplemental_loader.py ← Load external datasets → YOLO format
│       ├── synthetic_generator.py ← Generate synthetic container scenes
│       └── training_pipeline.py   ← Train / fine-tune YOLOv12
│
├── tests/
│   ├── unit/                      ← All unit + integration tests
│   └── output/                    ← Visual diagnostic PNGs and JSON reports
│
└── docs/
    ├── USER_GUIDE.md              ← This file
    ├── TRAINING_GUIDE.md          ← Deep-dive training reference
    └── CONFIGURATION_GUIDE.md    ← Config field reference
```

---

## Step 1 — Verify Installation

```cmd
python scripts/check_install.py
```

**Expected output (with GPU):**
```
Python: 3.11.x
PyTorch:           2.x.x+cu118
CUDA available:    True
GPU:               NVIDIA GeForce RTX 3050
ultralytics:       OK
hazard_detection:  OK
opencv:            OK
```

**Expected output (CPU only):**
```
Python: 3.11.x
PyTorch:           2.x.x+cpu
CUDA available:    False
GPU:               None (CPU mode)
ultralytics:       OK
hazard_detection:  OK
opencv:            OK
```

> If `hazard_detection: FAILED`, run `python -m pip install -e .` from the
> project root, then retry.
>
> If CUDA shows False but you have a GPU, see the Troubleshooting section.

---

## Step 2 — Understand the Roboflow Dataset

The base dataset lives in `roboflow data/` and contains ~100 annotated yard
images with these 17 detection classes:

```
 0  Boat - With Cargo
 1  Container - Misaligned       ← hazard
 2  Container - Open             ← hazard
 3  Container - Picked
 4  Container - Reefer
 5  Container - Water Drop
 6  Container - Separate
 7  Container - Stacked
 8  Crane
 9  Human
10  Human - No Safety Clothes    ← hazard (PPE violation)
11  Truck - No Container
12  Truck - With Container
13  Vehicle
14  Yard - Dropoff zone          ← zone annotation
15  Yard - No People             ← zone annotation
16  Yard - Operation Zone        ← zone annotation
```

Annotations are in YOLO format: one `.txt` file per image, each line:
```
class_id  x_center  y_center  width  height
```
All coordinates are normalized to [0.0, 1.0] relative to image dimensions.

### Inspect the dataset split

```cmd
python scripts/check_dataset.py
```

**Expected output:**
```
train:   85 images,    85 labels
valid:   12 images,    12 labels
test:    12 images,    12 labels
```

> The dataset is small. That is why we generate synthetic data in the next
> step — to give the model more variety before training.

---

## Step 3 — Generate Synthetic Training Data

> **Note on the root-level `generate_synthetic_data.py`:** This script was
> written for exp_2's folder structure (class subfolders like
> `misaligned - none/`). It does **not** work with the YOLO layout used here
> (`roboflow data/train/images/` + `roboflow data/train/labels/`). Skip
> Option A unless you have a class-subfolder dataset to augment.
>
> For YOLO-format training data, Ultralytics applies augmentation
> automatically during training (mosaic, flips, HSV shifts, etc.) — you do
> not need to pre-augment the Roboflow dataset before running Step 5.

### Option A — Augment the Roboflow YOLO dataset

Generates augmented copies of the train and test images in-place. Labels are
kept in sync — flip transforms recalculate bounding box coordinates automatically.

```cmd
REM Dry run first to see what would be created
python scripts/augment_dataset.py --dry-run

REM Generate 3 augmented copies per original image (default)
python scripts/augment_dataset.py

REM More copies for a larger dataset
python scripts/augment_dataset.py --copies 5
```

**What it does:**
- Reads every image in `roboflow data/train/images/` and `roboflow data/test/images/`
- Creates `_aug01`, `_aug02` ... copies with random rotation, brightness, contrast, noise, translation, blur, and horizontal flip
- Writes matching `.txt` label files alongside each augmented image
- Leaves `valid/` completely untouched
- Originals are never modified

**Expected output:**
```
======================================================================
 YOLO DATASET AUGMENTATION
======================================================================
 Dataset   : roboflow data
 Splits    : ['train', 'test']
 Copies    : 3 per original image
======================================================================

[train]
  85 original images found in train/images/
  Generating 3 augmented copies per image (255 new images)...
  Augmenting train: 100%|...| 85/85
  Done: 85 images processed, 255 copies created.

[test]
  12 original images found in test/images/
  Generating 3 augmented copies per image (36 new images)...
  Done: 12 images processed, 36 copies created.

 AUGMENTATION COMPLETE
 Total originals processed : 97
 Total copies created      : 291
======================================================================
```

> **Checkpoint 3A:** Run `python scripts/check_dataset.py` after augmentation.
> The train count should now be 4x the original (85 originals + 255 augmented = 340).

### Option B — Use SyntheticDataGenerator (container scene compositing)

This generates completely new scenes by superimposing container images onto
background scenes in safe and unsafe configurations. Requires background images
and container asset images.

```cmd
REM Create the asset directories first
mkdir data\backgrounds
mkdir data\container_assets

REM Then run via Python
python -c "
import sys; sys.path.insert(0, 'src')
from hazard_detection.data_pipeline.synthetic_generator import (
    SyntheticDataGenerator, SyntheticGeneratorConfig, DataAugmenter
)

config = SyntheticGeneratorConfig(
    containers_per_scene=3,
    samples_per_class=100,
    background_dir='data/backgrounds',
    container_assets_dir='data/container_assets',
    output_dir='data/synthetic_output',
    seed=42
)
gen = SyntheticDataGenerator(augmenter=DataAugmenter(), config=config)
result = gen.generate()

if result:
    print(f'Generated {result[\"total_images\"]} images')
    print(f'Balance deviation: {result[\"balance_deviation\"]*100:.1f}%')
    print('Per-class counts:', result['per_class_counts'])
"
```

> **Checkpoint 3A:** After generation, check the output folder has images and
> matching `.txt` label files before continuing.

```cmd
REM Verify outputs exist
dir data\synthetic_output\images
dir data\synthetic_output\labels
```

---

## Step 4 — Prepare Supplemental Data (Optional)

If you have access to additional container imagery datasets (COCO JSON, Pascal
VOC XML, or plain YOLO format), the `SupplementalDatasetLoader` can normalize
and merge them with the Roboflow dataset automatically.

```cmd
python -c "
import sys; sys.path.insert(0, 'src')
from hazard_detection.data_pipeline.supplemental_loader import (
    SupplementalDatasetLoader, SupplementalConfig
)

config = SupplementalConfig(
    dataset_roots=['path/to/your/external/dataset'],
    output_dir='data/supplemental_output',
    roboflow_data_yaml='roboflow data/data.yaml',
    split_ratios={'train': 0.70, 'valid': 0.15, 'test': 0.15},
    random_seed=42
)
loader = SupplementalDatasetLoader(config)
result = loader.load_and_normalize()

print(f'Total images: {result.total_images}')
print(f'Total annotations: {result.total_annotations}')
print(f'Discarded (no mapping): {result.discarded_annotations}')
print(f'Split counts: {result.split_counts}')
"
```

**Supported source formats:**
- COCO JSON (`annotations.json` with `images` + `annotations` keys)
- Pascal VOC XML (`.xml` files in `Annotations/` folder)
- YOLO txt (`.txt` labels in `labels/` folder alongside `images/`)

The loader remaps source class names to the Roboflow 17-class taxonomy
automatically (e.g. `"shipping_container"` → class ID 7 `"Container - Stacked"`).
Any annotation with no mapping is discarded and the count is logged.

> **Skip this step** if you only have the Roboflow dataset + synthetic data.
> It is purely additive.

---

## Step 5 — Train the YOLOv12 Model

### 5a — Train from scratch (GPU)

```cmd
python scripts/train_yolo.py
```

### 5b — Train on CPU only

```cmd
python scripts/train_yolo.py --device cpu --batch 8 --epochs 50
```

### 5c — Resume from a previous checkpoint

```cmd
python scripts/train_yolo.py --resume
```

### 5d — Fine-tune on supplemental or synthetic data

```cmd
python scripts/finetune_yolo.py --checkpoint runs/train/hazard_yolo/weights/best.pt
```

All scripts accept `--help` to see available options:

```cmd
python scripts/train_yolo.py --help
```

**Expected console output during training:**
```
Training started
  weights: checkpoints/yolov12_best.pt
  data_yaml: roboflow data/data.yaml
  epochs: 100  batch: 16  lr: 0.001  resolution: 640

Epoch 1/100: box_loss=4.21  cls_loss=3.87  dfl_loss=1.52
Epoch 5/100: box_loss=2.44  cls_loss=2.01  dfl_loss=1.31
...
Training complete in 847.3s
Best checkpoint (highest mAP@0.5) saved to: runs/train/hazard_yolo/weights/best.pt
```

> **Checkpoint 5:** After training, confirm `runs/train/hazard_yolo/weights/best.pt`
> exists before proceeding.

```cmd
dir runs\train\hazard_yolo\weights
```

---

## Step 6 — Evaluate the Trained Model

```cmd
python scripts/evaluate_yolo.py
```

Options:

```cmd
python scripts/evaluate_yolo.py --checkpoint checkpoints/yolov12_best.pt
python scripts/evaluate_yolo.py --conf 0.3 --device cpu
python scripts/evaluate_yolo.py --help
```

**Charts saved to `evaluation_results/`:**

| File | What it shows |
|---|---|
| `confusion_matrix.png` | Per-class prediction accuracy |
| `classification_metrics.png` | Precision / Recall / F1 per class |
| `class_distribution.png` | Ground truth vs predictions spread |
| `roc_curves.png` | ROC with AUC per class |
| `confidence_distribution.png` | Confidence histogram: correct vs incorrect |
| `evaluation_summary.json` | Full per-class metrics as JSON |

**Interpreting a poor evaluation:**
- Low F1 on `Container - Misaligned` → dataset too small for this class → generate more synthetic misaligned scenes
- High confusion between `Container - Stacked` and `Container - Misaligned` → IoU threshold may need tuning
- Confidence distribution skewed toward 0.3–0.5 → consider lower `conf_threshold` in config

> **Checkpoint 6:** Open `evaluation_results/evaluation_summary.json` and
> check `overall_accuracy`. Anything above 0.65 is a reasonable starting
> point for this dataset size. Retrain with more synthetic data if needed.

---

## Step 7 — Configure the System

All runtime parameters live in `config/hazard_detection.yaml`.
Open it and update the values that matter most:

### 7a — Point to your best checkpoint

```yaml
yolo:
  checkpoint_path: "runs/train/hazard_yolo/weights/best.pt"  # ← update this
  device: "cuda"        # or "cpu" for CPU-only mode
  input_resolution: 640
  confidence_threshold: 0.5
```

> This is the **model path** pattern — change this line whenever you
> train a better model or transfer a checkpoint from another machine.
> You never need to edit Python code to swap models.

### 7b — Set your camera sequence

```yaml
cameras:
  sequence: ["cam_01", "cam_02"]   # ← list your cameras here
```

### 7c — Configure zone maps

Each camera can have a zone map YAML that defines no-go polygon regions.
If no map is configured, the entire field of view defaults to a no-people zone.

```yaml
zone_maps:
  cam_01: "config/zones/cam_01_zones.yaml"
  cam_02: "config/zones/cam_02_zones.yaml"
```

Zone files use normalized coordinates (0.0–1.0):

```yaml
# config/zones/cam_01_zones.yaml
zones:
  cam_01:
    - zone_type: "no_people"
      vertices: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    - zone_type: "operation"
      vertices: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
```

### 7d — CPU vs GPU

```yaml
yolo:
  device: "cpu"    # Force CPU (for machines without CUDA)
```

The system also automatically falls back to CPU if CUDA is requested but
unavailable, logging a warning rather than crashing.

### 7e — Alert rate limiting

```yaml
alerts:
  rate_limit_seconds: 60   # Suppress duplicate alerts within this window (10-300)
  channels: ["email", "dashboard"]
```

> See `docs/CONFIGURATION_GUIDE.md` for the full field reference.

---

## Step 8 — Run Hazard Detection (Live or Fallback)

### 8a — Start with live cameras

```cmd
.venv\Scripts\activate

python -m src.hazard_detection.main --config config/hazard_detection.yaml
```

The system will:
1. Load and validate the config
2. Connect to camera feeds via `FrameSampler`
3. Cycle through each camera in the configured sequence
4. For each camera: sample 5–8 frames → run YOLO inference → analyze hazards → dispatch alerts
5. Restart from the first camera and repeat indefinitely

To stop: press **Ctrl+C** — the system shuts down gracefully.

**Expected startup output:**
```
========================================================================
HAZARD DETECTION SYSTEM — STARTUP
========================================================================
System info: version=1.0.0, python=3.11.x, os=nt
Configuration file: config/hazard_detection.yaml
Camera sequence (2 cameras): ['cam_01', 'cam_02']
Per-camera timeout: 30s
FrameSampler config: frame_count=6, timeout_ms=2000, max_retries=3
YOLO config: checkpoint='runs/train/hazard_yolo/weights/best.pt', device='cuda'
...
========================================================================
Pipeline starting...

Camera processing started: 'cam_01'
Camera processing complete: 'cam_01' — 0 events, total time: 142.3ms
Camera processing started: 'cam_02'
Camera processing complete: 'cam_02' — 1 events (1 hazards), total time: 138.7ms

Cycle 1 complete: 1 detections, 1 hazards, elapsed=284.2ms
```

---

### 8b — Fallback mode (no live cameras)

If no physical cameras are connected, the system automatically uses a **stub
frame sampler** that generates synthetic blank frames. Detection will run but
produce no real detections — useful for verifying the pipeline runs end-to-end.

```cmd
python -m src.hazard_detection.main ^
    --config config/hazard_detection.yaml ^
    --log-level DEBUG
```

You will see in the output:
```
WARNING  StubFrameSampler: generated 6 blank frames for camera 'cam_01'
```

---

### 8c — Run against the Roboflow test images (simulation)

```cmd
python scripts/run_on_test_images.py
```

Options:

```cmd
python scripts/run_on_test_images.py --checkpoint checkpoints/yolov12_best.pt
python scripts/run_on_test_images.py --conf 0.3 --device cpu
python scripts/run_on_test_images.py --help
```

Annotated images are saved to `runs/detect/test_run/`.

---

### 8d — CLI flags reference

```cmd
python -m src.hazard_detection.main --help
```

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `config/hazard_detection.yaml` | Config file to load |
| `--log-level LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |
| `--dump-dir DIR` | from config | Override diagnostic dump directory |

---

### 8e — What happens at shutdown

When you press Ctrl+C:
```
Signal SIGINT received — initiating graceful shutdown...

========================================================================
HAZARD DETECTION SYSTEM — SHUTDOWN
========================================================================
Final statistics: cycles=12, total_detections=47, total_hazards=3
Camera sequence processed: ['cam_01', 'cam_02']
Per-camera timeout configured: 30s
Shutdown timestamp: 2026-07-08T14:22:11Z
========================================================================
Hazard Detection System exited cleanly.
```

Final stats and diagnostic dumps are saved to `diagnostics/dumps/`.

---

## Model Checkpoint Management

### Checkpoint file naming convention

After training, Ultralytics saves two files:

```
runs/train/hazard_yolo/weights/
    best.pt    ← highest validation mAP@0.5 during training
    last.pt    ← final epoch checkpoint
```

### Saving a checkpoint for deployment

Copy `best.pt` to the `checkpoints/` folder with a descriptive name:

```cmd
REM Windows
copy runs\train\hazard_yolo\weights\best.pt checkpoints\yolov12_v2_100ep.pt

REM Then update config/hazard_detection.yaml:
REM   checkpoint_path: "checkpoints/yolov12_v2_100ep.pt"
```

### Transferring a model between machines

1. Copy the `.pt` file to the target machine's `checkpoints/` folder
2. Update `checkpoint_path` in `config/hazard_detection.yaml` on that machine
3. Verify the model loads:

```cmd
python -c "
import sys; sys.path.insert(0, 'src')
from ultralytics import YOLO
model = YOLO('checkpoints/yolov12_v2_100ep.pt')
print('Model loaded:', model.info())
"
```

### Keeping track of which checkpoint to use

The `checkpoint_path` in `config/hazard_detection.yaml` is your single point of
control. You do not need to rename or move files — just update that one line:

```yaml
yolo:
  checkpoint_path: "checkpoints/yolov12_v2_100ep.pt"   # ← swap here
```

This mirrors the pattern from exp_2 where model paths were kept in the
config file so the best-performing model could be swapped without code changes.

---

## Component Interaction Map

```
                    config/hazard_detection.yaml
                              │
                    ConfigurationManager
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    FrameSampler         ZoneMap           AlertDispatcher
    (5-8 frames)     (polygon zones)    (binary: hazard/not)
          │
    YOLODetector
    (17-class YOLO)
          │
    ┌─────┴──────┐
    │            │
HumanDetector   ContainerAnalyzer
(zone lookup)   (IoU, aspect ratio,
    │            optical flow)
    └─────┬──────┘
          │
    HazardEvent
    (is_hazard: True/False)
          │
    AlertDispatcher
    (dispatch if is_hazard=True,
     rate-limit by camera+type)
```

### Data flow per camera cycle

```
1. CameraSwitcher.transition(camera_id)         → stub returns True
2. FrameSampler.sample(camera_id)               → FrameSequence (5-8 frames)
3. YOLODetector.detect(frame_sequence)          → List[List[Detection]]
4. HumanDetector.analyze(detections, camera_id) → List[HazardEvent]
5. ContainerAnalyzer.analyze(detections, frames)→ List[HazardEvent]
6. AlertDispatcher.dispatch(event) × N          → channel delivery
7. FrameSampler.release()                       → memory freed
```

### Binary hazard classification

Every detection produces a `HazardEvent` with:

```python
is_hazard = True   # confirmed (≥2 consecutive frames above threshold)
                   # → alert dispatched through configured channels

is_hazard = False  # unconfirmed (1 frame only, or below threshold)
                   # → logged only, no alert
```

There are no severity levels (LOW/MEDIUM/HIGH/CRITICAL). Every confirmed
hazard is treated equally and dispatched immediately.

---

## Understanding the Output

### Diagnostic dumps

State snapshots between pipeline stages are saved as JSON to
`diagnostics/dumps/` (configurable in `hazard_detection.yaml`):

```
diagnostics/dumps/
    20260708T142201_000000_startup_config.json
    20260708T142201_000011_post_sampling_cam_01.json
    20260708T142201_000022_post_detection_cam_01.json
    20260708T142201_000033_post_human_analysis_cam_01.json
    20260708T142201_000044_cycle_summary.json
    20260708T142209_000055_shutdown_stats.json
```

Open any of these in a text editor to inspect intermediate state.

### Structured log format

Every log line is JSON:

```json
{"timestamp": "2026-07-08T14:22:01.123456Z", "level": "INFO",
 "module": "hazard_detection.detection_pipeline",
 "message": "Camera processing complete: 'cam_01' — 1 events (1 hazards)",
 "correlation_id": "a3f2-...",
 "extra_data": {"stage_timings_ms": {"frame_sampling": 45.2, "yolo_inference": 89.1, ...}}}
```

To enable file logging (in addition to console):

```yaml
diagnostics:
  log_file: "logs/hazard_detection.log"
```

### Test outputs

After running `pytest tests/unit/`, visual diagnostics are saved to
`tests/output/`. These show class distributions, confidence histograms,
annotated detection frames, and timing breakdowns — useful for debugging
whether the model is detecting the right classes.

---

## Troubleshooting

### "YOLO checkpoint not found"

```
WARNING  YOLO checkpoint 'checkpoints/yolov12_best.pt' not found
         — using StubYOLODetector. Detection functionality disabled.
```

**Fix:** Either run training first (Step 5) or point `checkpoint_path` to an
existing `.pt` file in `config/hazard_detection.yaml`.

---

### "CUDA unavailable — falling back to CPU"

```
WARNING  CUDA device requested but no CUDA-capable GPU available.
         Falling back to CPU inference.
```

This is not an error. Inference will work on CPU, just slower. If you
want to suppress the warning, set `device: "cpu"` in the config explicitly.

---

### "Configuration file not found"

```
ERROR    Configuration error: Configuration file not found: 'config/hazard_detection.yaml'
```

**Fix:** Run from the project root (`exp_3/`), not from a subdirectory:

```cmd
cd c:\Users\Miles\Desktop\intern work\exp_3
python -m src.hazard_detection.main
```

---

### YOLO training exits immediately with an error

```
ERROR    Invalid training hyperparameter: epochs=0 is out of range [1, 1000]
```

`TrainingConfig` validates parameters at construction time and calls
`sys.exit(1)` on invalid values. Valid ranges:

| Parameter | Range |
|---|---|
| `epochs` | 1 – 1000 |
| `batch_size` | 1 – 64 |
| `learning_rate` | 1e-6 – 0.1 |
| `image_resolution` | 320 – 1280 |
| `checkpoint_interval` | ≥ 1 |

---

### Training is very slow

Use the CPU flag with fewer epochs for a quick smoke test:

```cmd
python scripts/train_yolo.py --device cpu --batch 4 --epochs 5
```

---

### Zone violations not triggering alerts

**Check 1:** Is the camera configured in `zone_maps`?

```yaml
zone_maps:
  cam_01: "config/zones/cam_01_zones.yaml"
```

If no zone map is configured, the entire FOV defaults to a no-people zone
(so everything triggers a zone violation). If a map is configured but the
file doesn't exist, the system logs a warning and uses the default.

**Check 2:** Is the confidence threshold too high?

```yaml
detection:
  human:
    confidence_threshold: 0.5
```

Lower it to 0.3 to see if more detections appear.

**Check 3:** Is temporal confirmation failing?

Zone violations require detection in **≥2 consecutive frames** to be
confirmed (is_hazard=True). A single-frame detection is logged as transient
but no alert is sent. Check diagnostic dumps to see frame-by-frame detection counts.

---

### Container hazards not detected

**Check:** Is the aspect ratio threshold appropriate?

```yaml
detection:
  container:
    flipped_aspect_ratio_threshold: 1.5   # container is flagged flipped if h/w > 1.5
```

Shipping containers are always wider than they are tall in normal orientation.
If you are seeing legitimate containers being missed, lower this value slightly.

---

### Tests failing after changes

```cmd
REM Run the full test suite to catch regressions
python -m pytest tests/unit/ -v --tb=short
```

The suite covers 484 tests across 18 modules. If property-based tests
(Hypothesis) fail, they print the failing input that caused the failure —
use that to diagnose the edge case.

---

## Road Map: One-Click Desktop App

The goal is a double-clickable `.bat` (and eventually `.exe`) that:
1. Opens a terminal showing the detection pipeline running
2. Opens a terminal showing live training status (if retraining)
3. Falls back to simulated data if no cameras are detected

**Current state of `launch_all.bat` (planned):**

```bat
@echo off
REM Hazard Detection System — Full Launcher
REM =========================================
REM Double-click to start everything.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM 1. Start main detection pipeline
start "Hazard Detection" cmd /k ".venv\Scripts\activate && python -m src.hazard_detection.main --config config/hazard_detection.yaml"

REM 2. Open evaluation results folder (so you can see charts)
timeout /t 3 /nobreak >nul
start "" "evaluation_results"

echo All systems launched. Close the terminal windows to stop.
pause
```

> **Next milestone:** A `launch_all.bat` that starts the pipeline, opens
> the diagnostics folder, and provides a clear status message. Once the
> pipeline produces output we are happy with on real cameras, we wrap
> it into a desktop shortcut following the pattern from `exp_2/create_desktop_shortcut.bat`.

---

## Quick Reference: Key Commands

```cmd
REM Verify everything is installed
python scripts/check_install.py

REM Check dataset split sizes
python scripts/check_dataset.py

REM Augment train and test splits (3 copies per image)
python scripts/augment_dataset.py
python scripts/augment_dataset.py --copies 5

REM Train (GPU)
python scripts/train_yolo.py

REM Train (CPU)
python scripts/train_yolo.py --device cpu --batch 8 --epochs 50

REM Resume training
python scripts/train_yolo.py --resume

REM Fine-tune on new data
python scripts/finetune_yolo.py

REM Evaluate with visual charts
python scripts/evaluate_yolo.py

REM Run on Roboflow test images
python scripts/run_on_test_images.py

REM Run live detection pipeline
python -m hazard_detection.main --config config/hazard_detection.yaml

REM Run all tests
python -m pytest tests/unit/ -v --tb=short
```

---

*See also:*
- `docs/CONFIGURATION_GUIDE.md` — full config field reference
- `docs/TRAINING_GUIDE.md` — deep-dive training and hyperparameter tuning
- `.kiro/specs/hazard-detection-system/requirements.md` — system requirements
- `.kiro/specs/hazard-detection-system/design.md` — architecture and design decisions
