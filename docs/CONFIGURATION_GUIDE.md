# Configuration Guide: Hazard Detection System

**Config file:** `config/hazard_detection.yaml`  
**Last Updated:** 2026

---

## Quick Start

1. Open `config/hazard_detection.yaml`
2. Update `yolo.checkpoint_path` to point to your trained model
3. Update `cameras.sequence` with your camera IDs
4. Set `yolo.device` to `"cpu"` if you have no GPU
5. Run: `python -m src.hazard_detection.main`

---

## Full Config File with Annotations

```yaml
# ============================================================
# SYSTEM
# ============================================================
system:
  frame_sample_count: 6          # Frames to capture per camera (5–8)
  per_camera_timeout_seconds: 30 # Skip camera if processing exceeds this

# ============================================================
# CAMERAS
# ============================================================
cameras:
  sequence: ["cam_01", "cam_02"] # Cameras processed in this order, cycling

# ============================================================
# YOLO MODEL
# ============================================================
yolo:
  checkpoint_path: "checkpoints/yolov12_best.pt"  # ← CHANGE THIS
  device: "cuda"          # "cuda" or "cpu" — auto-falls back to cpu if needed
  input_resolution: 640   # Square input size (320–750)
  confidence_threshold: 0.5  # Discard detections below this

# ============================================================
# DETECTION THRESHOLDS
# ============================================================
detection:
  human:
    confidence_threshold: 0.5     # Min confidence to emit human event

  container:
    confidence_threshold: 0.5     # Min confidence to emit container event
    flipped_aspect_ratio_threshold: 1.5  # h/w ratio that flags "flipped"
    safe_overlap_threshold: 0.3   # IoA with crane below this = dangling
    ground_level_threshold: 0.4   # y_center above this = high in frame
    motion_threshold: 0.7         # Flow variance above this = motion flag
    iou_threshold: 0.5            # IoU for overlapping class disambiguation

# ============================================================
# ALERTS
# ============================================================
alerts:
  rate_limit_seconds: 60          # Suppress same camera+type within window (10–300)
  channels: ["email", "dashboard"]

# ============================================================
# ZONE MAPS (per camera)
# ============================================================
zone_maps:
  cam_01: "config/zones/cam_01_zones.yaml"  # omit camera to use default (full FOV = no-people)
  cam_02: "config/zones/cam_02_zones.yaml"

# ============================================================
# DIAGNOSTICS
# ============================================================
diagnostics:
  log_levels:
    hazard_detection.frame_sampler: INFO
    hazard_detection.yolo_detector: INFO
    hazard_detection.human_detector: INFO
    hazard_detection.container_analyzer: INFO
    hazard_detection.alert_dispatcher: INFO
    hazard_detection.performance: INFO
    hazard_detection.pipeline_tracer: DEBUG
    hazard_detection.diagnostic_dumper: DEBUG
  dump_directory: "diagnostics/dumps"
  dump_enabled: true
  max_dumps: 1000          # Old dumps removed when exceeded
  tracing_enabled: true
  log_file: null           # Set to "logs/hazard_detection.log" to write to file

# ============================================================
# TRAINING (used by training_pipeline.py only, not runtime)
# ============================================================
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  image_resolution: 640
  checkpoint_interval: 5
  data_yaml: "roboflow data/data.yaml"
```

---

## Field Reference

### `system`

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `frame_sample_count` | int | 5–8 | 6 | Frames captured per camera before switching |
| `per_camera_timeout_seconds` | int | >0 | 30 | Skip camera if pipeline exceeds this |

---

### `cameras`

| Field | Type | Description |
|---|---|---|
| `sequence` | list of strings | Camera IDs processed in order. Cycles indefinitely. |

---

### `yolo`

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `checkpoint_path` | string | — | `"checkpoints/yolov12_best.pt"` | Path to `.pt` model file |
| `device` | string | `"cuda"` / `"cpu"` | `"cuda"` | Auto-falls to CPU if CUDA unavailable |
| `input_resolution` | int | 320–750 | 640 | Square image size fed to YOLO |
| `confidence_threshold` | float | 0.0–1.0 | 0.5 | Detections below this are discarded |

---

### `detection.human`

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `confidence_threshold` | float | 0.0–1.0 | 0.5 | Min confidence for human zone/PPE events |

---

### `detection.container`

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `confidence_threshold` | float | 0.0–1.0 | 0.5 | Min confidence for container events |
| `flipped_aspect_ratio_threshold` | float | >0 | 1.5 | h/w ratio above this = flipped container |
| `safe_overlap_threshold` | float | 0.0–1.0 | 0.3 | IoA with crane below this = dangling |
| `ground_level_threshold` | float | 0.0–1.0 | 0.4 | y_center below this means container is "high" |
| `motion_threshold` | float | ≥0 | 0.7 | Optical flow variance above this = motion flag |
| `iou_threshold` | float | 0.0–1.0 | 0.5 | IoU for disambiguation (Misaligned vs Stacked) |

---

### `alerts`

| Field | Type | Range | Default | Description |
|---|---|---|---|---|
| `rate_limit_seconds` | int | 10–300 | 60 | Suppress duplicate alerts per camera+type |
| `channels` | list | — | `["email", "dashboard"]` | Alert channel names (stub adapters) |

---

### `zone_maps`

Maps camera IDs to zone YAML files. Each zone file defines polygonal regions:

```yaml
zones:
  cam_01:
    - zone_type: "no_people"
      vertices: [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    - zone_type: "operation"
      vertices: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
    - zone_type: "dropoff"
      vertices: [[0.7, 0.7], [1.0, 0.7], [1.0, 1.0], [0.7, 1.0]]
```

**Zone types:**

| Type | Meaning |
|---|---|
| `no_people` | Human presence triggers `zone_violation` |
| `operation` | Human presence is expected, no alert |
| `dropoff` | Human presence is expected, no alert |

**Rules:**
- Vertices are normalized (0.0–1.0 relative to frame width/height)
- Minimum 3 vertices per polygon
- If no zone map for a camera → entire FOV = `no_people`
- Invalid file → rejected, previous definitions retained, error logged

---

### `diagnostics`

| Field | Type | Default | Description |
|---|---|---|---|
| `log_levels` | dict | (see above) | Per-module log level: DEBUG/INFO/WARNING/ERROR |
| `dump_directory` | string | `"diagnostics/dumps"` | Where JSON state snapshots go |
| `dump_enabled` | bool | `true` | Set `false` to disable all dumps |
| `max_dumps` | int | 1000 | Oldest dumps removed when limit exceeded |
| `tracing_enabled` | bool | `true` | Pipeline execution traces (per-module timing) |
| `log_file` | string / null | null | Set to a path to also write logs to file |

---

## Common Mistakes

### ❌ Wrong: Backslashes in paths on Windows
```yaml
checkpoint_path: "checkpoints\yolov12_best.pt"   # fails on some systems
```

### ✅ Correct: Forward slashes work everywhere
```yaml
checkpoint_path: "checkpoints/yolov12_best.pt"
```

---

### ❌ Wrong: Frame count outside 5–8
```yaml
system:
  frame_sample_count: 10    # invalid — startup will fail
```

### ✅ Correct
```yaml
system:
  frame_sample_count: 6
```

---

### ❌ Wrong: Rate limit outside 10–300
```yaml
alerts:
  rate_limit_seconds: 5     # invalid — startup will fail
```

### ✅ Correct
```yaml
alerts:
  rate_limit_seconds: 60
```

---

## Swapping Models

To swap the active model, **only edit `checkpoint_path`**:

```yaml
yolo:
  checkpoint_path: "checkpoints/yolov12_v3_finetune.pt"   # ← point here
```

No code changes required. The system loads whichever checkpoint this path
points to at startup.

---

## Support

- **User walkthrough:** `docs/USER_GUIDE.md`
- **Training deep-dive:** `docs/TRAINING_GUIDE.md`
- **System requirements:** `.kiro/specs/hazard-detection-system/requirements.md`
- **Architecture design:** `.kiro/specs/hazard-detection-system/design.md`
