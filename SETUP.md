# SETUP.md — Full System Integration Guide

This is the top-level setup guide for the Yard Safety CCTV Hazard Detection
System. It ties together every subsystem in this repository — dataset,
annotation pipelines, training, evaluation, the runtime detection pipeline,
and the web dashboard — into one linear path you can follow on a fresh
machine.

Each subsystem also has its own deeper guide; this document tells you which
one to read when, and in what order, rather than repeating everything.

```
docs/USER_GUIDE.md              — deep walkthrough of dataset → train → run
docs/TRAINING_GUIDE.md          — training internals, hyperparameters
docs/CONFIGURATION_GUIDE.md     — full config/hazard_detection.yaml reference
docs/DASHBOARD_GUIDE.md         — web dashboard architecture and API
docs/LIVE_CAMERA_SETUP.md       — connecting a real RTSP camera to the dashboard
scripts/annotation/SETUP.md     — annotation pipeline (segmentation + CNN fallback)
.kiro/specs/cnn-fallback-annotation-pipeline/  — design docs for the CNN fallback
```

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Clone and Create the Main Environment](#step-1--clone-and-create-the-main-environment)
5. [Step 2 — Verify the Installation](#step-2--verify-the-installation)
6. [Step 3 — Get the Dataset](#step-3--get-the-dataset)
7. [Step 4 — Annotation Pipeline (Optional, Only If You Need More Labeled Data)](#step-4--annotation-pipeline-optional-only-if-you-need-more-labeled-data)
8. [Step 5 — Train the YOLO Detector](#step-5--train-the-yolo-detector)
9. [Step 6 — Evaluate the Trained Model](#step-6--evaluate-the-trained-model)
10. [Step 7 — Configure and Run the Detection System](#step-7--configure-and-run-the-detection-system)
11. [Step 8 — Run the Web Dashboard (Optional)](#step-8--run-the-web-dashboard-optional)
12. [Running the Test Suite](#running-the-test-suite)
13. [End-to-End Checklist](#end-to-end-checklist)
14. [Troubleshooting](#troubleshooting)

---

## 1. System Overview

The system detects three categories of hazards in industrial yard CCTV
footage using a YOLO object detector, with binary classification (hazard /
not-hazard, no severity tiers):

| Category | Examples |
|---|---|
| Human zone violations | People in no-go areas, PPE violations |
| Container misalignment | Shifted, open-door, or misaligned containers |
| Unsafe container orientation | Flipped containers, dangling loads |

End-to-end data flow:

```
 Annotation pipeline           Training                  Runtime
 (label raw footage)      (fit YOLO on labels)      (detect + alert)
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ auto_annotate.py  │      │                  │      │ main.py          │
│  (Grounded SAM 2) │─────▶│ train_yolo.py    │─────▶│  FrameSampler     │
│      or           │      │ finetune_yolo.py │      │  YOLODetector     │
│ cnn_auto_annotate │      │                  │      │  HumanDetector    │
│  (YOLO fallback)  │      │ evaluate_yolo.py │      │  ContainerAnalyzer│
└──────────────────┘      └──────────────────┘      │  AlertDispatcher  │
                                                      └──────────────────┘
                                                              │
                                                              ▼
                                              config/hazard_detection.yaml
                                              (points everything at the
                                               trained checkpoint)
```

There is also a standalone **web dashboard** (`src/dashboard/`) for
uploading a single image and seeing inference results in a browser — useful
for demos and spot-checking a checkpoint without wiring up real cameras.

---

## 2. Repository Layout

```
exp_4/
├── config/
│   ├── hazard_detection.yaml       ← main runtime config (edit this)
│   ├── zones/                      ← per-camera zone polygons
│   └── *.yaml                      ← alternate architecture configs
│
├── roboflow data/                  ← base labeled dataset (17 classes, YOLO format)
│                                      NOT committed — see Step 3
│
├── image_data_normal/              ← optional larger hazard-free dataset
│                                      (access-restricted; not on every machine)
│
├── scripts/
│   ├── annotation/                 ← auto-annotation pipelines (see Step 4)
│   │   ├── auto_annotate.py        ← Grounded SAM 2 + Grounding DINO (segmentation)
│   │   ├── cnn_auto_annotate.py    ← YOLO-based fallback (bounding boxes)
│   │   ├── run_auto_annotate.py    ← single entry point, --pipeline segmentation|cnn
│   │   ├── review_annotations.py   ← manual review of low-confidence labels
│   │   └── SETUP.md                ← annotation-specific setup guide
│   ├── check_dataset.py            ← verify dataset splits are populated
│   ├── check_install.py            ← quick dependency check
│   ├── train_yolo.py               ← train YOLO from scratch
│   ├── finetune_yolo.py            ← fine-tune an existing checkpoint
│   ├── evaluate_yolo.py            ← produce evaluation charts + JSON
│   ├── generate_hazard_augmentations.py  ← synthetic hazard injection (sanity check)
│   └── pretrain_hazard_sanity_check.py   ← CNN-fallback viability sanity check
│
├── src/
│   ├── hazard_detection/           ← runtime detection pipeline (main.py entry point)
│   ├── acquisition/                ← frame acquisition / camera sync
│   ├── cv/                         ← optical flow analysis
│   ├── alerting/                   ← alert dispatch
│   └── dashboard/                  ← Flask web dashboard (src/dashboard/app.py)
│
├── tests/unit/                     ← pytest test suite
│
├── docs/                           ← deep-dive guides + GitHub Pages demo site
│
└── .kiro/specs/                    ← spec-driven design docs (requirements/design/tasks)
```

---

## 3. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 or 3.11** | 3.12 also works for most of the system; the segmentation annotation pipeline (`Grounded-SAM-2`) specifically wants 3.10 — see Step 4 |
| **Git** | to clone this repo and, for the segmentation pipeline, Grounded SAM 2 |
| **NVIDIA GPU with CUDA** (recommended) | 4 GB+ VRAM for inference, 6 GB+ for training. CPU fallback exists everywhere but is much slower |
| **8+ GB free disk** | more if you also set up the segmentation annotation pipeline (SAM 2 + Grounding DINO weights are several GB) |

You do **not** need a GPU to read this guide or run most scripts — every
training/inference script accepts `--device cpu` and the runtime pipeline
auto-falls-back to CPU with a logged warning if CUDA isn't available.

---

## Step 1 — Clone and Create the Main Environment

```bash
git clone https://github.com/lailakaddoura953-max/exp_4.git
cd exp_4

python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Install PyTorch first, with the CUDA build matching your driver (check
https://pytorch.org/get-started/locally/ for the current command). Example
for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest of the core dependencies used across training,
evaluation, and the runtime pipeline:

```bash
pip install ultralytics opencv-python numpy pillow pyyaml matplotlib seaborn scipy pandas hypothesis pytest pytest-cov flask flask-cors
```

Install the `hazard_detection` package itself in editable mode so `import
hazard_detection...` resolves correctly from anywhere (scripts, tests, the
dashboard):

```bash
pip install -e .
```

> If there is no `pyproject.toml`/`setup.py` at the repo root in your
> checkout, the package still works by adding `src/` to `sys.path` — every
> script in `scripts/` and `tests/conftest.py` already does this
> (`sys.path.insert(0, "src")`), so `pip install -e .` is a convenience, not
> a hard requirement.

---

## Step 2 — Verify the Installation

```bash
python scripts/check_install.py
```

Expected output (GPU machine):
```
Python: 3.11.x
PyTorch:           2.x.x+cu118
CUDA available:    True
GPU:               NVIDIA GeForce RTX 3050
ultralytics:       OK
hazard_detection:  OK
opencv:            OK
```

On a CPU-only machine, `CUDA available` will read `False` and `GPU` will
read `None (CPU mode)` — everything else should still say `OK`. If
`hazard_detection: FAILED`, re-run `pip install -e .` from the repo root, or
confirm `src/` is on `PYTHONPATH`.

For a more thorough hardware/dependency check (training vs. inference
minimums, optional dev packages):

```bash
python scripts/verify_installation.py
```

---

## Step 3 — Get the Dataset

The base dataset (`roboflow data/`) is **not committed to this repository**
— it's excluded via `.gitignore` because it's large binary image data. Set
it up locally:

1. Export or download your Roboflow project in **YOLOv8/YOLOv12 TXT**
   format.
2. Place it at the repo root as `roboflow data/`, matching this layout:
   ```
   roboflow data/
   ├── data.yaml
   ├── train/images/ + train/labels/
   ├── valid/images/ + valid/labels/
   └── test/images/  + test/labels/
   ```
3. Verify it loaded correctly:
   ```bash
   python scripts/check_dataset.py
   ```
   Expected output:
   ```
   train:   85 images,    85 labels
   valid:   12 images,    12 labels
   test:    12 images,    12 labels
   ```

The 17 detection classes (hazard classes marked) are documented in
`docs/USER_GUIDE.md` (Step 2) and encoded in `roboflow data/data.yaml`.

If you have a separate, larger `image_data_normal/` dataset (hazard-free
background footage — used as raw input for the annotation pipelines below),
place it at the repo root too. It's optional: every script that defaults to
`image_data_normal` automatically falls back to `roboflow data` if the
former isn't present on the machine, with a printed warning.

---

## Step 4 — Annotation Pipeline (Optional, Only If You Need More Labeled Data)

Skip this step if `roboflow data/` already has enough labeled images for
your needs and you don't have unlabeled raw footage to annotate. Otherwise,
this step turns unlabeled images (`image_data_normal/` or any other
directory of images) into YOLO-format labels.

There are two interchangeable backends, selected via one entry point:

```bash
python scripts/annotation/run_auto_annotate.py --pipeline segmentation   # Grounded SAM 2 + Grounding DINO
python scripts/annotation/run_auto_annotate.py --pipeline cnn            # plain YOLO fallback
```

| | Segmentation (`auto_annotate.py`) | CNN fallback (`cnn_auto_annotate.py`) |
|---|---|---|
| Output | Pixel-accurate masks (4-corner polygons) | Bounding-box polygons |
| Setup cost | High — separate `.venv_annotation`, CUDA-compiled Grounding DINO, multi-GB weight downloads | Low — `pip install ultralytics` in the **main** `.venv`, needs a trained YOLO checkpoint |
| Python version | 3.10 required | Same as main `.venv` (3.10–3.12) |
| When to use | Best label quality; use when SAM 2 setup is feasible | Faster to stand up; use when you already have (or can quickly train) a YOLO checkpoint, or when SAM 2 is unstable/unavailable on the machine |

Full step-by-step instructions for **both** backends — including SAM 2/
Grounding DINO weight downloads, the CNN fallback's pre-training sanity
check, and the `image_data_normal` → `roboflow data` automatic fallback
behavior — are in **`scripts/annotation/SETUP.md`**. Read that file in full
before running either pipeline; this section is deliberately just a
pointer to avoid duplicating (and drifting out of sync with) that guide.

After annotating, review uncertain labels before using them for training:

```bash
python scripts/annotation/review_annotations.py --review_dir <output_dir>/review_queue
```

---

## Step 5 — Train the YOLO Detector

```bash
# GPU, default settings (150 epochs)
python scripts/train_yolo.py

# CPU, fewer epochs, smaller batch
python scripts/train_yolo.py --device cpu --batch 8 --epochs 50

# Resume an interrupted run
python scripts/train_yolo.py --resume

# Fine-tune an existing checkpoint on new data
python scripts/finetune_yolo.py --checkpoint runs/train/hazard_yolo/weights/best.pt
```

The best checkpoint is written to `runs/train/hazard_yolo/weights/best.pt`
(this path — and the whole `runs/` directory — is gitignored; checkpoints
must be produced locally or copied between machines, they are never
committed).

See `--help` on any script for the full option list, and
`docs/TRAINING_GUIDE.md` for hyperparameter guidance and troubleshooting
training-specific issues.

---

## Step 6 — Evaluate the Trained Model

```bash
python scripts/evaluate_yolo.py --checkpoint runs/train/hazard_yolo/weights/best.pt
```

Writes charts and a JSON summary to `evaluation_results/`:
`confusion_matrix.png`, `classification_metrics.png`,
`class_distribution.png`, `roc_curves.png`, `confidence_distribution.png`,
`evaluation_summary.json`. See `docs/USER_GUIDE.md` Step 6 for how to read
these when deciding whether a checkpoint is ready to deploy.

---

## Step 7 — Configure and Run the Detection System

Edit `config/hazard_detection.yaml` — at minimum, point `yolo.checkpoint_path`
at your trained checkpoint and set your camera sequence:

```yaml
yolo:
  checkpoint_path: "runs/train/hazard_yolo/weights/best.pt"
  device: "cuda"          # or "cpu"
  confidence_threshold: 0.5

cameras:
  sequence: ["cam_01", "cam_02"]
```

The full field reference (zone maps, alert channels, diagnostics, training
block) is in `docs/CONFIGURATION_GUIDE.md`.

Run the detection pipeline (from the repo root):

```bash
python src/hazard_detection/main.py --config config/hazard_detection.yaml
```

Optional flags: `--log-level DEBUG|INFO|WARNING|ERROR` and `--dump-dir <path>`
to override where diagnostic state snapshots are written. Stop with
`Ctrl+C` (SIGINT) — the pipeline logs final statistics before exiting.

If the configured checkpoint doesn't exist, the system logs a warning and
substitutes a stub detector that returns no detections — useful for
verifying the rest of the pipeline (frame sampling, zone logic, alerting)
without a trained model.

---

## Step 8 — Run the Web Dashboard (Optional)

A Flask app for uploading a single image and viewing hazard inference
results in a browser — independent of the live camera pipeline above.

```bash
python src/dashboard/app.py
```

Open `http://localhost:5000`. It looks for a checkpoint at
`checkpoints/yolov12_best.pt` by default (override with the
`DASHBOARD_*` environment variables — see `docs/DASHBOARD_GUIDE.md` for the
full list and the REST API reference). If no checkpoint is found there, the
app still starts; the `/api/inference` endpoint returns HTTP 500 until a
valid checkpoint path is configured.

---

## Running the Test Suite

```bash
pytest tests/unit -q
```

`tests/conftest.py` adds `src/` to `sys.path` automatically, so no extra
setup is needed as long as you're running `pytest` from the repo root.

---

## End-to-End Checklist

Use this to confirm a fresh machine is fully wired up:

- [ ] `python scripts/check_install.py` → all lines say `OK` / `True`
- [ ] `roboflow data/` present, `python scripts/check_dataset.py` shows non-zero counts for all 3 splits
- [ ] (if annotating new data) `scripts/annotation/SETUP.md` followed for your chosen backend, `--verify` passes
- [ ] `runs/train/hazard_yolo/weights/best.pt` exists (trained or copied from another machine)
- [ ] `python scripts/evaluate_yolo.py` produces `evaluation_results/evaluation_summary.json` with acceptable per-class metrics
- [ ] `config/hazard_detection.yaml`'s `yolo.checkpoint_path` points at that checkpoint
- [ ] `python src/hazard_detection/main.py` starts without a `ConfigurationError`
- [ ] `pytest tests/unit -q` passes (excluding any pre-existing unrelated failures noted in `.kiro/specs/*/tasks.md`)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `hazard_detection: FAILED` from `check_install.py` | Run `pip install -e .` from the repo root, or confirm `src/` is on `PYTHONPATH` |
| `CUDA available: False` but you have a GPU | Reinstall PyTorch with the `--index-url` matching your installed CUDA driver version; a plain `pip install torch` pulls the CPU-only wheel |
| `ConfigurationError` on startup | `config/hazard_detection.yaml` is missing a required field or a referenced path (checkpoint, zone map) doesn't exist — the error message names the exact field |
| Training checkpoint not found by the dashboard/runtime | Both default to different paths (`runs/train/hazard_yolo/weights/best.pt` for training scripts vs. `checkpoints/yolov12_best.pt` for the dashboard) — copy or symlink the checkpoint to whichever path the consumer expects, or override the path in config / env vars |
| `image_data_normal` not found | Expected on machines without access to that dataset — every script that defaults to it automatically falls back to `roboflow data` with a printed warning; no action needed unless you explicitly need the real dataset |
| Annotation pipeline setup issues | See the Troubleshooting table in `scripts/annotation/SETUP.md` (SAM 2/Grounding DINO compile errors, CUDA OOM, false detection tuning) |
| `pytest` import errors (`ModuleNotFoundError: hazard_detection`) | Run pytest from the repo root, not from inside `tests/` — `conftest.py`'s path fix is relative to its own location |

---

## Where to Go Next

- New to the system? Read `docs/USER_GUIDE.md` for the full narrative walkthrough with expected console output at every step.
- Tuning training? Read `docs/TRAINING_GUIDE.md`.
- Every config field explained? Read `docs/CONFIGURATION_GUIDE.md`.
- Setting up annotation (either backend)? Read `scripts/annotation/SETUP.md`.
- Curious about the CNN fallback's design rationale and requirements? Read `.kiro/specs/cnn-fallback-annotation-pipeline/`.
- Building/demoing the web dashboard? Read `docs/DASHBOARD_GUIDE.md`.
- Connecting a real camera to the dashboard (RTSP setup, credentials, another device)? Read `docs/LIVE_CAMERA_SETUP.md`.
