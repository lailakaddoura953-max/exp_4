# Auto-Annotation Pipeline Setup Guide
## Grounded SAM 2 + Grounding DINO (Local, No Internet Required After Setup)

This guide sets up the full pipeline on the private machine. Run every step
in order. Estimated total time: 20–40 minutes depending on internet speed.

---

## Prerequisites

- Windows 10/11 or Ubuntu 20.04+
- Python 3.10 (required — SAM 2 has compile issues on 3.11+ with some CUDA builds)
- CUDA 12.0 or later + matching cuDNN installed (run `nvcc --version` to confirm)
- At least 4 GB VRAM (**use `sam2.1_hiera_small` on 4 GB cards — see Step 7**)
- At least 8 GB free disk space for weights

If you only have CPU (no GPU), the pipeline still works but is very slow
(~30–60s per image). Set `DEVICE = "cpu"` in `auto_annotate.py`.

---

## Step 1 — Clone Grounded SAM 2

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```

Keep note of this directory — you'll set `GSAM2_REPO` in `auto_annotate.py`
to point here.

---

## Step 2 — Create a virtual environment

```bash
python -m venv .venv_annotation
# Windows:
.venv_annotation\Scripts\activate
# Linux/Mac:
source .venv_annotation/bin/activate
```

---

## Step 3 — Install PyTorch with CUDA

Go to https://pytorch.org/get-started/locally/ and copy the install command
for your CUDA version. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 4 — Install SAM 2

From inside the Grounded-SAM-2 repo directory:

```bash
pip install -e .
```

---

## Step 5 — Install Grounding DINO

This requires CUDA compilation. Make sure `CUDA_HOME` is set first:

```bash
# Windows (adjust path to your CUDA install):
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

# Linux:
export CUDA_HOME=/usr/local/cuda-12.1
```

Then install:
```bash
pip install --no-build-isolation -e grounding_dino
```

If this fails on Windows with a compiler error, install Visual Studio Build
Tools (the C++ workload) from https://visualstudio.microsoft.com/downloads/
and retry.

---

## Step 6 — Install remaining dependencies

```bash
pip install opencv-python supervision transformers numpy pillow tqdm
```

---

## Step 7 — Download SAM 2 weights

From inside `Grounded-SAM-2/checkpoints/`:

```bash
# Windows (PowerShell):
cd checkpoints
bash download_ckpts.sh
# If bash isn't available on Windows, download manually:
```

Manual download URLs (save into `Grounded-SAM-2/checkpoints/`):
- **sam2.1_hiera_large.pt** (recommended, ~2.4 GB):
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
- **sam2.1_hiera_base_plus.pt** (smaller, ~800 MB, faster but less accurate):
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Use large if you have 16 GB VRAM, base_plus if you have 8 GB.

---

## Step 8 — Download Grounding DINO weights

From inside `Grounded-SAM-2/gdino_checkpoints/`:

```bash
cd gdino_checkpoints
bash download_ckpts.sh
```

Manual download (save into `Grounded-SAM-2/gdino_checkpoints/`):
- **groundingdino_swint_ogc.pth** (~700 MB):
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- **GroundingDINO_SwinT_OGC.py** (config file, already in repo at
  `grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py`)

---

## Step 9 — Configure paths in auto_annotate.py

Open `scripts/annotation/auto_annotate.py` and set these at the top:

```python
GSAM2_REPO       = r"C:\path\to\Grounded-SAM-2"
SAM2_CHECKPOINT  = r"C:\path\to\Grounded-SAM-2\checkpoints\sam2.1_hiera_large.pt"
SAM2_CONFIG      = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_CHECKPOINT = r"C:\path\to\Grounded-SAM-2\gdino_checkpoints\groundingdino_swint_ogc.pth"
GDINO_CONFIG     = r"C:\path\to\Grounded-SAM-2\grounding_dino\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DEVICE           = "cuda"   # or "cpu"
```

---

## Step 10 — Verify the setup

```bash
python scripts/annotation/auto_annotate.py --verify
```

This loads both models and runs inference on a single test image without
reading any real data. If you see "Models loaded OK" the setup is complete.

---

## Running the pipeline

```bash
# Auto-annotate all images, 2 workers
python scripts/annotation/auto_annotate.py \
    --input_dir image_data_normal \
    --output_dir image_data_annotated \
    --confidence 0.35 \
    --review_threshold 0.55

# Then review uncertain annotations
python scripts/annotation/review_annotations.py \
    --review_dir image_data_annotated/review_queue
```

See each script's `--help` for all options.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Switch to `sam2.1_hiera_base_plus.pt` or reduce `--batch_size` to 1 |
| Grounding DINO compile error | Ensure `CUDA_HOME` is set and matches your PyTorch CUDA version |
| Black/blank masks | Image may have Ocularis UI overlay — increase `--chrome_margin` |
| Too many false detections | Raise `--confidence` threshold (try 0.45–0.55) |
| Too few detections | Lower `--confidence` (try 0.25–0.30) and check text prompts |
| `ModuleNotFoundError: groundingdino` | Run from inside the Grounded-SAM-2 repo or add it to PYTHONPATH |

---

## Fallback: CNN-Based Pipeline

If the Grounded SAM 2 + Grounding DINO stack above turns out to be
unstable, too slow, or not worth maintaining on this machine, use
`scripts/annotation/cnn_auto_annotate.py` instead. It runs a conventional
Ultralytics YOLO detector — no SAM 2, no Grounding DINO, no `.venv_annotation`,
no CUDA compilation step. It produces bounding-box labels (encoded as
4-corner rectangle polygons) rather than pixel-accurate masks, but the
output directory layout and label format are unchanged, so
`review_annotations.py` and the training scripts work with either backend.

### Setup

1. Install `ultralytics` in the project's **main** `.venv` (not
   `.venv_annotation` — the CNN fallback runs entirely in the main
   environment):
   ```bash
   pip install ultralytics
   ```
   Verify it's importable:
   ```bash
   python -c "import ultralytics; print(ultralytics.__version__)"
   ```

2. Produce a `YOLO_Checkpoint` trained on `roboflow data/data.yaml` if you
   don't already have one:
   ```bash
   python scripts/train_yolo.py
   ```
   This writes `runs/train/hazard_yolo/weights/best.pt`. If you already
   have a trained checkpoint (e.g. copied from another machine), just make
   sure it ends up at that path, or pass `--checkpoint <path>` explicitly
   when running the fallback.

3. Verify the checkpoint loads and can run inference, without touching any
   real data:
   ```bash
   python scripts/annotation/cnn_auto_annotate.py --verify \
       --checkpoint runs/train/hazard_yolo/weights/best.pt
   ```

### Running the fallback

```bash
python scripts/annotation/run_auto_annotate.py --pipeline cnn \
    --checkpoint runs/train/hazard_yolo/weights/best.pt \
    --confidence 0.35 \
    --review_threshold 0.55
```

(`cnn_auto_annotate.py` can also be run directly with the same flags; the
`run_auto_annotate.py` entry point is just a single command that can switch
between `--pipeline segmentation` and `--pipeline cnn`.)

### `image_data_normal` availability

The real normal-operations dataset (`image_data_normal`) lives on a
separate, access-restricted device and is not present on every machine.
Both `cnn_auto_annotate.py` and `run_auto_annotate.py` default `--input_dir`
to `image_data_normal` if it exists, and automatically fall back to
`roboflow data` otherwise — you'll see a `[warn]` line naming both paths
when this happens. If you explicitly pass `--input_dir` yourself and that
path doesn't exist, the fallback does **not** kick in — the script exits
with an error instead, since an explicit path is assumed to be intentional.

### Pre-training hazard sanity check

Because `image_data_normal` is hazard-free by definition, there's nothing
in it to validate detection against. Before trusting the CNN fallback at
scale, run the sanity check, which injects synthetic hazard instances
(open container, human, human without PPE) into copies of the normal
images and measures detection recall against the known injected ground
truth:

```bash
python scripts/pretrain_hazard_sanity_check.py
```

This also defaults `--normal_dir` to `image_data_normal` → `roboflow data`
fallback behavior. Note: if it falls back to `roboflow data`, the result
only confirms the pipeline's mechanics work (injection → training → recall
evaluation) — `roboflow data` images already contain real hazards, so
they aren't representative hazard-free background, and recall numbers from
that run don't predict real-world performance. A PASS/FAIL verdict and
per-class recall are printed and written to a summary JSON. See
`.kiro/specs/cnn-fallback-annotation-pipeline/` for the full requirements
and design behind this fallback.
