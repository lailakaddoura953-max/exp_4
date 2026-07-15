# Training Guide: Deep Learning Misalignment Detection System

This guide covers training both Architecture A (LiteFlowNet2) and Architecture B (SpyNet) for camera misalignment detection.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Hardware Requirements](#hardware-requirements)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Training Architecture A](#training-architecture-a)
6. [Training Architecture B](#training-architecture-b)
7. [Monitoring Training](#monitoring-training)
8. [Checkpoint Management](#checkpoint-management)
9. [Training Parameters](#training-parameters)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software Requirements

```bash
# Python 3.8+
python --version

# Required packages
pip install torch torchvision tensorboard numpy opencv-python pillow pyyaml

# Verify installation
python scripts/verify_installation.py
```

### Verify Training Pipeline

Before starting full training, verify the pipeline is working:

```bash
python scripts/verify_training_pipeline.py
```

This runs quick tests to ensure:
- Loss functions work correctly
- Checkpoint saving/loading functions
- Training loop executes without errors
- TensorBoard logging is configured

---

## Hardware Requirements

### Architecture A (LiteFlowNet2)

**Training:**
- GPU: 8GB+ VRAM (NVIDIA GTX 1080 Ti, RTX 3060, or better)
- RAM: 16GB+ system memory
- CUDA: Compute capability 6.1+ (Pascal architecture or newer)
- Storage: 50GB+ free space (dataset + checkpoints)

**Inference:**
- GPU: 4GB+ VRAM
- RAM: 8GB+ system memory

### Architecture B (SpyNet)

**Training:**
- GPU: 6GB+ VRAM (more efficient than Architecture A)
- RAM: 16GB+ system memory
- CUDA: Compute capability 6.1+ (Pascal architecture or newer)
- Storage: 50GB+ free space

**Inference:**
- GPU: 3GB+ VRAM (more efficient than Architecture A)
- RAM: 8GB+ system memory

### Checking Your Hardware

```bash
python -c "from dl_misalignment.utils.hardware import get_gpu_info; print(get_gpu_info())"
```

---

## Dataset Preparation

### Download KITTI Dataset

1. Visit [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
2. Download the stereo dataset (left/right camera images)
3. Extract to `kitti_data/` directory

Expected structure:
```
kitti_data/
├── 2011_09_26/
│   ├── 2011_09_26_drive_0001_sync/
│   │   ├── image_02/  (left camera)
│   │   │   └── data/
│   │   │       ├── 0000000000.png
│   │   │       ├── 0000000001.png
│   │   │       └── ...
│   │   └── image_03/  (right camera)
│   └── ...
└── ...
```

### Verify Dataset

```bash
python -c "from dl_misalignment.data.kitti_dataset import KITTIDataset; d = KITTIDataset('kitti_data'); print(f'Loaded {len(d)} samples')"
```

### Dataset Splits

The training pipeline automatically creates train/val/test splits:
- **Training:** 70% (used for training)
- **Validation:** 15% (used for hyperparameter tuning and early stopping)
- **Test:** 15% (reserved for final evaluation, never seen during training)

Splits are deterministic and saved to `kitti_data/split_indices.npz` for reproducibility.

---

## Training Configuration

### Key Parameters

All parameters can be specified via command-line arguments or configuration files.

**Critical Parameters:**
- `--batch-size`: Number of samples per batch (auto-detected based on VRAM)
- `--learning-rate`: Initial learning rate (default: 1e-4)
- `--num-epochs`: Maximum training epochs (default: 50)
- `--target-resolution`: Image resolution [H W] (max: 750x750)

**Memory Optimization:**
- Mixed precision training (FP16/FP32) is enabled by default
- Gradient checkpointing reduces VRAM usage
- Automatic batch size adjustment based on available VRAM

**Early Stopping:**
- Validation runs every 500 steps
- Learning rate reduced by 0.5× after 5 evaluations without improvement
- Training stops after 10 evaluations without improvement

**Checkpointing:**
- Saves checkpoint every 1000 steps
- Keeps 3 most recent checkpoints
- Saves best model separately (lowest validation loss)

---

## Training Architecture A

### Basic Training

```bash
python scripts/train_architecture_a.py \
    --data-dir kitti_data \
    --checkpoint-dir checkpoints/architecture_a \
    --tensorboard-dir runs \
    --num-epochs 50
```

### With Custom Parameters

```bash
python scripts/train_architecture_a.py \
    --data-dir kitti_data \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --num-epochs 100 \
    --target-resolution 640 640 \
    --num-workers 4
```

### Resume from Checkpoint

```bash
python scripts/train_architecture_a.py \
    --resume checkpoints/architecture_a/best_model.pth \
    --num-epochs 50
```

### Expected Training Time

- **8GB VRAM GPU:** ~20-24 hours for 50 epochs
- **16GB VRAM GPU:** ~15-18 hours for 50 epochs
- **Batch size 2:** ~24 hours
- **Batch size 4:** ~18 hours

---

## Training Architecture B

### Basic Training

```bash
python scripts/train_architecture_b.py \
    --data-dir kitti_data \
    --checkpoint-dir checkpoints/architecture_b \
    --tensorboard-dir runs \
    --num-epochs 50
```

### Advantages of Architecture B

- **Lower VRAM:** 6GB vs 8GB for training
- **Faster inference:** ~30ms vs ~50ms per batch
- **Larger batch sizes:** Can use batch_size=4 on 6GB GPU

### Expected Training Time

- **6GB VRAM GPU:** ~18-22 hours for 50 epochs
- **8GB VRAM GPU:** ~15-18 hours for 50 epochs
- Generally 10-20% faster than Architecture A

---

## Monitoring Training

### TensorBoard

Monitor training in real-time with TensorBoard:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

**Metrics Logged:**

1. **Training Losses (every 10 steps):**
   - Total loss
   - Classification loss (BCE)
   - Regression loss (Smooth L1)

2. **Validation Metrics (every 500 steps):**
   - Validation loss
   - Classification loss
   - Regression loss
   - Accuracy

3. **Learning Rate (every 100 steps):**
   - Current learning rate
   - Tracks automatic reductions

4. **GPU Memory (every 100 steps):**
   - Allocated memory (GB)
   - Reserved memory (GB)

5. **Sample Predictions (every 1000 steps):**
   - Visual comparison of predictions vs ground truth

### Command Line Output

Training progress is also logged to console:

```
Step 500: Val Loss=0.3245, Accuracy=0.9123
Step 1000: Val Loss=0.2987, Accuracy=0.9245
✓ Saved checkpoint: checkpoints/architecture_a/checkpoint_step_1000.pth
Step 1500: Val Loss=0.2756, Accuracy=0.9367
✓ Saved best model: checkpoints/architecture_a/best_model.pth (val_loss=0.2756)
```

### Monitoring GPU Usage

In a separate terminal:

```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows PowerShell
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

---

## Checkpoint Management

### Checkpoint Structure

Checkpoints contain:
- Model weights (feature extractor, flow network, pose estimator)
- Optimizer state
- Learning rate scheduler state
- Training step and epoch
- Best validation loss
- Training history
- Configuration and metadata

### Checkpoint Files

**Automatic checkpoints (every 1000 steps):**
```
checkpoints/architecture_a/
├── checkpoint_step_1000.pth
├── checkpoint_step_2000.pth
├── checkpoint_step_3000.pth  (only 3 most recent kept)
└── best_model.pth  (lowest validation loss)
```

### Loading Checkpoints

For inference or continued training:

```python
from dl_misalignment.training import CheckpointManager
from dl_misalignment.models import CNNFeatureExtractor, LiteFlowNet2, PoseEstimator

# Load models
feature_extractor = CNNFeatureExtractor()
flow_network = LiteFlowNet2()
pose_estimator = PoseEstimator()

# Load checkpoint
manager = CheckpointManager('checkpoints/architecture_a')
checkpoint = manager.load_checkpoint(
    'checkpoints/architecture_a/best_model.pth',
    feature_extractor,
    flow_network,
    pose_estimator
)

print(f"Loaded checkpoint from step {checkpoint['training_step']}")
print(f"Validation loss: {checkpoint['best_validation_loss']:.4f}")
```

---

## Training Parameters

### Complete Parameter Reference

```bash
python scripts/train_architecture_a.py --help
```

**Data:**
- `--data-dir`: Path to KITTI dataset (default: `kitti_data`)
- `--num-workers`: Number of data loading workers (default: 2)
- `--no-augmentation`: Disable data augmentation

**Model:**
- `--target-resolution H W`: Image resolution (default: 640 640, max: 750 750)

**Training:**
- `--batch-size`: Samples per batch (auto-detected if not specified)
- `--num-epochs`: Maximum epochs (default: 50)
- `--learning-rate`: Initial LR (default: 1e-4)
- `--device`: Training device (cuda or cpu, default: cuda)

**Output:**
- `--checkpoint-dir`: Checkpoint directory (default: `checkpoints/architecture_a`)
- `--tensorboard-dir`: TensorBoard logs (default: `runs`)

**Resumption:**
- `--resume PATH`: Resume from checkpoint

### Loss Function Weights

Default weights (can be changed in config):
- Classification (BCE): 0.6
- Regression (Smooth L1): 0.4

To customize, edit the training script or config file.

### Augmentation Settings

Data augmentation is applied with 50% probability:
- **Rotation:** -10° to +10°
- **Translation:** -50 to +50 pixels (X and Y)
- **Brightness:** 0.7× to 1.3×
- **Contrast:** 0.8× to 1.2×
- **Gaussian noise:** σ=0.01
- **Horizontal flip:** 50% probability
- **Random crop:** 90-100% scale

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size:
   ```bash
   python scripts/train_architecture_a.py --batch-size 1
   ```

2. Reduce image resolution:
   ```bash
   python scripts/train_architecture_a.py --target-resolution 512 512
   ```

3. Use Architecture B (lower VRAM):
   ```bash
   python scripts/train_architecture_b.py
   ```

4. Close other GPU-using applications

### Slow Training

**Problem:** Training is slower than expected

**Solutions:**
1. Increase num_workers:
   ```bash
   python scripts/train_architecture_a.py --num-workers 4
   ```

2. Check GPU utilization (should be >80%):
   ```bash
   nvidia-smi
   ```

3. Ensure data is on SSD, not HDD

4. Use larger batch size if VRAM allows

### Training Not Converging

**Problem:** Loss not decreasing, poor accuracy

**Checks:**
1. Verify dataset loaded correctly:
   ```python
   from dl_misalignment.data.kitti_dataset import KITTIDataset
   dataset = KITTIDataset('kitti_data', split='train')
   print(f"Dataset size: {len(dataset)}")
   sample = dataset[0]
   print(f"Sample: {sample[1]}")  # Check labels
   ```

2. Check learning rate:
   - If too high (>1e-3): loss explodes
   - If too low (<1e-6): training too slow

3. Monitor loss components in TensorBoard:
   - Both classification and regression should decrease
   - If one is stuck, adjust loss weights

4. Verify augmentation not too aggressive:
   ```bash
   python scripts/train_architecture_a.py --no-augmentation
   ```

### Checkpoint Loading Errors

**Problem:** `RuntimeError: Error(s) in loading state_dict`

**Solution:** Ensure checkpoint is from correct architecture:
- Architecture A checkpoints only work with LiteFlowNet2
- Architecture B checkpoints only work with SpyNet
- Check `model_config` in checkpoint for architecture type

### Low Accuracy on Validation

**Problem:** Validation accuracy <90%

**Possible Causes:**
1. **Insufficient training:** Continue training for more epochs
2. **Overfitting:** Check if training accuracy >> validation accuracy
   - Solution: More augmentation, stronger dropout
3. **Data quality:** Verify KITTI dataset is complete
4. **Hyperparameters:** Try different learning rates

### TensorBoard Not Showing Data

**Problem:** TensorBoard shows no graphs

**Solutions:**
1. Check log directory exists:
   ```bash
   ls runs/
   ```

2. Verify training is writing logs:
   ```bash
   ls runs/architecture_a_liteflownet2/
   ```

3. Restart TensorBoard:
   ```bash
   tensorboard --logdir runs --reload_interval 5
   ```

---

## Next Steps

After training completes:

1. **Evaluate Models:**
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/architecture_a/best_model.pth
   python scripts/evaluate.py --checkpoint checkpoints/architecture_b/best_model.pth
   ```

2. **Compare Architectures:**
   ```bash
   python scripts/compare_architectures.py \
       --checkpoint-a checkpoints/architecture_a/best_model.pth \
       --checkpoint-b checkpoints/architecture_b/best_model.pth
   ```

3. **Run Inference:**
   ```bash
   python scripts/run_inference.py \
       --checkpoint checkpoints/architecture_a/best_model.pth \
       --input test_images/
   ```

4. **Deploy Model:**
   See `docs/DEPLOYMENT.md` for deployment instructions

---

## Additional Resources

- **Model Architecture:** See `deep-learning-misalignment-detection/design.md`
- **Requirements:** See `deep-learning-misalignment-detection/requirements.md`
- **API Reference:** See `docs/API.md`
- **Deployment:** See `docs/DEPLOYMENT.md`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs and TensorBoard
3. Verify hardware requirements are met
4. Check dataset integrity
