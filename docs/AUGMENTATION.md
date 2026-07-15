# Data Augmentation Engine Documentation

## Overview

The `AugmentationEngine` generates synthetic camera misalignment examples from naturally-aligned KITTI images. This is essential for training the deep learning model, as the KITTI dataset contains only properly aligned camera imagery.

## Purpose

- **Problem**: KITTI images are naturally aligned (no misalignment)
- **Solution**: Apply transformations to simulate misalignment
- **Benefit**: Creates labeled training data with ground truth pose offsets

## Architecture

```
Input: Aligned KITTI Image
   ↓
Apply Geometric Transformations (affect pose)
   ├─ Rotation (-10° to +10°)
   ├─ Translation (-50 to +50 pixels)
   ├─ Random Crop (90-100% scale)
   └─ Horizontal Flip (50% probability)
   ↓
Apply Photometric Transformations (add realism)
   ├─ Brightness (0.7 to 1.3 factor)
   ├─ Contrast (0.8 to 1.2 factor)
   └─ Gaussian Noise (σ=0.01)
   ↓
Generate Ground Truth Labels
   ├─ is_misaligned = 1
   ├─ misalignment_probability (based on magnitude)
   └─ pose = [X, Y, Z, roll, pitch, yaw]
   ↓
Output: Augmented Image + Labels
```

## Transformations

### Geometric Transformations

These transformations affect camera pose and are reflected in ground truth labels:

#### 1. Rotation
- **Range**: -10° to +10°
- **Affects**: Camera yaw (pose[5])
- **Purpose**: Simulates camera rotation misalignment
- **Implementation**: PyTorch affine transformation with bilinear interpolation

#### 2. Translation
- **Range**: -50 to +50 pixels in X and Y directions
- **Affects**: Camera X and Y position (pose[0:2])
- **Conversion**: Pixels → meters using 0.01 m/pixel approximation
- **Purpose**: Simulates camera position shift
- **Implementation**: PyTorch affine transformation

#### 3. Random Crop
- **Range**: 90% to 100% of original scale
- **Affects**: Camera Z position (pose[2])
- **Reasoning**: Closer camera = larger apparent scale
- **Purpose**: Simulates depth variation
- **Implementation**: Crop followed by resize to original dimensions

#### 4. Horizontal Flip
- **Probability**: 50%
- **Affects**: Mirrors X position and yaw (pose[0] and pose[5])
- **Purpose**: Doubles training data diversity
- **Implementation**: PyTorch flip operation on width dimension

### Photometric Transformations

These add realism but don't affect pose labels:

#### 5. Brightness Adjustment
- **Range**: 0.7 to 1.3 multiplicative factor
- **Purpose**: Simulates lighting variations
- **Implementation**: Multiply normalized image by factor

#### 6. Contrast Adjustment
- **Range**: 0.8 to 1.2 multiplicative factor
- **Purpose**: Simulates exposure variations
- **Implementation**: (image - mean) × factor + mean

#### 7. Gaussian Noise
- **Std Dev**: σ = 0.01
- **Purpose**: Simulates sensor noise
- **Implementation**: Add random normal noise to image

## Usage

### Basic Usage

```python
from dl_misalignment.data import AugmentationEngine

# Create augmentation engine
aug_engine = AugmentationEngine(
    apply_probability=0.5,  # Apply to 50% of samples
    split='train',          # Only augment train/val (not test)
    log_every=1000,         # Log statistics every 1000 samples
    seed=42                 # For reproducibility
)

# Apply to image and label
augmented_image, augmented_label = aug_engine(image, label)
```

### Integration with KITTIDataset

```python
from dl_misalignment.data import KITTIDataset, create_augmentation_engine

# Create augmentation for training
train_aug = create_augmentation_engine('train', apply_probability=0.5)

# Create dataset with augmentation
train_dataset = KITTIDataset(
    root_dir='kitti_data/',
    split='train',
    transform=train_aug,  # Pass augmentation here
    target_resolution=(640, 640)
)

# Load augmented samples
for image, label in train_dataset:
    # image: augmented image tensor [3, H, W]
    # label: contains ground truth pose and misalignment info
    pass
```

### Factory Function

```python
from dl_misalignment.data import create_augmentation_engine

# Automatically handles split-specific behavior
train_aug = create_augmentation_engine('train')    # Returns AugmentationEngine
val_aug = create_augmentation_engine('val')        # Returns AugmentationEngine
test_aug = create_augmentation_engine('test')      # Returns None (no augmentation)
```

## Ground Truth Label Generation

The augmentation engine automatically generates ground truth labels:

```python
{
    'is_misaligned': 1,  # 0 = aligned, 1 = misaligned
    
    'misalignment_probability': 0.45,  # [0, 1] based on pose magnitude
    
    'pose': torch.tensor([
        0.12,   # X position offset (meters)
        -0.05,  # Y position offset (meters)
        0.08,   # Z position offset (meters)
        0.0,    # Roll (degrees)
        0.0,    # Pitch (degrees)
        7.5     # Yaw (degrees)
    ]),
    
    'augmentation_applied': {
        'rotation': 7.5,
        'translation': (12, -5),
        'brightness': 1.1,
        'random_crop': 0.96
    }
}
```

### Probability Calculation

Misalignment probability is computed from pose magnitude:

```python
magnitude = sqrt(X² + Y² + Z² + roll² + pitch² + yaw²)
probability = min(1.0, magnitude / 10.0)
```

This provides a continuous confidence score suitable for training.

## Statistics Tracking

The engine tracks augmentation statistics:

```python
stats = aug_engine.get_statistics()

# Returns:
{
    'total_samples': 10000,      # Total samples processed
    'augmented_samples': 5012,   # Samples that were augmented
    'rotation': 2508,            # Times rotation was applied
    'translation': 2495,         # Times translation was applied
    'brightness': 2503,          # Times brightness was applied
    'contrast': 2489,            # Times contrast was applied
    'noise': 2501,               # Times noise was applied
    'horizontal_flip': 2506,     # Times flip was applied
    'random_crop': 2498          # Times crop was applied
}
```

Statistics are automatically logged every N samples (configurable).

## Requirements Mapping

The augmentation engine satisfies these requirements:

- **Requirement 6.1**: Rotation augmentation (-10° to +10°) ✓
- **Requirement 6.2**: Translation augmentation (-50 to +50 pixels) ✓
- **Requirement 6.3**: Brightness augmentation (0.7 to 1.3 factor) ✓
- **Requirement 6.4**: Contrast augmentation (0.8 to 1.2 factor) ✓
- **Requirement 6.5**: Ground truth label generation ✓
- **Requirement 6.6**: Apply only to train/val (exclude test) ✓
- **Requirement 6.7**: Preserve image dimensions ✓
- **Requirement 21.1**: At least 3 transformations per sample ✓
- **Requirement 21.2**: Random combination selection ✓
- **Requirement 21.3**: 50% application probability ✓
- **Requirement 21.4**: Gaussian noise (σ=0.01) ✓
- **Requirement 21.5**: Horizontal flip (50% probability) ✓
- **Requirement 21.6**: Random cropping (90-100% scale) ✓
- **Requirement 21.7**: Log statistics every 1000 steps ✓

## Performance Considerations

### Memory Efficiency
- All transformations operate in-place when possible
- No intermediate copies for most operations
- Negligible memory overhead (~few KB per sample)

### Computational Cost
- Primarily GPU operations (when CUDA available)
- Typical overhead: 5-10ms per sample
- Parallelized within DataLoader workers

### Determinism
- Reproducible with fixed seed
- Same seed → same transformations
- Useful for debugging and validation

## Testing

Run the test suite:

```bash
# Unit tests
pytest tests/test_augmentation.py -v

# Integration test
python scripts/test_augmentation_integration.py
```

## Troubleshooting

### Issue: Test split being augmented

**Cause**: Wrong split parameter or missing factory function
**Solution**: Use `create_augmentation_engine('test')` which returns None

### Issue: Images look corrupted

**Cause**: Transformations applied to already-normalized images
**Solution**: Ensure augmentation is applied after normalization in dataset

### Issue: Pose labels seem incorrect

**Cause**: Transformation order or sign errors
**Solution**: Check that transformations match pose update logic

### Issue: Low transformation diversity

**Cause**: Too low apply_probability or unlucky random seed
**Solution**: Increase apply_probability or try different seed

## Advanced Usage

### Custom Augmentation Ranges

Modify the augmentation engine for custom ranges:

```python
aug = AugmentationEngine(split='train')

# Override default ranges (careful - may violate requirements)
aug.rotation_range = (-15.0, 15.0)      # Wider rotation
aug.translation_range = (-100, 100)     # Larger translation
aug.brightness_range = (0.5, 1.5)       # More extreme brightness
```

### Conditional Augmentation

Apply different augmentation rates for different scenarios:

```python
# Higher augmentation for small datasets
if len(dataset) < 1000:
    aug = AugmentationEngine(apply_probability=0.8)
else:
    aug = AugmentationEngine(apply_probability=0.5)
```

### Debugging Transformations

Visualize what transformations are doing:

```python
import matplotlib.pyplot as plt

# Load sample
image, label = dataset[0]

# Apply augmentation
aug_image, aug_label = aug_engine(image, label)

# Denormalize for visualization
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img * std + mean

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(denormalize(image).permute(1, 2, 0))
axes[0].set_title('Original')
axes[1].imshow(denormalize(aug_image).permute(1, 2, 0))
axes[1].set_title(f'Augmented\nPose: {aug_label["pose"].numpy()}')
plt.show()
```

## References

- Requirements Document: Section 6 (Synthetic Misalignment Augmentation)
- Requirements Document: Section 21 (Training Data Augmentation Diversity)
- Design Document: Data Models → Training Sample Structure
- PyTorch Documentation: [Geometric Transformations](https://pytorch.org/docs/stable/nn.functional.html#grid-sample)
