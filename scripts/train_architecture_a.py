"""
Training Script for Architecture A (LiteFlowNet2)

This script trains the complete Architecture A pipeline:
- CNN Feature Extractor
- LiteFlowNet2 optical flow network
- Pose Estimator

Target Performance:
- Training VRAM: ≤8GB
- Inference VRAM: ≤4GB
- Training time: ≤24 hours on consumer GPU

Task 9.6: Training scripts for both architectures
Requirements: 4.1, 18.1-18.6
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dl_misalignment.models.cnn_feature_extractor import CNNFeatureExtractor
from dl_misalignment.models.liteflownet2 import LiteFlowNet2
from dl_misalignment.models.pose_estimator import PoseEstimator
from dl_misalignment.data.kitti_dataset import create_dataloaders
from dl_misalignment.data.augmentation import AugmentationEngine
from dl_misalignment.training.trainer import Trainer
from dl_misalignment.utils.hardware import check_hardware_requirements, get_gpu_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Architecture A (LiteFlowNet2)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='kitti_data',
        help='Path to KITTI dataset directory'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/architecture_a',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='runs',
        help='Directory for TensorBoard logs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--target-resolution',
        type=int,
        nargs=2,
        default=[640, 640],
        help='Target image resolution (H W), max 750x750'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 80)
    print("Architecture A (LiteFlowNet2) Training")
    print("=" * 80)
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Install with: pip install torch torchvision")
        sys.exit(1)
    
    # Check hardware requirements
    logger.info("Checking hardware requirements...")
    hardware_ok, hardware_info = check_hardware_requirements(
        min_vram_gb=8.0,
        min_ram_gb=16.0,
        min_cuda_compute=6.1
    )
    
    if not hardware_ok:
        logger.error("Hardware requirements not met!")
        logger.error("Architecture A requires:")
        logger.error("  - 8GB+ VRAM for training")
        logger.error("  - 16GB+ system RAM")
        logger.error("  - CUDA compute capability 6.1+ (Pascal or newer)")
        logger.error(f"\nYour system: {hardware_info}")
        sys.exit(1)
    
    logger.info("✓ Hardware requirements met")
    logger.info(f"GPU: {hardware_info['gpu_name']}")
    logger.info(f"VRAM: {hardware_info['vram_total_gb']:.1f} GB")
    logger.info(f"CUDA: {hardware_info['cuda_version']}")
    
    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Auto-detect batch size based on VRAM
    if args.batch_size is None:
        vram_gb = hardware_info.get('vram_total_gb', 8.0)
        if vram_gb <= 8:
            args.batch_size = 2
        elif vram_gb <= 16:
            args.batch_size = 4
        else:
            args.batch_size = 6
        logger.info(f"Auto-detected batch size: {args.batch_size}")
    
    # Validate target resolution
    if args.target_resolution[0] > 750 or args.target_resolution[1] > 750:
        logger.error(f"Target resolution {args.target_resolution} exceeds 750x750 limit")
        sys.exit(1)
    
    # Create augmentation engine
    augmentation = None
    if not args.no_augmentation:
        logger.info("Creating augmentation engine...")
        augmentation = AugmentationEngine(
            augmentation_probability=0.5,
            rotation_range=(-10, 10),
            translation_range=(-50, 50),
            brightness_range=(0.7, 1.3),
            contrast_range=(0.8, 1.2),
            noise_std=0.01,
            horizontal_flip_prob=0.5,
            crop_scale_range=(0.9, 1.0)
        )
    
    # Create data loaders
    logger.info("Loading KITTI dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        target_resolution=tuple(args.target_resolution),
        num_workers=args.num_workers,
        train_transform=augmentation,
        val_transform=None  # No augmentation for validation
    )
    
    logger.info(f"✓ Dataset loaded")
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    # Create models
    logger.info("Initializing Architecture A models...")
    
    feature_extractor = CNNFeatureExtractor(input_channels=3)
    flow_network = LiteFlowNet2()
    pose_estimator = PoseEstimator(
        feature_channels=64,
        flow_channels=2,
        shared_dim=256,
        branch_dim=128,
        dropout_rate=0.3
    )
    
    logger.info("✓ Models initialized")
    logger.info(f"  CNN Feature Extractor: {sum(p.numel() for p in feature_extractor.parameters()):,} parameters")
    logger.info(f"  LiteFlowNet2: {sum(p.numel() for p in flow_network.parameters()):,} parameters")
    logger.info(f"  Pose Estimator: {sum(p.numel() for p in pose_estimator.parameters()):,} parameters")
    
    # Training configuration
    config = {
        'architecture': 'Architecture_A_LiteFlowNet2',
        'target_resolution': tuple(args.target_resolution),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'classification_weight': 0.6,
        'regression_weight': 0.4,
        'mixed_precision': True,
        'checkpoint_dir': args.checkpoint_dir,
        'tensorboard_dir': args.tensorboard_dir,
        'run_name': 'architecture_a_liteflownet2',
        'validation_interval': 500,
        'early_stopping_patience': 10
    }
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        feature_extractor=feature_extractor,
        flow_network=flow_network,
        pose_estimator=pose_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=str(device)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            feature_extractor=feature_extractor,
            flow_network=flow_network,
            pose_estimator=pose_estimator,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler
        )
        trainer.training_step = checkpoint['training_step']
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_validation_loss']
        logger.info(f"✓ Resumed from step {trainer.training_step}, epoch {trainer.epoch}")
    
    # Start training
    logger.info("")
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Steps per epoch: ~{len(train_loader)}")
    logger.info(f"Validation every {config['validation_interval']} steps")
    logger.info(f"Early stopping patience: {config['early_stopping_patience']} evaluations")
    logger.info("")
    logger.info("Monitor training with: tensorboard --logdir runs")
    logger.info("")
    
    try:
        trainer.train(num_epochs=args.num_epochs)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
    
    # Training complete
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best model saved at: {trainer.checkpoint_manager.best_checkpoint_path}")
    logger.info(f"Total training steps: {trainer.training_step}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate model: python scripts/evaluate.py --checkpoint <best_model.pth>")
    logger.info("  2. View training curves: tensorboard --logdir runs")
    logger.info("  3. Compare architectures: python scripts/compare_architectures.py")


if __name__ == '__main__':
    main()
