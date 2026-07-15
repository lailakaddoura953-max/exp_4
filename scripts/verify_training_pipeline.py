"""
Training Pipeline Verification Script

This script verifies that the training pipeline is working correctly by:
1. Creating small dummy datasets
2. Running a few training steps
3. Verifying checkpoint saving/loading
4. Checking TensorBoard logging
5. Testing both architectures

This is a smoke test to ensure everything is wired correctly before
running the full 24-hour training session.

Run with: python scripts/verify_training_pipeline.py
"""

import sys
import logging
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dl_misalignment.models.cnn_feature_extractor import CNNFeatureExtractor
from dl_misalignment.models.liteflownet2 import LiteFlowNet2
from dl_misalignment.models.spynet import SpyNet
from dl_misalignment.models.pose_estimator import PoseEstimator
from dl_misalignment.training.trainer import Trainer, MisalignmentLoss, CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_data_loader(batch_size=2, num_samples=10, resolution=(320, 320)):
    """Create dummy data loader for testing."""
    logger.info(f"Creating dummy data loader: {num_samples} samples, batch_size={batch_size}")
    
    # Create random images and labels
    images_t = torch.randn(num_samples, 3, resolution[0], resolution[1])
    images_t1 = torch.randn(num_samples, 3, resolution[0], resolution[1])
    
    # Create labels (probabilities and poses)
    probabilities = torch.rand(num_samples, 1)
    poses = torch.randn(num_samples, 6)
    
    # Create dataset
    class DummyDataset:
        def __init__(self, images_t, images_t1, probabilities, poses):
            self.images_t = images_t
            self.images_t1 = images_t1
            self.probabilities = probabilities
            self.poses = poses
        
        def __len__(self):
            return len(self.images_t)
        
        def __getitem__(self, idx):
            labels = {
                'misalignment_probability': self.probabilities[idx],
                'pose': self.poses[idx]
            }
            return self.images_t[idx], self.images_t1[idx], labels
    
    dataset = DummyDataset(images_t, images_t1, probabilities, poses)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def test_loss_function():
    """Test loss function implementation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Loss Function")
    logger.info("=" * 60)
    
    criterion = MisalignmentLoss(
        classification_weight=0.6,
        regression_weight=0.4
    )
    
    # Create dummy predictions and targets
    pred_prob = torch.rand(4, 1)
    pred_pose = torch.randn(4, 6)
    target_prob = torch.rand(4, 1)
    target_pose = torch.randn(4, 6)
    
    # Compute loss
    losses = criterion(pred_prob, pred_pose, target_prob, target_pose)
    
    logger.info(f"✓ Total loss: {losses['total'].item():.4f}")
    logger.info(f"✓ Classification loss: {losses['classification'].item():.4f}")
    logger.info(f"✓ Regression loss: {losses['regression'].item():.4f}")
    
    assert 'total' in losses
    assert 'classification' in losses
    assert 'regression' in losses
    
    logger.info("✓ Loss function test passed")
    return True


def test_checkpoint_manager():
    """Test checkpoint save/load functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Checkpoint Manager")
    logger.info("=" * 60)
    
    # Create temporary checkpoint directory
    checkpoint_dir = Path('test_checkpoints_temp')
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        max_checkpoints=3,
        save_interval=10
    )
    
    # Create dummy models
    feature_extractor = CNNFeatureExtractor()
    flow_network = LiteFlowNet2()
    pose_estimator = PoseEstimator()
    
    # Create optimizer
    all_params = list(feature_extractor.parameters()) + \
                 list(flow_network.parameters()) + \
                 list(pose_estimator.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)
    
    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(
        feature_extractor=feature_extractor,
        flow_network=flow_network,
        pose_estimator=pose_estimator,
        optimizer=optimizer,
        scheduler=None,
        training_step=10,
        epoch=0,
        validation_loss=0.5,
        validation_accuracy=0.9,
        training_loss_history=[0.7, 0.6, 0.55, 0.5],
        model_config={'test': True},
        is_best=False
    )
    
    logger.info(f"✓ Saved checkpoint: {checkpoint_path}")
    assert Path(checkpoint_path).exists()
    
    # Load checkpoint
    new_feature_extractor = CNNFeatureExtractor()
    new_flow_network = LiteFlowNet2()
    new_pose_estimator = PoseEstimator()
    
    checkpoint_data = manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        feature_extractor=new_feature_extractor,
        flow_network=new_flow_network,
        pose_estimator=new_pose_estimator
    )
    
    logger.info(f"✓ Loaded checkpoint: step={checkpoint_data['training_step']}")
    assert checkpoint_data['training_step'] == 10
    
    # Clean up
    shutil.rmtree(checkpoint_dir)
    logger.info("✓ Checkpoint manager test passed")
    
    return True


def test_training_loop_architecture_a():
    """Test training loop with Architecture A (LiteFlowNet2)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Training Loop - Architecture A (LiteFlowNet2)")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create dummy data loaders
    train_loader = create_dummy_data_loader(batch_size=2, num_samples=10, resolution=(320, 320))
    val_loader = create_dummy_data_loader(batch_size=2, num_samples=6, resolution=(320, 320))
    
    # Create models
    feature_extractor = CNNFeatureExtractor()
    flow_network = LiteFlowNet2()
    pose_estimator = PoseEstimator()
    
    # Create temporary directories
    checkpoint_dir = Path('test_checkpoints_a_temp')
    tensorboard_dir = Path('test_runs_a_temp')
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
    
    # Configuration
    config = {
        'architecture': 'Architecture_A_Test',
        'target_resolution': (320, 320),
        'batch_size': 2,
        'learning_rate': 1e-4,
        'classification_weight': 0.6,
        'regression_weight': 0.4,
        'mixed_precision': False,  # Disable for testing on CPU
        'checkpoint_dir': str(checkpoint_dir),
        'tensorboard_dir': str(tensorboard_dir),
        'run_name': 'test_architecture_a',
        'validation_interval': 5,  # Validate every 5 steps
        'early_stopping_patience': 10
    }
    
    # Create trainer
    trainer = Trainer(
        feature_extractor=feature_extractor,
        flow_network=flow_network,
        pose_estimator=pose_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Run a few training steps
    logger.info("Running 5 training steps...")
    
    # Manually run a few steps instead of full training
    for step in range(5):
        for batch in train_loader:
            images_t, images_t1, labels = batch
            losses = trainer.train_step(images_t, images_t1, labels)
            trainer.training_step += 1
            
            logger.info(
                f"  Step {trainer.training_step}: "
                f"Loss={losses['total']:.4f}, "
                f"Cls={losses['classification']:.4f}, "
                f"Reg={losses['regression']:.4f}"
            )
            
            if trainer.training_step >= 5:
                break
        if trainer.training_step >= 5:
            break
    
    logger.info("✓ Training steps completed")
    
    # Test validation
    logger.info("Running validation...")
    val_metrics = trainer.validate()
    logger.info(f"✓ Validation completed: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
    
    # Test checkpoint saving
    logger.info("Testing checkpoint save...")
    trainer.checkpoint_manager.save_checkpoint(
        feature_extractor=feature_extractor,
        flow_network=flow_network,
        pose_estimator=pose_estimator,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        training_step=trainer.training_step,
        epoch=0,
        validation_loss=val_metrics['loss'],
        validation_accuracy=val_metrics['accuracy'],
        training_loss_history=trainer.training_loss_history,
        model_config=config,
        is_best=True
    )
    
    # Verify checkpoint exists
    best_checkpoint = checkpoint_dir / 'best_model.pth'
    assert best_checkpoint.exists(), "Best model checkpoint not found"
    logger.info(f"✓ Best model saved: {best_checkpoint}")
    
    # Verify TensorBoard directory exists
    assert tensorboard_dir.exists(), "TensorBoard directory not created"
    logger.info(f"✓ TensorBoard logs created: {tensorboard_dir}")
    
    # Clean up
    trainer.writer.close()
    shutil.rmtree(checkpoint_dir)
    shutil.rmtree(tensorboard_dir)
    
    logger.info("✓ Architecture A training loop test passed")
    return True


def test_training_loop_architecture_b():
    """Test training loop with Architecture B (SpyNet)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Training Loop - Architecture B (SpyNet)")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create dummy data loaders
    train_loader = create_dummy_data_loader(batch_size=2, num_samples=10, resolution=(320, 320))
    val_loader = create_dummy_data_loader(batch_size=2, num_samples=6, resolution=(320, 320))
    
    # Create models (use SpyNet instead of LiteFlowNet2)
    feature_extractor = CNNFeatureExtractor()
    flow_network = SpyNet()  # <-- Different from Architecture A
    pose_estimator = PoseEstimator()
    
    # Create temporary directories
    checkpoint_dir = Path('test_checkpoints_b_temp')
    tensorboard_dir = Path('test_runs_b_temp')
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
    
    # Configuration
    config = {
        'architecture': 'Architecture_B_Test',
        'target_resolution': (320, 320),
        'batch_size': 2,
        'learning_rate': 1e-4,
        'classification_weight': 0.6,
        'regression_weight': 0.4,
        'mixed_precision': False,  # Disable for testing on CPU
        'checkpoint_dir': str(checkpoint_dir),
        'tensorboard_dir': str(tensorboard_dir),
        'run_name': 'test_architecture_b',
        'validation_interval': 5,
        'early_stopping_patience': 10
    }
    
    # Create trainer
    trainer = Trainer(
        feature_extractor=feature_extractor,
        flow_network=flow_network,
        pose_estimator=pose_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Run a few training steps
    logger.info("Running 5 training steps...")
    
    for step in range(5):
        for batch in train_loader:
            images_t, images_t1, labels = batch
            losses = trainer.train_step(images_t, images_t1, labels)
            trainer.training_step += 1
            
            logger.info(
                f"  Step {trainer.training_step}: "
                f"Loss={losses['total']:.4f}, "
                f"Cls={losses['classification']:.4f}, "
                f"Reg={losses['regression']:.4f}"
            )
            
            if trainer.training_step >= 5:
                break
        if trainer.training_step >= 5:
            break
    
    logger.info("✓ Training steps completed")
    
    # Test validation
    logger.info("Running validation...")
    val_metrics = trainer.validate()
    logger.info(f"✓ Validation completed: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
    
    # Clean up
    trainer.writer.close()
    shutil.rmtree(checkpoint_dir)
    shutil.rmtree(tensorboard_dir)
    
    logger.info("✓ Architecture B training loop test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("Training Pipeline Verification")
    print("=" * 80)
    print()
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Install with: pip install torch")
        sys.exit(1)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tests = [
        ("Loss Function", test_loss_function),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Architecture A Training", test_training_loop_architecture_a),
        ("Architecture B Training", test_training_loop_architecture_b),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"\n❌ {test_name} failed: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print()
        print("✓ All tests passed! Training pipeline is ready.")
        print()
        print("Next steps:")
        print("  1. Download KITTI dataset (see docs/INSTALLATION.md)")
        print("  2. Train Architecture A: python scripts/train_architecture_a.py")
        print("  3. Train Architecture B: python scripts/train_architecture_b.py")
        print("  4. Monitor with TensorBoard: tensorboard --logdir runs")
        return 0
    else:
        print()
        print("❌ Some tests failed. Please fix errors before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
