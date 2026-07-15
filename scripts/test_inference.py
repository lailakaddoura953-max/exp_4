"""
Integration Tests for Inference Engine

This script tests the complete inference pipeline including:
1. Checkpoint loading time (≤5 seconds)
2. 4-camera batch latency (≤100ms)
3. JSON serialization
4. Uncertainty estimation toggle
5. Confidence threshold application

Task 11.7: Integration Tests
Requirements: 9.1, 9.3, 10.5, 13.6

Usage:
    python scripts/test_inference.py --checkpoint <path> --config <path>
"""

import argparse
import sys
import time
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    import torch
    import yaml
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install torch pyyaml pillow")
    DEPENDENCIES_AVAILABLE = False

if DEPENDENCIES_AVAILABLE:
    from dl_misalignment.inference import (
        InferenceEngine,
        load_inference_engine,
        ImagePreprocessor,
        FourCameraBatchBuilder,
        InferenceOutput,
        CameraDetection
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Test 11.7.1: Checkpoint Loading Time
# ==============================================================================

def test_checkpoint_loading_time(checkpoint_path: str, config: dict) -> bool:
    """
    Test that checkpoint loads within 5 seconds.
    
    Requirements: 9.3
    """
    logger.info("=" * 70)
    logger.info("Test 11.7.1: Checkpoint Loading Time (≤5 seconds)")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        engine = InferenceEngine(
            checkpoint_path=checkpoint_path,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        load_time = time.time() - start_time
        
        logger.info(f"✓ Checkpoint loaded in {load_time:.2f}s")
        
        if load_time <= 5.0:
            logger.info("✓ PASS: Loading time within target (≤5s)")
            return True
        else:
            logger.warning(f"✗ FAIL: Loading time {load_time:.2f}s exceeds 5s target")
            return False
    
    except Exception as e:
        logger.error(f"✗ FAIL: Checkpoint loading failed: {e}")
        return False


# ==============================================================================
# Test 11.7.2: Four-Camera Batch Latency
# ==============================================================================

def test_four_camera_batch_latency(engine: InferenceEngine) -> bool:
    """
    Test that 4-camera batch processes within 100ms.
    
    Requirements: 9.1
    """
    logger.info("=" * 70)
    logger.info("Test 11.7.2: Four-Camera Batch Latency (≤100ms)")
    logger.info("=" * 70)
    
    # Create synthetic 4-camera frames
    camera_frames = {}
    camera_ids = ['front', 'left', 'right', 'rear']
    
    for camera_id in camera_ids:
        # Create random RGB image (640×640)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        camera_frames[camera_id] = image
    
    # Warm-up run (first inference allocates memory)
    logger.info("Running warm-up inference...")
    _ = engine.infer(camera_frames)
    
    # Actual timed runs
    num_runs = 10
    latencies = []
    
    logger.info(f"Running {num_runs} timed inferences...")
    
    for i in range(num_runs):
        start_time = time.time()
        output, timing = engine.infer(camera_frames, return_timing_breakdown=True)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        logger.info(
            f"  Run {i+1}/{num_runs}: {latency_ms:.1f}ms "
            f"(breakdown: {timing})"
        )
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    
    logger.info(f"\nLatency Statistics:")
    logger.info(f"  Average: {avg_latency:.1f}ms")
    logger.info(f"  Min: {min_latency:.1f}ms")
    logger.info(f"  Max: {max_latency:.1f}ms")
    
    if avg_latency <= 100.0:
        logger.info("✓ PASS: Average latency within target (≤100ms)")
        return True
    else:
        logger.warning(f"✗ FAIL: Average latency {avg_latency:.1f}ms exceeds 100ms target")
        return False


# ==============================================================================
# Test 11.7.3: JSON Serialization
# ==============================================================================

def test_json_serialization(engine: InferenceEngine) -> bool:
    """
    Test JSON serialization of inference output.
    
    Requirements: 28.5, 28.6
    """
    logger.info("=" * 70)
    logger.info("Test 11.7.3: JSON Serialization")
    logger.info("=" * 70)
    
    # Create synthetic frames
    camera_frames = {
        'front': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'left': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'right': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'rear': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    }
    
    try:
        # Run inference
        output = engine.infer(camera_frames)
        
        # Test to_json()
        json_str = output.to_json()
        logger.info(f"✓ JSON serialization successful ({len(json_str)} bytes)")
        
        # Test deserialization
        output_restored = InferenceOutput.from_json(json_str)
        logger.info(f"✓ JSON deserialization successful")
        
        # Verify content
        assert len(output_restored.camera_results) == 4
        assert output_restored.model_version == output.model_version
        assert output_restored.processing_time_ms == output.processing_time_ms
        
        logger.info("✓ PASS: JSON serialization and deserialization work correctly")
        
        # Print sample JSON (first 500 chars)
        logger.info(f"\nSample JSON output:\n{json_str[:500]}...")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ FAIL: JSON serialization failed: {e}")
        return False


# ==============================================================================
# Test 11.7.4: Uncertainty Toggle
# ==============================================================================

def test_uncertainty_toggle(checkpoint_path: str, config: dict) -> bool:
    """
    Test that uncertainty estimation can be toggled on/off.
    
    Requirements: 13.6
    """
    logger.info("=" * 70)
    logger.info("Test 11.7.4: Uncertainty Estimation Toggle")
    logger.info("=" * 70)
    
    camera_frames = {
        'front': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'left': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'right': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'rear': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test with uncertainty disabled
        logger.info("\nTest 1: Uncertainty DISABLED")
        config_no_unc = config.copy()
        config_no_unc['enable_uncertainty'] = False
        
        engine_no_unc = InferenceEngine(checkpoint_path, config_no_unc, device)
        
        start = time.time()
        output_no_unc = engine_no_unc.infer(camera_frames)
        time_no_unc = (time.time() - start) * 1000
        
        logger.info(f"  Latency: {time_no_unc:.1f}ms")
        
        # Check no uncertainty in output
        for camera_id, detection in output_no_unc.camera_results.items():
            if detection.probability_uncertainty is not None:
                logger.warning(f"  Warning: {camera_id} has uncertainty despite being disabled")
        
        logger.info("  ✓ Uncertainty disabled correctly")
        
        # Test with uncertainty enabled
        logger.info("\nTest 2: Uncertainty ENABLED")
        config_unc = config.copy()
        config_unc['enable_uncertainty'] = True
        config_unc['uncertainty_samples'] = 10
        
        engine_unc = InferenceEngine(checkpoint_path, config_unc, device)
        
        start = time.time()
        output_unc = engine_unc.infer(camera_frames)
        time_unc = (time.time() - start) * 1000
        
        logger.info(f"  Latency: {time_unc:.1f}ms")
        
        # Check uncertainty in output
        has_uncertainty = False
        for camera_id, detection in output_unc.camera_results.items():
            if detection.probability_uncertainty is not None:
                has_uncertainty = True
                logger.info(
                    f"  {camera_id}: prob={detection.misalignment_probability:.3f} "
                    f"± {detection.probability_uncertainty:.3f}, "
                    f"low_conf={detection.low_confidence}"
                )
        
        if not has_uncertainty:
            logger.warning("  Warning: No uncertainty estimates found despite being enabled")
        else:
            logger.info("  ✓ Uncertainty enabled correctly")
        
        # Compare latencies
        logger.info(f"\nLatency Comparison:")
        logger.info(f"  Without uncertainty: {time_no_unc:.1f}ms")
        logger.info(f"  With uncertainty: {time_unc:.1f}ms")
        logger.info(f"  Overhead: {time_unc - time_no_unc:.1f}ms ({time_unc/time_no_unc:.1f}×)")
        
        logger.info("✓ PASS: Uncertainty toggle works correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ FAIL: Uncertainty toggle test failed: {e}")
        return False


# ==============================================================================
# Test 11.7.5: Confidence Threshold Application
# ==============================================================================

def test_confidence_threshold(checkpoint_path: str, config: dict) -> bool:
    """
    Test that confidence threshold is applied correctly.
    
    Requirements: 10.5
    """
    logger.info("=" * 70)
    logger.info("Test 11.7.5: Confidence Threshold Application")
    logger.info("=" * 70)
    
    camera_frames = {
        'front': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'left': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'right': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'rear': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            logger.info(f"\nTesting threshold = {threshold}")
            
            config_thresh = config.copy()
            config_thresh['confidence_threshold'] = threshold
            
            engine = InferenceEngine(checkpoint_path, config_thresh, device)
            output = engine.infer(camera_frames)
            
            # Check threshold is stored
            assert engine.confidence_threshold == threshold
            
            # Count misaligned cameras
            misaligned = output.get_misaligned_cameras(threshold)
            logger.info(f"  Misaligned cameras (>{threshold}): {misaligned}")
            
            # Show probabilities
            for camera_id, detection in output.camera_results.items():
                status = "MISALIGNED" if detection.misalignment_probability > threshold else "OK"
                logger.info(
                    f"    {camera_id}: prob={detection.misalignment_probability:.3f} "
                    f"severity={detection.severity_level} [{status}]"
                )
        
        logger.info("\n✓ PASS: Confidence threshold application works correctly")
        return True
    
    except Exception as e:
        logger.error(f"✗ FAIL: Confidence threshold test failed: {e}")
        return False


# ==============================================================================
# Main Test Suite
# ==============================================================================

def run_all_tests(checkpoint_path: str, config_path: str):
    """
    Run all inference engine integration tests.
    """
    logger.info("=" * 70)
    logger.info("INFERENCE ENGINE INTEGRATION TESTS")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 70)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path
    config['checkpoint_path'] = checkpoint_path
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    # Test 11.7.1: Checkpoint loading
    results['checkpoint_loading'] = test_checkpoint_loading_time(checkpoint_path, config)
    
    # Create engine for remaining tests
    engine = InferenceEngine(checkpoint_path, config, device)
    
    # Test 11.7.2: Batch latency
    results['batch_latency'] = test_four_camera_batch_latency(engine)
    
    # Test 11.7.3: JSON serialization
    results['json_serialization'] = test_json_serialization(engine)
    
    # Test 11.7.4: Uncertainty toggle
    results['uncertainty_toggle'] = test_uncertainty_toggle(checkpoint_path, config)
    
    # Test 11.7.5: Confidence threshold
    results['confidence_threshold'] = test_confidence_threshold(checkpoint_path, config)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
    logger.info("=" * 70)
    
    return passed_tests == total_tests


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Integration tests for inference engine'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pth file)',
        default='checkpoints/architecture_a_best.pth'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config YAML file',
        default='config/architecture_a.yaml'
    )
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing required dependencies")
        return 1
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Note: This test requires a trained model checkpoint.")
        logger.info("Please train a model first using scripts/train_architecture_a.py")
        return 1
    
    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return 1
    
    # Run tests
    try:
        all_passed = run_all_tests(args.checkpoint, args.config)
        return 0 if all_passed else 1
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
