"""
Example: Using the Inference Engine

This script demonstrates how to use the inference engine for
real-time camera misalignment detection.

Usage:
    python scripts/example_inference.py --config config/architecture_a.yaml
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from dl_misalignment.inference import load_inference_engine
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install torch pyyaml pillow")
    DEPENDENCIES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_camera_frames():
    """Create synthetic camera frames for demonstration."""
    camera_frames = {}
    camera_ids = ['front', 'left', 'right', 'rear']
    
    for camera_id in camera_ids:
        # Create random RGB image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        camera_frames[camera_id] = image
    
    return camera_frames


def main():
    """Main example."""
    parser = argparse.ArgumentParser(
        description='Example inference engine usage'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/architecture_a.yaml',
        help='Path to config YAML file'
    )
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing required dependencies")
        return 1
    
    logger.info("=" * 70)
    logger.info("INFERENCE ENGINE EXAMPLE")
    logger.info("=" * 70)
    
    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    try:
        # Load inference engine
        logger.info(f"Loading inference engine from {config_path}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        engine = load_inference_engine(str(config_path), device=device)
        logger.info(f"✓ Engine loaded: {engine}")
        
        # Create synthetic camera frames
        logger.info("\nCreating synthetic 4-camera frames...")
        camera_frames = create_synthetic_camera_frames()
        logger.info(f"✓ Created frames for cameras: {list(camera_frames.keys())}")
        
        # Run inference
        logger.info("\nRunning inference...")
        output, timing = engine.infer(camera_frames, return_timing_breakdown=True)
        
        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("INFERENCE RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"Processing time: {output.processing_time_ms:.1f}ms")
        logger.info(f"Model version: {output.model_version}")
        logger.info(f"Architecture: {output.architecture}")
        
        logger.info("\nTiming breakdown:")
        for key, value in timing.items():
            logger.info(f"  {key}: {value:.2f}")
        
        logger.info("\nPer-camera results:")
        for camera_id, detection in output.camera_results.items():
            logger.info(f"\n  {camera_id.upper()}:")
            logger.info(f"    Probability: {detection.misalignment_probability:.3f}")
            logger.info(f"    Severity: {detection.severity_level}")
            logger.info(f"    Position: X={detection.position['X']:.3f}m, "
                       f"Y={detection.position['Y']:.3f}m, "
                       f"Z={detection.position['Z']:.3f}m")
            logger.info(f"    Orientation: roll={detection.orientation['roll']:.2f}°, "
                       f"pitch={detection.orientation['pitch']:.2f}°, "
                       f"yaw={detection.orientation['yaw']:.2f}°")
            
            if detection.probability_uncertainty is not None:
                logger.info(f"    Uncertainty: ±{detection.probability_uncertainty:.3f}")
            
            if detection.low_confidence:
                logger.info(f"    ⚠ LOW CONFIDENCE")
        
        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        
        max_severity = output.get_max_severity()
        logger.info(f"Maximum severity: {max_severity.value}")
        
        misaligned = output.get_misaligned_cameras(threshold=0.5)
        if misaligned:
            logger.info(f"Misaligned cameras (>0.5): {', '.join(misaligned)}")
        else:
            logger.info("No misalignment detected (all cameras OK)")
        
        low_conf = output.get_low_confidence_cameras()
        if low_conf:
            logger.info(f"Low confidence cameras: {', '.join(low_conf)}")
        
        # JSON export
        logger.info("\n" + "=" * 70)
        logger.info("JSON EXPORT")
        logger.info("=" * 70)
        
        json_str = output.to_json()
        logger.info(f"JSON output ({len(json_str)} bytes):")
        logger.info(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        
        # Engine statistics
        logger.info("\n" + "=" * 70)
        logger.info("ENGINE STATISTICS")
        logger.info("=" * 70)
        
        stats = engine.get_statistics()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\n✓ Example completed successfully!")
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("\nNote: This example requires a trained model checkpoint.")
        logger.info("The checkpoint path is specified in the config YAML file.")
        logger.info("Please train a model first using scripts/train_architecture_a.py")
        return 1
    
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
