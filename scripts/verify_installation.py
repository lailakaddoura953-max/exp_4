"""
Installation Verification Script

This script verifies that all dependencies are correctly installed and
the system meets hardware requirements for training and inference.

Run this after installing dependencies:
    python scripts/verify_installation.py

The script will check:
1. Python version compatibility
2. Core package installations (PyTorch, torchvision, etc.)
3. GPU availability and CUDA setup
4. Hardware requirements for training and inference
5. Optional development packages
"""

import sys
import logging
from importlib import import_module
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """
    Check if Python version meets minimum requirements (3.8+).
    
    Returns:
        bool: True if Python version is compatible
    """
    major, minor = sys.version_info[:2]
    
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f"Python {major}.{minor} detected. Python 3.8+ required.")
        logger.error("Please upgrade Python:")
        logger.error("  - Windows: Download from python.org")
        logger.error("  - Linux: sudo apt install python3.8 (or newer)")
        logger.error("  - macOS: brew install python@3.8 (or newer)")
        return False
    
    logger.info(f"✓ Python {major}.{minor} (compatible)")
    return True


def check_package_installation(package_name: str, display_name: str = None) -> Tuple[bool, str]:
    """
    Check if a Python package is installed and importable.
    
    Args:
        package_name: Import name of the package (e.g., 'cv2' for opencv-python)
        display_name: Display name for logging (e.g., 'OpenCV'). Defaults to package_name
    
    Returns:
        tuple: (is_installed, version_string)
    """
    display_name = display_name or package_name
    
    try:
        module = import_module(package_name)
        
        # Try to get version (different packages store version differently)
        version = "unknown"
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = str(getattr(module, attr))
                break
        
        return True, version
    
    except ImportError as e:
        return False, str(e)


def check_core_dependencies() -> List[Tuple[str, bool, str]]:
    """
    Check all core dependencies required for the system to run.
    
    Returns:
        list: List of tuples (package_name, is_installed, version_or_error)
    """
    # Core dependencies from requirements.txt
    # Format: (import_name, display_name)
    core_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('yaml', 'PyYAML'),
        ('tensorboard', 'TensorBoard'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('pydantic', 'Pydantic'),
    ]
    
    results = []
    
    logger.info("\nChecking core dependencies:")
    logger.info("-" * 80)
    
    for import_name, display_name in core_packages:
        is_installed, info = check_package_installation(import_name, display_name)
        results.append((display_name, is_installed, info))
        
        if is_installed:
            logger.info(f"✓ {display_name:15} version {info}")
        else:
            logger.error(f"✗ {display_name:15} NOT INSTALLED")
            logger.error(f"  Error: {info}")
    
    return results


def check_dev_dependencies() -> List[Tuple[str, bool, str]]:
    """
    Check optional development dependencies.
    
    Returns:
        list: List of tuples (package_name, is_installed, version_or_error)
    """
    # Development dependencies from requirements-dev.txt
    dev_packages = [
        ('pytest', 'pytest'),
        ('black', 'black'),
        ('flake8', 'flake8'),
        ('mypy', 'mypy'),
        ('sphinx', 'sphinx'),
    ]
    
    results = []
    
    logger.info("\nChecking development dependencies (optional):")
    logger.info("-" * 80)
    
    for import_name, display_name in dev_packages:
        is_installed, info = check_package_installation(import_name, display_name)
        results.append((display_name, is_installed, info))
        
        if is_installed:
            logger.info(f"✓ {display_name:15} version {info}")
        else:
            logger.warning(f"○ {display_name:15} not installed (optional)")
    
    return results


def check_cuda_setup() -> bool:
    """
    Check CUDA setup and GPU availability.
    
    Returns:
        bool: True if CUDA is available and working
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Cannot check CUDA.")
        return False
    
    logger.info("\nChecking CUDA and GPU setup:")
    logger.info("-" * 80)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        logger.warning("✗ CUDA is NOT available")
        logger.warning("  This system will run VERY SLOWLY on CPU")
        logger.warning("  Possible reasons:")
        logger.warning("    - No NVIDIA GPU detected")
        logger.warning("    - CUDA Toolkit not installed")
        logger.warning("    - PyTorch installed without CUDA support")
        logger.warning("    - Using AMD GPU (requires ROCm-enabled PyTorch)")
        logger.warning("")
        logger.warning("  To enable GPU acceleration:")
        logger.warning("    1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        logger.warning("    2. Reinstall PyTorch with CUDA: https://pytorch.org/get-started/locally/")
        return False
    
    # CUDA is available - get details
    logger.info(f"✓ CUDA is available")
    logger.info(f"  CUDA Version: {torch.version.cuda}")
    logger.info(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    # Show information for each GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"\n  GPU {i}: {props.name}")
        logger.info(f"    VRAM: {props.total_memory / (1024**3):.2f} GB")
        logger.info(f"    Compute Capability: {props.major}.{props.minor}")
        logger.info(f"    Multiprocessors: {props.multi_processor_count}")
    
    return True


def check_hardware_requirements() -> Tuple[bool, bool]:
    """
    Check if hardware meets requirements for training and inference.
    
    Returns:
        tuple: (training_valid, inference_valid)
    """
    try:
        from dl_misalignment.utils.hardware import (
            validate_training_hardware,
            validate_inference_hardware
        )
    except ImportError:
        logger.warning("Cannot import hardware validation module")
        logger.warning("Make sure package is installed: pip install -e .")
        return False, False
    
    logger.info("\nValidating hardware requirements:")
    logger.info("-" * 80)
    
    # Check training requirements
    training_valid, training_msg = validate_training_hardware()
    if training_valid:
        logger.info(f"✓ Training: {training_msg}")
    else:
        logger.warning(f"✗ Training: {training_msg}")
    
    # Check inference requirements
    inference_valid, inference_msg = validate_inference_hardware()
    if inference_valid:
        logger.info(f"✓ Inference: {inference_msg}")
    else:
        logger.warning(f"✗ Inference: {inference_msg}")
    
    return training_valid, inference_valid


def main():
    """
    Main verification routine.
    """
    print("\n" + "=" * 80)
    print("Deep Learning Misalignment Detection System")
    print("Installation Verification")
    print("=" * 80)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
        print("\n❌ Installation verification FAILED: Python version too old")
        sys.exit(1)
    
    # Check core dependencies
    core_results = check_core_dependencies()
    missing_core = [name for name, installed, _ in core_results if not installed]
    
    if missing_core:
        all_checks_passed = False
        logger.error(f"\n❌ Missing core dependencies: {', '.join(missing_core)}")
        logger.error("Install them with: pip install -r requirements.txt")
    
    # Check development dependencies (optional)
    dev_results = check_dev_dependencies()
    missing_dev = [name for name, installed, _ in dev_results if not installed]
    
    if missing_dev:
        logger.info(f"\n○ Optional development packages not installed: {', '.join(missing_dev)}")
        logger.info("  Install them with: pip install -r requirements-dev.txt")
    
    # Check CUDA (if PyTorch is installed)
    if not missing_core or 'PyTorch' not in missing_core:
        cuda_ok = check_cuda_setup()
        if not cuda_ok:
            logger.warning("\n⚠️  CUDA not available - system will be VERY SLOW")
            logger.warning("   Training and inference strongly recommended on GPU")
    
    # Check hardware requirements (if everything else is installed)
    if not missing_core:
        training_ok, inference_ok = check_hardware_requirements()
        
        if not training_ok:
            logger.warning("\n⚠️  Hardware does not meet training requirements")
            logger.warning("   Training may fail or be extremely slow")
        
        if not inference_ok:
            logger.warning("\n⚠️  Hardware does not meet inference requirements")
            logger.warning("   Inference may fail or be slow")
    
    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed and not missing_core:
        print("✅ Installation verification PASSED")
        print("   All core dependencies are installed correctly")
        print("   System is ready for use!")
    elif missing_core:
        print("❌ Installation verification FAILED")
        print(f"   Missing dependencies: {', '.join(missing_core)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("⚠️  Installation verification completed with warnings")
        print("   System may work but performance could be impacted")
    print("=" * 80 + "\n")
    
    # Return appropriate exit code
    sys.exit(0 if all_checks_passed and not missing_core else 1)


if __name__ == "__main__":
    main()
