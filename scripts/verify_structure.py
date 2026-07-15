#!/usr/bin/env python3
"""
Script to verify the project structure is correctly set up
"""
import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ MISSING {description}: {filepath}")
        return False


def verify_project_structure():
    """Verify all required files and directories exist"""
    root = Path(__file__).parent.parent
    all_good = True
    
    print("Verifying Camera Misalignment Detection Project Structure\n")
    print("=" * 70)
    
    # CMake files
    print("\n[CMake Build System]")
    all_good &= check_file_exists(root / "CMakeLists.txt", "Root CMakeLists.txt")
    all_good &= check_file_exists(root / "cpp" / "CMakeLists.txt", "C++ CMakeLists.txt")
    all_good &= check_file_exists(root / "cpp" / "tests" / "CMakeLists.txt", "C++ tests CMakeLists.txt")
    all_good &= check_file_exists(root / "python_bindings" / "CMakeLists.txt", "Python bindings CMakeLists.txt")
    
    # C++ headers
    print("\n[C++ Headers]")
    headers = [
        "data_structures.h",
        "calibration_data.h",
        "feature_extraction_engine.h",
        "optical_flow_analyzer.h",
        "slam_position_tracker.h",
    ]
    for header in headers:
        all_good &= check_file_exists(root / "cpp" / "include" / header, f"Header: {header}")
    
    # C++ source files
    print("\n[C++ Source Files]")
    sources = [
        "calibration_data.cpp",
        "feature_extraction_engine.cpp",
        "optical_flow_analyzer.cpp",
        "slam_position_tracker.cpp",
    ]
    for source in sources:
        all_good &= check_file_exists(root / "cpp" / "src" / source, f"Source: {source}")
    
    # C++ test files
    print("\n[C++ Test Files]")
    tests = [
        "test_calibration_data.cpp",
        "test_feature_extraction.cpp",
        "test_optical_flow.cpp",
        "test_slam_tracker.cpp",
    ]
    for test in tests:
        all_good &= check_file_exists(root / "cpp" / "tests" / test, f"Test: {test}")
    
    # Python bindings
    print("\n[Python Bindings]")
    all_good &= check_file_exists(root / "python_bindings" / "bindings.cpp", "pybind11 bindings")
    
    # Python package
    print("\n[Python Package]")
    py_files = [
        "__init__.py",
        "data_models.py",
        "frame_acquisition.py",
        "misalignment_detector.py",
        "alert_system.py",
        "database_logger.py",
    ]
    for py_file in py_files:
        all_good &= check_file_exists(root / "src" / "camera_misalignment" / py_file, f"Python: {py_file}")
    
    # Setup files
    print("\n[Setup and Configuration]")
    all_good &= check_file_exists(root / "setup.py", "setup.py")
    all_good &= check_file_exists(root / "pytest.ini", "pytest.ini")
    all_good &= check_file_exists(root / "requirements_dev.txt", "requirements_dev.txt")
    all_good &= check_file_exists(root / ".gitignore", ".gitignore")
    
    # Documentation
    print("\n[Documentation]")
    all_good &= check_file_exists(root / "README.md", "README.md")
    all_good &= check_file_exists(root / "BUILD.md", "BUILD.md")
    
    # Configuration examples
    print("\n[Configuration Examples]")
    all_good &= check_file_exists(root / "config" / "system_config_example.yaml", "System config example")
    all_good &= check_file_exists(root / "config" / "calibration_example.json", "Calibration example")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ All required files are present!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements_dev.txt")
        print("2. Build C++ components: See BUILD.md")
        print("3. Install package: pip install -e .")
        return 0
    else:
        print("✗ Some files are missing. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_project_structure())
