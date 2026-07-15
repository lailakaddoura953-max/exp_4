"""Step 1 - Verify installation."""
import sys
sys.path.insert(0, 'src')

print("Python:", sys.version.split()[0])

try:
    import torch
    print("PyTorch:          ", torch.__version__)
    print("CUDA available:   ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:              ", torch.cuda.get_device_name(0))
    else:
        print("GPU:               None (CPU mode)")
except ImportError:
    print("PyTorch:           NOT INSTALLED")

try:
    from ultralytics import YOLO
    print("ultralytics:       OK")
except ImportError:
    print("ultralytics:       NOT INSTALLED")

try:
    from hazard_detection.models import HazardEvent
    print("hazard_detection:  OK")
except ImportError as e:
    print("hazard_detection:  FAILED -", e)

try:
    import cv2
    print("opencv:            OK")
except ImportError:
    print("opencv:            NOT INSTALLED")
