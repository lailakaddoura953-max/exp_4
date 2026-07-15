"""
src/dashboard
=============
Yard Hazard Inference Dashboard package.

This package provides:
- Inference engine for single-frame YOLO-based hazard classification
- Rule engine implementing the 17-class Roboflow hazard rule set
- Flask backend with REST endpoints
- Dashboard UI (static assets in `static/`)
- Camera stub for camera-agnostic integration
- HazardStore for in-memory recent event tracking
"""
