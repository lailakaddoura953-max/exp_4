"""
Flask backend for the Yard Hazard Inference Dashboard.

Exposes the following REST endpoints:
  GET  /                     — serve dashboard SPA (index.html)
  POST /api/inference        — run hazard inference on an uploaded image
  GET  /api/hazards/recent   — return the 3 most recent hazard events
  GET  /api/status           — return system health and model status
  GET  /api/test-image       — serve a test image from the CameraStub
                               (tries roboflow data/test/images/ first,
                                falls back to roboflow data/train/images/)

Module-level singletons (inference_engine, hazard_store, camera_stub) are
initialised at import time.  If the InferenceEngine fails to load (e.g. bad
checkpoint path), inference_engine is set to None and model_loaded is False;
the Flask app still starts and all other endpoints remain functional.

Requirements covered:
- 10.1: POST /api/inference, GET /api/hazards/recent, GET /api/status
- 10.2: decode image, delegate to InferenceEngine, return results + annotated_image
- 10.3: HTTP 400 when image field absent
- 10.4: append HazardEvent to HazardStore for each is_hazard=True result
- 10.5: GET /api/hazards/recent returns newest-first, [] when empty
- 10.6: GET /api/status returns status/model_loaded/hazard_count/camera_id
- 10.7: CORS enabled for all routes
- 10.8: HTTP 500 + log traceback on unhandled engine exception
- 11.1: annotate() called; red/green boxes per is_hazard
- 11.3: annotated_image returned as base64 PNG (or null)
- 11.4: annotation failure sets annotated_image=None in response
- 15.1: GET /api/test-image returns real image/jpeg
- 15.2: 404 when no test image configured or readable
- 15.3: image from roboflow data/test/images/ with fallback to train/images/
- 17.2: port, hazard_store_capacity, camera_stub_id configurable via env
- 17.5: startup log excludes sensitive checkpoint paths
"""

from __future__ import annotations

import io
import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from dashboard.annotator import annotate
from dashboard.camera_stub import CameraStub
from dashboard.hazard_store import HazardStore
from dashboard.inference_engine import InferenceEngine
from dashboard.models import (
    HazardEvent,
    InferenceEngineConfig,
    LocationContext,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

# Static files live in src/dashboard/static/
_STATIC_DIR = Path(__file__).parent / "static"

app = Flask(
    __name__,
    static_folder=str(_STATIC_DIR),
    static_url_path="/static",
)

# Enable CORS for all routes (Requirement 10.7)
CORS(app)


@app.route("/")
def index():
    """Serve the dashboard SPA entry point."""
    return send_file(_STATIC_DIR / "index.html")

# ---------------------------------------------------------------------------
# Configuration — read from environment variables with safe defaults
# ---------------------------------------------------------------------------

_PORT: int = int(os.environ.get("DASHBOARD_PORT", "5000"))
_DEVICE: str = os.environ.get("DASHBOARD_DEVICE", "cpu")
_CONF_THRESHOLD: float = float(os.environ.get("DASHBOARD_CONF_THRESHOLD", "0.5"))
_STORE_CAPACITY: int = int(os.environ.get("DASHBOARD_STORE_CAPACITY", "20"))
_CAMERA_STUB_ID: str = os.environ.get("DASHBOARD_CAMERA_STUB_ID", "cam_stub_01")

# Paths resolved relative to the workspace root (3 levels up from this file:
#   src/dashboard/app.py → src/dashboard → src → workspace root)
_WORKSPACE_ROOT: Path = Path(__file__).parent.parent.parent
_CHECKPOINT_PATH: Path = _WORKSPACE_ROOT / "checkpoints" / "yolov12_best.pt"
_TEST_IMAGE_DIR: Path = _WORKSPACE_ROOT / "roboflow data" / "test" / "images"
_TRAIN_IMAGE_DIR: Path = _WORKSPACE_ROOT / "roboflow data" / "train" / "images"

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

# HazardStore — always available
hazard_store = HazardStore(capacity=_STORE_CAPACITY)

# CameraStub — always available; test image path may not exist (returns None)
camera_stub = CameraStub(
    camera_id=_CAMERA_STUB_ID,
    test_image_path=str(_TEST_IMAGE_DIR),
)

# InferenceEngine — may fail if checkpoint is missing; set model_loaded flag
model_loaded: bool = False
inference_engine: InferenceEngine | None = None

try:
    _config = InferenceEngineConfig(
        checkpoint_path=str(_CHECKPOINT_PATH),
        device=_DEVICE,
        confidence_threshold=_CONF_THRESHOLD,
    )
    inference_engine = InferenceEngine(_config)
    model_loaded = True
except Exception as _exc:  # noqa: BLE001
    logger.error(
        "Failed to initialise InferenceEngine — model_loaded=False. "
        "Inference endpoint will return HTTP 500 until the engine is available. "
        "Error: %s: %s",
        type(_exc).__name__,
        _exc,
    )

# ---------------------------------------------------------------------------
# Startup log — no sensitive file paths (Requirement 17.5)
# ---------------------------------------------------------------------------

logger.info(
    "Dashboard startup — port=%d, device=%r, confidence_threshold=%s, "
    "hazard_store_capacity=%d, camera_stub_id=%r, model_loaded=%s",
    _PORT,
    _DEVICE,
    _CONF_THRESHOLD,
    _STORE_CAPACITY,
    _CAMERA_STUB_ID,
    model_loaded,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/inference", methods=["POST"])
def api_inference():
    """
    POST /api/inference

    Accepts multipart/form-data with:
      image     (required) — JPEG or PNG file
      camera_id (optional) — defaults to "cam_stub_01"

    Returns JSON:
      {"results": [...], "annotated_image": "<base64 PNG or null>"}

    Requirements: 10.2, 10.3, 10.4, 10.8, 11.1, 11.3, 11.4
    """
    # --- Guard: model not loaded ----------------------------------------
    if inference_engine is None:
        return jsonify({"error": "Model not loaded"}), 500

    # --- Guard: image field absent (Requirement 10.3) --------------------
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    camera_id: str = request.form.get("camera_id", "cam_stub_01")

    # --- Decode image (Requirement 10.2) ---------------------------------
    try:
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        image = None

    if image is None:
        return jsonify({"error": "Invalid or unreadable image"}), 400

    # --- Run inference (Requirement 10.2) --------------------------------
    try:
        results = inference_engine.run(image, camera_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unhandled exception during InferenceEngine.run — camera_id=%r:\n%s",
            camera_id,
            traceback.format_exc(),
        )
        return jsonify({"error": str(exc)}), 500

    # --- Annotate (Requirements 11.1, 11.3, 11.4) ------------------------
    try:
        annotated_image_b64: str | None = annotate(image, results)
    except Exception:  # noqa: BLE001
        logger.warning(
            "annotate() raised an exception — setting annotated_image=None:\n%s",
            traceback.format_exc(),
        )
        annotated_image_b64 = None

    # --- Store hazard events (Requirement 10.4) --------------------------
    timestamp_utc: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for result in results:
        if result.is_hazard:
            event = HazardEvent(
                event_id=str(uuid.uuid4()),
                hazard_type=result.hazard_reason,
                camera_id=camera_id,
                timestamp=timestamp_utc,
                confidence=result.confidence,
                bbox=result.bbox,
                annotated_image=annotated_image_b64,
                location=LocationContext.from_camera_id(camera_id),
            )
            hazard_store.append(event)

    # --- Return response (Requirement 10.2) ------------------------------
    return jsonify(
        {
            "results": [r.to_dict() for r in results],
            "annotated_image": annotated_image_b64,
        }
    )


@app.route("/api/live/images", methods=["GET"])
def api_live_images():
    """
    GET /api/live/images

    Returns a list of inference results from dataset images.

    Priority (mirrors exp_2's pattern):
      1. Hazard store — if events already recorded from real uploads, return those.
      2. Dataset fallback — run inference on random images from roboflow data/
         and return annotated results so the UI is never empty on first load.

    Query params:
        limit   (int, default 6)   — max number of results to return
        source  (str, default auto) — 'live' (store only), 'dataset' (force
                                       dataset), or 'auto' (store first, then
                                       dataset fallback)

    Returns JSON: { success, data: [...], count, source }
    Each item: { event_id, hazard_type, is_hazard, camera_id, timestamp,
                 confidence, bbox, annotated_image, location }
    """
    import random
    import glob

    limit  = request.args.get("limit",  6,      type=int)
    source = request.args.get("source", "auto")

    results: list[dict] = []

    # --- Priority 1: existing hazard store events ---
    if source in ("live", "auto"):
        stored = hazard_store.get_recent(limit)
        results = [e.to_dict() for e in stored]

    # --- Priority 2: dataset fallback ---
    if source == "dataset" or (source == "auto" and len(results) < limit):
        needed = limit - len(results)

        # Collect candidate image paths
        candidates: list[str] = []
        for d in [str(_TEST_IMAGE_DIR), str(_TRAIN_IMAGE_DIR)]:
            candidates.extend(glob.glob(f"{d}/*.jpg"))
            candidates.extend(glob.glob(f"{d}/*.jpeg"))
            candidates.extend(glob.glob(f"{d}/*.png"))

        if candidates and inference_engine is not None:
            sample = random.sample(candidates, min(needed * 3, len(candidates)))
            dataset_items: list[dict] = []

            for path in sample:
                if len(dataset_items) >= needed:
                    break

                image = cv2.imread(path)
                if image is None:
                    continue

                cam_id = _CAMERA_STUB_ID
                try:
                    detections = inference_engine.run(image, cam_id)
                except Exception as exc:
                    logger.warning("api_live_images: inference failed for %s: %s", path, exc)
                    continue

                try:
                    annotated_b64 = annotate(image, detections)
                except Exception:
                    annotated_b64 = None

                timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                # Pick the most significant detection for the card:
                #   hazard detections first, else the highest-confidence result
                hazards = [r for r in detections if r.is_hazard]
                representative = hazards[0] if hazards else (detections[0] if detections else None)

                if representative is None:
                    # No detections at all — still show the annotated image
                    from hazard_detection.models import BBox
                    item: dict = {
                        "event_id":       str(uuid.uuid4()),
                        "hazard_type":    "no_hazard",
                        "is_hazard":      False,
                        "camera_id":      cam_id,
                        "timestamp":      timestamp_utc,
                        "confidence":     0.0,
                        "bbox":           {"x_center": 0.5, "y_center": 0.5,
                                           "width": 0.0, "height": 0.0},
                        "annotated_image": annotated_b64,
                        "location":       LocationContext.from_camera_id(cam_id).to_dict(),
                        "source":         "dataset",
                    }
                else:
                    item = {
                        "event_id":       str(uuid.uuid4()),
                        "hazard_type":    representative.hazard_reason or "no_hazard",
                        "is_hazard":      representative.is_hazard,
                        "camera_id":      cam_id,
                        "timestamp":      timestamp_utc,
                        "confidence":     representative.confidence,
                        "bbox":           {
                            "x_center": representative.bbox.x_center,
                            "y_center": representative.bbox.y_center,
                            "width":    representative.bbox.width,
                            "height":   representative.bbox.height,
                        },
                        "annotated_image": annotated_b64,
                        "location":       LocationContext.from_camera_id(cam_id).to_dict(),
                        "source":         "dataset",
                    }

                dataset_items.append(item)

            results.extend(dataset_items)

    actual_source = "live"
    if results and all(r.get("source") == "dataset" for r in results):
        actual_source = "dataset"
    elif not results:
        actual_source = "none"

    return jsonify({
        "success": True,
        "data":    results[:limit],
        "count":   len(results[:limit]),
        "source":  actual_source,
    })



def api_hazards_recent():
    """
    GET /api/hazards/recent

    Returns the 3 most recent hazard events, newest first.
    Returns [] when no events are stored.

    Requirement: 10.5
    """
    recent = hazard_store.get_recent(3)
    return jsonify([e.to_dict() for e in recent])


@app.route("/api/status", methods=["GET"])
def api_status():
    """
    GET /api/status

    Returns system health information.

    Requirement: 10.6
    """
    return jsonify(
        {
            "status": "running",
            "model_loaded": model_loaded,
            "hazard_count": hazard_store.count(),
            "camera_id": camera_stub.get_camera_id(),
        }
    )


@app.route("/api/test-image", methods=["GET"])
def api_test_image():
    """
    GET /api/test-image

    Returns a randomly selected test image from the CameraStub as JPEG bytes.
    Tries ``roboflow data/test/images/`` first; falls back to
    ``roboflow data/train/images/`` if the test directory yields no image.
    Returns HTTP 404 when neither directory yields an image.

    Requirements: 15.1, 15.2, 15.3
    """
    image = camera_stub.get_test_image()

    # Fallback: if test/images/ returned nothing, try train/images/
    if image is None:
        logger.info(
            "api_test_image: primary test image path returned None; "
            "falling back to train/images/"
        )
        fallback_stub = CameraStub(
            camera_id=camera_stub.get_camera_id(),
            test_image_path=str(_TRAIN_IMAGE_DIR),
        )
        image = fallback_stub.get_test_image()

    if image is None:
        return jsonify({"error": "No dataset images found"}), 404

    # Encode as JPEG bytes and serve with the correct MIME type
    success, buffer = cv2.imencode(".jpg", image)
    if not success or buffer is None:
        logger.error("api_test_image: cv2.imencode failed")
        return jsonify({"error": "No dataset images found"}), 404

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/jpeg",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=_PORT, debug=True)
