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

# CheckpointResolver — auto-discovers best.pt with config override (Task 9.1)
from dashboard.checkpoint_resolver import CheckpointResolver

# Read config checkpoint_path from hazard_detection.yaml if available
_config_checkpoint_path: str | None = None
try:
    import yaml as _yaml
    _hd_config_path = _WORKSPACE_ROOT / "config" / "hazard_detection.yaml"
    if _hd_config_path.is_file():
        with open(_hd_config_path, "r", encoding="utf-8") as _f:
            _hd_config = _yaml.safe_load(_f.read())
        _config_checkpoint_path = (_hd_config or {}).get("yolo", {}).get("checkpoint_path")
except Exception:
    pass

_checkpoint_resolver = CheckpointResolver(
    config_path=_config_checkpoint_path,
    discovery_pattern=str(_WORKSPACE_ROOT / "runs" / "train" / "*" / "weights" / "best.pt"),
)
_resolved_checkpoint = _checkpoint_resolver.resolve()

# If auto-discovery/config both failed, try workspace-root .pt files as last resort
if _resolved_checkpoint is None:
    _fallback_pts = sorted(_WORKSPACE_ROOT.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if _fallback_pts:
        _resolved_checkpoint = _fallback_pts[0]
        logger.info("CheckpointResolver: using workspace-root fallback: %s", _resolved_checkpoint)

# InferenceEngine — uses resolved checkpoint
model_loaded: bool = False
inference_engine: InferenceEngine | None = None

if _resolved_checkpoint is not None:
    try:
        _config = InferenceEngineConfig(
            checkpoint_path=str(_resolved_checkpoint),
            device=_DEVICE,
            confidence_threshold=_CONF_THRESHOLD,
        )
        inference_engine = InferenceEngine(_config)
        model_loaded = True
    except Exception as _exc:  # noqa: BLE001
        logger.error(
            "Failed to initialise InferenceEngine — model_loaded=False. "
            "Error: %s: %s",
            type(_exc).__name__,
            _exc,
        )
else:
    logger.warning(
        "No YOLO checkpoint found (config, auto-discovery, or workspace root). "
        "model_loaded=False; inference will return HTTP 500."
    )

# ---------------------------------------------------------------------------
# Startup log — no sensitive file paths (Requirement 17.5)
# ---------------------------------------------------------------------------

logger.info(
    "Dashboard startup — port=%d, device=%r, confidence_threshold=%s, "
    "hazard_store_capacity=%d, camera_stub_id=%r, model_loaded=%s, "
    "checkpoint_source=%r, checkpoint=%s",
    _PORT,
    _DEVICE,
    _CONF_THRESHOLD,
    _STORE_CAPACITY,
    _CAMERA_STUB_ID,
    model_loaded,
    _checkpoint_resolver.source,
    _resolved_checkpoint or "(none)",
)

# ---------------------------------------------------------------------------
# FrameSourceManager + Auto-Cycle Thread (Dashboard v2)
# ---------------------------------------------------------------------------

import threading
import time as _time

from dashboard.frame_source import FrameSourceManager, FrameInfo, load_map_config

_SYNTH_DIR: Path = _WORKSPACE_ROOT / "image_data_with_synth"
_FALLBACK_DIR: Path = _WORKSPACE_ROOT / "roboflow data"
_MAP_CONFIG_PATH: Path = _WORKSPACE_ROOT / "config" / "dashboard_map.json"

# CHANGE LATER WHEN SUPERVISOR REVIEWS — hourly cycling is a placeholder;
# reduce interval for real-time demonstration once live cameras are integrated.
# Set DASHBOARD_CYCLE_MINUTES=10 for demo mode (10-minute cycle),
# or leave unset for the default 60-minute (hourly) production cycle.
_CYCLE_MINUTES: int = int(os.environ.get("DASHBOARD_CYCLE_MINUTES", "60"))
_CYCLE_INTERVAL_SECONDS: int = _CYCLE_MINUTES * 60

_map_config = load_map_config(_MAP_CONFIG_PATH)

frame_source_manager = FrameSourceManager(
    synth_dir=_SYNTH_DIR,
    fallback_dir=_FALLBACK_DIR,
    map_config=_map_config,
    cycle_interval_seconds=_CYCLE_INTERVAL_SECONDS,
)

# Current auto-cycle result (updated by background thread)
_current_cycle_result: dict | None = None
_cycle_lock = threading.Lock()


def _auto_cycle_loop():
    """
    Background thread: periodically selects a new frame from the frame source,
    runs inference, and stores the result for /api/cycle/current.
    """
    # CHANGE LATER WHEN SUPERVISOR REVIEWS — hourly cycling is a placeholder;
    # reduce interval for real-time demonstration once live cameras are integrated.
    global _current_cycle_result

    while True:
        frame_info = frame_source_manager.get_current_frame()
        if frame_info is not None and inference_engine is not None:
            try:
                results = inference_engine.run(
                    frame_info.image,
                    camera_id=f"location_{frame_info.map_location}",
                    folder_name=frame_info.folder_name,
                )

                # Annotate the image
                try:
                    annotated_b64 = annotate(frame_info.image, results)
                except Exception:
                    annotated_b64 = None

                # Store hazard events
                timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                for r in results:
                    if r.is_hazard:
                        event = HazardEvent(
                            event_id=str(uuid.uuid4()),
                            hazard_type=r.hazard_reason,
                            camera_id=f"location_{frame_info.map_location}",
                            timestamp=timestamp_utc,
                            confidence=r.confidence,
                            bbox=r.bbox,
                            annotated_image=annotated_b64,
                            location=LocationContext.from_camera_id(f"cam_stub_{frame_info.map_location:02d}"),
                        )
                        hazard_store.append(event)

                # Build cycle result for the API
                with _cycle_lock:
                    _current_cycle_result = {
                        "annotated_image": annotated_b64,
                        "detections": [r.to_dict() for r in results],
                        "map_location": frame_info.map_location,
                        "folder_name": frame_info.folder_name,
                        "bucket": frame_info.bucket,
                        "is_synthetic": frame_info.is_synthetic,
                        "disclaimer": frame_source_manager.source_disclaimer,
                        "timestamp": timestamp_utc,
                    }

            except Exception as exc:
                logger.error("Auto-cycle inference failed: %s: %s", type(exc).__name__, exc)

        # CHANGE LATER WHEN SUPERVISOR REVIEWS — hourly cycling is a placeholder;
        # reduce interval for real-time demonstration once live cameras are integrated.
        _time.sleep(frame_source_manager._cycle_interval)


# Start auto-cycle in a daemon thread (won't block shutdown)
_auto_cycle_thread = threading.Thread(target=_auto_cycle_loop, daemon=True, name="auto-cycle")
_auto_cycle_thread.start()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/cycle/current", methods=["GET"])
def api_cycle_current():
    """
    GET /api/cycle/current

    Returns the most recent auto-cycle inference result, or null fields
    when no cycle has completed yet.

    Response JSON:
      { annotated_image, detections, map_location, folder_name,
        bucket, is_synthetic, disclaimer, timestamp }
    """
    with _cycle_lock:
        if _current_cycle_result is None:
            return jsonify({
                "annotated_image": None,
                "detections": [],
                "map_location": None,
                "folder_name": None,
                "bucket": None,
                "is_synthetic": False,
                "disclaimer": "",
                "timestamp": None,
            })
        return jsonify(_current_cycle_result)


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



@app.route("/api/hazards/recent", methods=["GET"])
def api_hazards_recent():
    """
    GET /api/hazards/recent

    Returns the 3 most recent hazard events, newest first.
    Returns [] when no events are stored.

    Requirement: 10.5
    """
    recent = hazard_store.get_recent(3)
    return jsonify([e.to_dict() for e in recent])


@app.route("/api/map/config", methods=["GET"])
def api_map_config():
    """
    GET /api/map/config

    Returns the map configuration (pin positions, location names, folder-to-location
    mapping) from config/dashboard_map.json. Used by terminal_map.js to position
    pins on the site map PNG.

    Requirement: 3.4 (yard-hazard-inference-dashboard-v2 spec)
    """
    import json as _json
    map_config_path = _WORKSPACE_ROOT / "config" / "dashboard_map.json"
    if map_config_path.is_file():
        try:
            with open(map_config_path, "r", encoding="utf-8") as f:
                return jsonify(_json.load(f))
        except Exception as e:
            logger.warning("api_map_config: failed to read dashboard_map.json: %s", e)
    return jsonify({"error": "Map config not found"}), 404


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
