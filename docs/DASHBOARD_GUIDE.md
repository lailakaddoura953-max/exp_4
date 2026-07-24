# Yard Hazard Inference Dashboard — User & Developer Guide

## Quick Start

**One-step launch:**
```batch
start_dashboard.bat
```

The dashboard opens at `http://localhost:5000` with the Flask backend serving both the web UI and REST API.

---

## System Architecture (Current State)

### Overview

The dashboard is a **single Flask process** serving:
- **Static files** (HTML, CSS, JavaScript) on `/` and `/static/`
- **REST API endpoints** on `/api/*` for inference, storage, and status

This is a **minimal demo system** with no user-facing authentication and no multi-camera integration. Most components are stubbed or mocked to enable development without external dependencies — **with one exception**: the dashboard now supports one real live Wisenet camera over RTSP (see [Live Camera Capture](#live-camera-capture-real-rtsp-camera) below), including a persistent SQLite-backed archive of every captured frame. Full setup instructions for connecting a real camera are in `docs/LIVE_CAMERA_SETUP.md`.

```
Browser (http://localhost:5000)
    ↓
Flask Backend (src/dashboard/app.py)
    ├── GET /                          → index.html (SPA entry point)
    ├── GET /static/*.css|js|svg        → static files
    │
    ├── POST /api/inference             → upload image → run YOLO → return results + annotated image
    ├── GET  /api/hazards/recent        → fetch 3 most recent events from in-memory store
    ├── GET  /api/status                → return system health (model_loaded, hazard_count, etc.)
    ├── GET  /api/test-image            → fetch random dataset image for demo
    ├── POST /api/live-camera/capture   → trigger a real-camera 5-frame RTSP burst + inference
    ├── GET  /api/live-camera/status    → poll live-camera config/progress/last result
    └── GET  /api/live-camera/history   → query archived frames from capture_log.db (SQLite)
        ↓
    Inference Engine (src/dashboard/inference_engine.py)
        ├── Load checkpoint at startup
        ├── Accept image + camera_id
        └── Run YOLO → apply hazard rules → return HazardResult list
            ↓
        YOLO Detector (existing, src/hazard_detection/yolo_detector.py)
            └── Load pretrained model, run inference on frame
                ↓
            Roboflow Dataset (roboflow data/test|train/images)
                ├── Used for /api/test-image endpoint
                └── User can also upload custom images
```

**Data flow for inference:**

1. User uploads an image via the dashboard UI.
2. Browser POST to `/api/inference` with image + `camera_id` form data.
3. Flask backend:
   - Decodes image from multipart form
   - Passes to InferenceEngine with `camera_id="cam_stub_01"` (hardcoded stub)
   - Receives list of HazardResult records
   - Draws annotated bounding boxes on the image (red=hazard, green=safe)
   - Stores hazard events in in-memory HazardStore (capped at 20 events)
   - Returns JSON: `{ "results": [...], "annotated_image": "<base64 PNG>" }`
4. Browser displays annotated image, detection list, and location metadata (all stubbed).
5. Page refreshes fetch `/api/hazards/recent` to populate the Recent Detections card grid.

---

## Frontend: How It Works

### Page Structure

**Header**
- Title: "Yard Hazard Inference Dashboard"
- Status badge: "Connected" (green) or "Disconnected" (red) — updated on page load via `/api/status` poll

**Stats Section**
- Three cards showing:
  - Hazard Count (from HazardStore)
  - Model Status (Loaded / Not Loaded)
  - Active Cameras (stub: always shows 1 or 0)

**Terminal Map**
- SVG schematic of the yard (stub layout, not geo-referenced)
- 15 camera pins representing stub camera positions
- When you click a Recent Detection card, the corresponding camera pin highlights
- **TODO (future)**: Replace SVG with real geo-referenced map image and real camera coordinates

**Recent Detections Section**
- Grid of up to 6 cards, each showing:
  - **Annotated image** (from previous inference runs or dataset)
  - **Metadata**: Hazard type, Location (berth/crane/camera), Timestamp, Confidence
  - **View Details button**: Opens modal with full event information
- Cards fetched from `/api/hazards/recent` at page load
- **TODO (future)**: Real-time card updates as new hazards are detected (WebSocket or polling)

**Run Inference Section**
- **Upload area**: Drag-and-drop or click-to-browse for JPEG/PNG images
- **Preview**: Shows selected image before submission
- **Run Inference button**: POSTs image to `/api/inference`, shows spinner, displays results
- **Use Dataset Image button**: Fetches random image from `/api/test-image` and pre-loads it
- **Results area**:
  - Annotated image (with bounding boxes drawn by backend)
  - Detection table: Class Label | Confidence | Status (HAZARD/NOT HAZARD) | Reason
  - Location strip: Facility · Berth · Crane · Camera (all stub values)

### Location Context (Stubbed)

Every hazard event includes a `LocationContext`:
```json
{
  "facility": "Railyard",
  "berth": "Berth 403",
  "crane": "Crane 01",
  "camera": "Camera 01",
  "landmark": ""
}
```

**Current behavior**: Derived from `camera_id` string suffix using a hardcoded lookup table.

**Example**:
- `cam_stub_01` → Berth 403, Crane 01, Camera 01
- `cam_stub_05` → Berth 405, Crane 05, Camera 05

**TODO (future)**:
- Replace with real camera registry (database or config file)
- Fetch actual berth/crane/GPS coordinates from camera metadata
- Support arbitrary number of berths, cranes, and cameras
- Add landmark names (e.g., "Gate 01", "Reefer Racks") based on yard map

### Hazard Type Categorisation

Hazard cards are colour-coded by type:
- **Orange border**: Container hazards (misaligned, water drop, open unsecured, picked no crane, picked person below crane, flipped)
- **Red border**: Human/PPE hazards (ppe_violation, human_below_crane, human_detected_stub)
- **Yellow/Grey**: Other or no hazard

**Current values from Roboflow YOLO classes:**
```python
CONTAINER_REASONS = {
    'misaligned_container',
    'water_drop_container',
    'open_container_unsecured',
    'picked_no_crane',
    'picked_person_below_crane',
    'flipped_container',
}

HUMAN_REASONS = {
    'ppe_violation',
    'human_below_crane',
    'human_detected_stub',
}
```

**TODO (future)**:
- Hazard severity levels (Critical / Warning / Info) with numeric scoring
- Alert routing based on hazard type (different recipients for PPE vs. container)
- Snooze/acknowledge logic (suppress duplicate alerts for 5 min after first alert)

---

## Backend: Flask API

### Endpoints

#### `POST /api/inference`

Runs hazard detection on a single image.

**Request**:
```
Content-Type: multipart/form-data

image      (required, file)  — JPEG or PNG image
camera_id  (optional, str)   — defaults to "cam_stub_01"
```

**Response (200 OK)**:
```json
{
  "results": [
    {
      "class_label": "Container - Misaligned",
      "confidence": 0.87,
      "bbox": {
        "x_center": 0.45,
        "y_center": 0.33,
        "width": 0.12,
        "height": 0.09
      },
      "is_hazard": true,
      "hazard_reason": "misaligned_container",
      "camera_id": "cam_stub_01"
    },
    {
      "class_label": "Human",
      "confidence": 0.62,
      "bbox": { "x_center": 0.5, "y_center": 0.7, "width": 0.05, "height": 0.1 },
      "is_hazard": true,
      "hazard_reason": "human_detected_stub",
      "camera_id": "cam_stub_01"
    }
  ],
  "annotated_image": "iVBORw0KGgoAAAANSUhEUgAA..." (base64 PNG)
}
```

**Response (400 Bad Request)**:
```json
{ "error": "No image provided" }
```

**Response (500 Internal Server Error)**:
```json
{ "error": "<exception message>" }
```

**Side effects**:
- Every detection with `is_hazard=True` creates a HazardEvent record
- HazardEvent stored in-memory (capped at 20 most recent)
- Annotated image (base64) included in HazardEvent for modal display

**Current limitations**:
- Only one `camera_id` parameter (no multi-camera upload)
- No image dimensions or quality validation
- Annotation failures (e.g., invalid color space) return partial/null result

**TODO (future)**:
- Batch inference (multiple images per request)
- Real camera integration: read `camera_id` from camera metadata (EXIF)
- Dynamic thresholds per camera (different lighting/angles)
- Async inference with job queue (current is synchronous)

---

#### `GET /api/hazards/recent`

Fetch up to 3 most recent hazard events.

**Request**:
```
No parameters
```

**Response (200 OK)**:
```json
[
  {
    "event_id": "a1b2c3d4-e5f6-...",
    "hazard_type": "misaligned_container",
    "camera_id": "cam_stub_01",
    "timestamp": "2026-07-09T14:32:45Z",
    "confidence": 0.87,
    "bbox": { "x_center": 0.45, "y_center": 0.33, "width": 0.12, "height": 0.09 },
    "annotated_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "location": {
      "facility": "Railyard",
      "berth": "Berth 403",
      "crane": "Crane 01",
      "camera": "Camera 01",
      "landmark": ""
    }
  }
]
```

Returns empty array `[]` if no hazards stored.

**Current limitations**:
- In-memory only (lost on restart)
- Fixed 3-event limit (no pagination)
- No filtering (can't query by hazard type, camera, time range)

**TODO (future)**:
- PostgreSQL backend for persistent storage
- Pagination: `?limit=10&offset=0`
- Filtering: `?hazard_type=ppe_violation&camera_id=cam_01&since=2026-07-09T00:00:00Z`
- Full-text search on hazard reasons
- Export to CSV

---

#### `GET /api/status`

Check system health and configuration.

**Request**:
```
No parameters
```

**Response (200 OK)**:
```json
{
  "status": "running",
  "model_loaded": true,
  "hazard_count": 5,
  "camera_id": "cam_stub_01"
}
```

**Interpretation**:
- `status`: Always "running" (no shutdown endpoint yet)
- `model_loaded`: True if YOLO checkpoint loaded successfully, False if checkpoint is missing/corrupted
- `hazard_count`: Current size of HazardStore (≤ 20)
- `camera_id`: Active stub camera ID (hardcoded, not configurable via API)

**Current limitations**:
- No detailed error messages if model failed to load
- No performance metrics (inference latency, FPS, GPU memory)
- No alert threshold or configuration status

**TODO (future)**:
- Return per-camera connection status (online/offline/error)
- Include inference latency percentiles (p50, p95, p99)
- Include GPU/CPU utilization and memory
- Return active alert thresholds and configuration

---

#### `GET /api/test-image`

Fetch a random image from the Roboflow dataset for demo/testing.

**Request**:
```
No parameters
```

**Response (200 OK)**:
```
Content-Type: image/jpeg

<raw JPEG bytes>
```

**Response (404 Not Found)**:
```json
{ "error": "No dataset images found" }
```

**Current behavior**:
1. Tries to load from `roboflow data/test/images/`
2. Falls back to `roboflow data/train/images/` if test/ is empty
3. Randomly selects one image from available JPEG/PNG files
4. Returns as raw JPEG bytes (not base64)

**Current limitations**:
- No image selection strategy (purely random, may return blurry/useless images)
- No caching (re-reads filesystem on every call)
- No way to request images by class or specific camera

**TODO (future)**:
- Replace with live camera feed endpoint
- Stratified sampling (equal distribution of classes)
- Image quality scoring to skip blurry frames
- Rate limiting (e.g., max 1 image per second per camera)
- Streaming live video as Motion JPEG or HLS

---

## Live Camera Capture (Real RTSP Camera)

Unlike every other data path in this dashboard, the live camera feature
connects to **one real physical camera** (a Hanwha Wisenet IP camera) over
RTSP — it's the one part of the system that isn't stubbed or mocked.

### How it works

1. **Trigger**: either the "Capture 5-Frame Burst" button in the Live
   Camera UI section, or an automatic hourly timer (60-minute floor,
   enforced — see `MIN_AUTO_CAPTURE_INTERVAL_MINUTES` in
   `src/dashboard/live_camera.py`).
2. **Capture**: `LiveCameraCapture` opens `rtsp://user:pass@ip:554/profileN/media.smp`
   via `cv2.VideoCapture`, reads 5 frames ~150ms apart, then closes the
   connection. Connections are opened fresh per burst rather than held
   open — simpler and more robust for an hourly/on-demand cadence.
3. **Inference**: each of the 5 frames goes through the exact same
   `InferenceEngine.run()` and `annotate()` calls used by dataset auto-cycle
   and manual upload — no separate code path for "real" vs. "dataset"
   frames downstream of capture.
4. **Archival**: every frame — hazard or not — is saved to
   `live_camera_captures/<camera_id>/<date>/<frame_id>.jpg` + a `.json`
   sidecar, and mirrored into a SQLite database
   (`live_camera_captures/capture_log.db`). This is deliberately different
   from `HazardStore`, which only ever retains hazard events. See
   [Data Retention: Keep Everything](#data-retention-keep-everything) below.

### Endpoints

- **`POST /api/live-camera/capture`** — triggers an on-demand burst,
  returns the full result synchronously (frame capture + 5x inference
  typically takes a few seconds). Returns HTTP 409 if a burst is already
  running (from either this endpoint or the hourly timer).
- **`GET /api/live-camera/status`** — returns whether the feature is
  configured (`config/ip_addresses.json` present + model loaded), whether a
  capture is currently in progress, the auto-capture interval, the total
  `archived_frame_count` in the database, and the most recent burst result.
- **`GET /api/live-camera/history?limit=20&hazard_only=false`** — returns
  recent rows straight from `capture_log.db`'s `frames` table (including
  the same location fields — facility/berth/crane/camera/landmark — shown
  elsewhere in the dashboard's detail view), without touching the raw JSON
  files.

### Data Retention: Keep Everything

Most hazard-detection systems only retain flagged events and discard
everything else — that's exactly what `HazardStore` does on its own
(in-memory, 20-event cap, hazard-only, lost on restart). The live camera
path adds a second, parallel archive that keeps **every** frame:

```
live_camera_captures/
├── capture_log.db                       ← SQLite, portable with this folder
└── <camera_id>/
    └── <YYYY-MM-DD>/
        ├── <frame_id>.jpg                ← raw frame, kept forever
        └── <frame_id>.json               ← detections + hazard flag
```

`frame_id` is the frame's own timestamp-derived filename — already unique,
and it's the primary key in both the filesystem layout and the database's
`frames` table, so any database row traces back to an exact file pair on
disk.

`capture_log.db` is created automatically (schema included) the first time
a frame is saved. If it's ever missing or deleted, calling
`sync_archive_to_db()` (in `src/dashboard/live_camera.py`) rebuilds it
completely by walking every `.json` sidecar already on disk — no manual
import step required, and safe to re-run at any time (upserts by
`frame_id`, never duplicates).

Nothing in this path deletes a file. Disk usage grows without bound by
design — monitor available disk space on whichever machine runs long-term
captures. There's no pruning logic; adding retention limits later would be
a deliberate, separate change.

### Configuration

Real camera credentials live in `config/ip_addresses.json` (gitignored,
per-machine — never committed). Copy
`config/ip_addresses_template.json` to get started. Environment variables
`DASHBOARD_LIVE_CAMERA_AUTO` (default `true`) and
`DASHBOARD_LIVE_CAMERA_INTERVAL_MINUTES` (default `60`, clamped to a
60-minute floor) control the automatic timer. Full setup walkthrough
(including what to change on a second device that has the physical
camera): **`docs/LIVE_CAMERA_SETUP.md`**.

---

## Inference Engine: How It Works

### Single-Frame Hazard Classification

**File**: `src/dashboard/inference_engine.py`

**Flow**:

1. **Init**: Load YOLO checkpoint, validate config (threshold in [0.0, 1.0], device is "cpu" or "cuda")
2. **Run**: 
   ```python
   def run(self, image: np.ndarray, camera_id: str) -> List[HazardResult]:
       # Wrap image in single-frame FrameSequence
       frame_seq = FrameSequence([image], [0])  # timestamp=0 (unused)
       
       # Run YOLO detector
       detections = yolo_detector.detect(frame_seq)  # returns List[List[Detection]]
       
       # Get detections for frame 0 (only frame in sequence)
       frame_detections = detections[0]
       
       # Apply hazard rules
       hazard_results = rule_engine.classify_all(
           detections=frame_detections,
           config=self.config,
           camera_id=camera_id
       )
       
       return hazard_results
   ```

3. **Output**: List of `HazardResult` records (one per detection above confidence threshold)

### Hazard Rules (Current Implementation)

**File**: `src/dashboard/rules.py`

Rules are applied in priority order. Each detection is classified once and skips remaining rules.

#### Rule 1: Unconditional Hazards
```python
if class_label in ["Container - Misaligned", "Container - Water Drop"]:
    is_hazard = True
    hazard_reason = "misaligned_container" or "water_drop_container"
```

**TODO (future)**: Add severity levels; some hazards (fire, structural failure) should escalate immediately.

#### Rule 2: Container Open — Loading Operation Suppression
```python
if class_label == "Container - Open":
    if any(crane_or_picked_box has IoU >= 0.5 with this box):
        is_hazard = False  # Suppressed: active loading
        hazard_reason = ""
    else:
        is_hazard = True
        hazard_reason = "open_container_unsecured"
```

**Current limitation**: No temporal awareness — rule only checks current frame, not history.

**TODO (future)**:
- Track loading operations across frames (duration, progress)
- Suppress alerts only during declared loading windows
- Learn suppression windows from operations API (crane status, vessel schedule)

#### Rule 3: Container Picked — Crane Proximity
```python
if class_label == "Container - Picked":
    if no Crane detection exists:
        is_hazard = True
        hazard_reason = "picked_no_crane"
    elif any(Human with y_center >= Crane y_center):
        is_hazard = True
        hazard_reason = "picked_person_below_crane"
    else:
        is_hazard = False
```

**Current limitation**: "Below crane" is pixel-based, not real meters. Works on a single frame.

**TODO (future)**:
- Calibrate crane bounding box to real distance (GPS or marker-based)
- Track people across frames to measure exposure duration
- Require minimum confidence (e.g., 70%) before flagging person under crane

#### Rule 4: Human — PPE Violation
```python
if class_label == "Human - No Safety Clothes":
    is_hazard = True
    hazard_reason = "ppe_violation"
```

**TODO (future)**: 
- Refine PPE classes (hard hat only vs. full body harness)
- Check zone-based PPE rules (e.g., different zones require different PPE)
- Track PPE violation duration (issue alert only if > 30 seconds in zone)

#### Rule 5: Human — Stub (No Zone Logic Yet)
```python
if class_label == "Human" (and not "No Safety Clothes"):
    # STUB: No zone map available yet
    is_hazard = True
    hazard_reason = "human_detected_stub"
    
    # Special case: person below crane takes precedence
    if any(Crane and person y_center >= Crane y_center):
        is_hazard = True
        hazard_reason = "human_below_crane"
```

**TODO (future)**:
- Load zone map (polygon coordinates per zone: "Yard - No People", "Yard - Operation Zone", "Yard - Dropoff")
- Assign detection to zone (point-in-polygon test)
- Classify based on zone:
  - Zone "No People": `is_hazard = True`
  - Zone "Operation Zone": `is_hazard = conditional` (depends on vessel schedule)
  - Zone "Dropoff": `is_hazard = False`

#### Rule 6: Flipped Container
```python
if class_label in [container classes]:
    aspect_ratio = bbox.height / bbox.width
    if aspect_ratio > flipped_threshold (default 1.5):
        is_hazard = True
        hazard_reason = "flipped_container"
```

**Current limitation**: Simple aspect ratio check; no rotated bounding box support.

**TODO (future)**:
- Use rotated bounding boxes (YOLO can provide angle)
- Distinguish "flipped" (upside down) from "tilted" (angled)
- Check container damage (cracks, dents) via separate model

#### Rule 7: Non-Hazard Classes
```python
if class_label in [
    "Container - Separate",
    "Container - Stacked",
    "Container - Reefer",
    "Vehicle",
    "Truck - No Container",
    "Truck - With Container",
    "Boat - With Cargo",
]:
    is_hazard = False
    hazard_reason = ""
    # Unless flipped (Rule 6 takes precedence)
```

---

## Configuration & Environment Variables

**File**: `src/dashboard/app.py`

All configuration read from environment variables at startup:

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DASHBOARD_PORT` | `5000` | int | Flask port |
| `DASHBOARD_DEVICE` | `cpu` | str | "cpu" or "cuda" for YOLO inference |
| `DASHBOARD_CONF_THRESHOLD` | `0.5` | float | YOLO confidence threshold [0.0, 1.0] |
| `DASHBOARD_STORE_CAPACITY` | `20` | int | Max hazard events in memory |
| `DASHBOARD_CAMERA_STUB_ID` | `cam_stub_01` | str | Stub camera ID |

**Example**: Run on GPU with lower confidence threshold:
```batch
set DASHBOARD_DEVICE=cuda
set DASHBOARD_CONF_THRESHOLD=0.3
python -m src.dashboard.app
```

**TODO (future)**:
- Load from YAML config file instead of environment variables
- Support multiple camera configurations
- Per-camera thresholds (different cameras may need different settings)

---

## Running Commands (Manual)

If you prefer not to use `start_dashboard.bat`, here are the individual commands:

### Activate virtual environment
```batch
.venv\Scripts\activate.bat
```

### Run Flask backend only
```batch
python -m src.dashboard.app
```

The dashboard will be available at `http://localhost:5000`.

### Run with custom configuration
```batch
set DASHBOARD_DEVICE=cuda
set DASHBOARD_CONF_THRESHOLD=0.4
python -m src.dashboard.app
```

### Run tests
```batch
pytest tests/
```

### Build requirements from source (if dependencies change)
```batch
pip freeze > requirements.txt
```

---

## Current Limitations & Future Work

### Data Storage
- **Current**: `HazardStore` (hazard events only) is in-memory (20 events, lost on restart). Separately, the live camera path now persists **every** frame (hazard or not) to `live_camera_captures/` on disk with a SQLite index (`capture_log.db`) — see [Live Camera Capture](#live-camera-capture-real-rtsp-camera) above. Only the hazard-event side is still purely in-memory.
- **TODO**: Add PostgreSQL backend for `HazardStore`-equivalent persistent event storage and querying; the raw-frame archive already persists independently

### Camera Integration
- **Current**: The dashboard's main upload/auto-cycle path still uses the stubbed `CameraStub` class (`camera_id="cam_stub_01"`). **One real camera is wired in separately** via `src/dashboard/live_camera.py` — RTSP burst capture, on-demand button + hourly timer, feeding the same `InferenceEngine`.
- **TODO**:
  - Multi-camera support (currently only the first entry in `config/ip_addresses.json` is used)
  - Camera health monitoring (connection status, frame rate)
  - Automatic failover (if camera offline, use backup)
  - Extend `CameraStub`'s upload/auto-cycle path itself to use real cameras (currently only the separate live-camera panel does)

### Zone & Facility Mapping
- **Current**: Hardcoded berth/crane/camera names in stub lookup table
- **TODO**:
  - Load from yard operations database
  - Geo-referenced map (Google Maps, Mapbox, or custom tile layer)
  - Polygon zones for hazard classification (no-people zone, operation zone)
  - Real-time yard map updates

### Hazard Rules
- **Current**: Pixel-based spatial checks (IoU, bounding box overlap)
- **TODO**:
  - Temporal rules (duration of hazard, rate of change)
  - Multi-frame tracking (follow person across frames)
  - Calibrated distance (convert pixels to meters using camera intrinsics)
  - Advanced ML (learned rule weights via training)

### Alerting & Notifications
- **Current**: No alerting; only stores events in memory
- **TODO**:
  - Email/SMS/Slack alerts for Critical hazards
  - Alert thresholds and tuning
  - Snooze/acknowledge UI
  - Alert history and audit log

### Real-Time Updates
- **Current**: Dashboard polls `/api/hazards/recent` once on page load
- **TODO**:
  - WebSocket for live hazard feed
  - Server-Sent Events (SSE) as WebSocket alternative
  - Live video stream overlay (bounding boxes drawn in real-time)

### Performance & Scalability
- **Current**: Single-threaded Flask dev server, synchronous inference
- **TODO**:
  - Production WSGI server (gunicorn, uWSGI)
  - Async inference with job queue (Celery + Redis)
  - GPU inference batching (process multiple images in parallel)
  - Load balancing across multiple inference engines
  - Inference caching (same image input → same output)

### Model Improvements
- **Current**: Fixed YOLO model, no retraining pipeline
- **TODO**:
  - Model versioning (A/B testing multiple models)
  - Automated retraining on new labeled data
  - Online learning (fine-tune model with recent detections)
  - Ensemble models (combine predictions from multiple models)

---

## Troubleshooting

### Model fails to load (HTTP 500 on inference)
```
ERROR: Failed to initialise InferenceEngine — model_loaded=False
```

**Cause**: Checkpoint file missing or corrupted
**Fix**: Copy a trained model to `checkpoints/yolov12_best.pt`
```batch
copy runs\train\hazard_yolo\weights\best.pt checkpoints\yolov12_best.pt
```

### "No dataset images found" error
```
GET /api/test-image returns 404
```

**Cause**: `roboflow data/test/images/` directory is empty
**Fix**: Ensure the Roboflow dataset is downloaded
```batch
REM Roboflow data should be at: c:\...\exp_3\roboflow data\test\images\
dir "roboflow data\test\images" /b | find "." >nul
if errorlevel 1 echo No images found
```

### Dashboard shows "Disconnected"
**Cause**: Flask backend is not running or `/api/status` returns error
**Fix**: Check console output for startup errors; restart with:
```batch
start_dashboard.bat
```

### Inference is slow
**Cause**: Running on CPU instead of GPU
**Fix**: Install CUDA and run with:
```batch
set DASHBOARD_DEVICE=cuda
python -m src.dashboard.app
```

### Port 5000 already in use
```
Address already in use
```

**Cause**: Another process is using port 5000
**Fix**: Either kill the process or use a different port:
```batch
set DASHBOARD_PORT=5001
python -m src.dashboard.app
```

---

## Next Steps

1. **Test the dashboard**: Upload images, verify hazard detection works
2. **Integrate real cameras**: Replace `CameraStub` with RTSP/IP camera feed
3. **Add database**: Replace in-memory HazardStore with PostgreSQL
4. **Build alerting**: Send alerts to operators (email, SMS, dashboard notification)
5. **Create zone maps**: Load real yard geometry and zone definitions
6. **Tune rules**: Validate and calibrate hazard thresholds with yard operators
7. **Deploy**: Move to production server with HTTPS, authentication, and monitoring

