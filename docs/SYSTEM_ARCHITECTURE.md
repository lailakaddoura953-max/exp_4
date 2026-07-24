# System Architecture & Future Implementation Guide

**Project:** Yard Safety CCTV — Hazard Detection  
**Status:** Development / POC Stage (one real live camera wired into the dashboard via RTSP; production `main.py` pipeline still stubbed)  
**Last Updated:** July 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Inventory](#component-inventory)
4. [Data Flow: Current State](#data-flow-current-state)
5. [Live Camera Capture & Archive](#live-camera-capture--archive)
6. [Configuration Files](#configuration-files)
7. [What Works Today](#what-works-today)
8. [What's Still Stubbed](#whats-still-stubbed)
9. [TODO: Camera Integration](#todo-camera-integration)
10. [TODO: Production Deployment](#todo-production-deployment)
11. [TODO: Model Improvements](#todo-model-improvements)
12. [TODO: Dashboard Enhancements](#todo-dashboard-enhancements)
13. [TODO: Alerting System](#todo-alerting-system)

---

## System Overview

This system monitors an industrial container terminal (16 major zones, hundreds of cameras) for safety hazards using YOLOv12 object detection combined with a **location-aware hazard rule engine** that applies zone-specific safety policies to each detection before deciding whether it's a real hazard or expected/permitted behavior.

Two entry points exist:
- **Dashboard** (`src/dashboard/app.py`) — web-based inference UI, auto-cycles through dataset imagery, demonstrates the full pipeline visually. Now also supports pulling frames from **one real Wisenet RTSP camera** (`src/dashboard/live_camera.py`) — either on-demand (a "Capture 5-Frame Burst" button) or on an hourly timer — as a smaller, dashboard-scoped slice of full camera integration (see [What's Still Stubbed](#whats-still-stubbed) below).
- **Detection Pipeline** (`src/hazard_detection/main.py`) — the future production loop for live camera feeds, currently runs with a stub frame sampler serving dataset images

Both paths share the same YOLO model, rule engine, and class taxonomy.

A third, related concern — **data retention** — is also handled differently than a typical hazard-only system: every frame captured from the live camera is archived to disk and indexed in a SQLite database, regardless of whether it contained a hazard. See [Live Camera Capture & Archive](#live-camera-capture--archive) below.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Web Dashboard (Flask, port 5000)          Terminal Pipeline (main.py)  │
│  ├─ index.html + app.js + styles.css       ├─ Continuous camera loop    │
│  ├─ Auto-cycle (hourly / 10-min demo)      ├─ _StubFrameSampler         │
│  ├─ Manual image upload                    │   └─ FrameSourceManager    │
│  ├─ Live Camera (RTSP burst, real camera)  │       (real images)        │
│  │   ├─ On-demand button                   └─ Per-camera timeout/skip   │
│  │   └─ Hourly timer (60 min floor)                                    │
│  └─ REST API                                                            │
│      POST /api/inference                                                │
│      GET  /api/cycle/current                                            │
│      GET  /api/hazards/recent                                           │
│      GET  /api/status                                                   │
│      GET  /api/map/config                                               │
│      GET  /api/test-image                                               │
│      POST /api/live-camera/capture                                      │
│      GET  /api/live-camera/status                                       │
│      GET  /api/live-camera/history                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  YOLODetector (ultralytics YOLOv12)                                     │
│  ├─ Checkpoint: auto-discovered or config-specified                     │
│  ├─ Input: 640×640 normalized BGR frame                                 │
│  └─ Output: List[Detection] per frame (class_label, confidence, bbox)   │
│                                                                          │
│  Hazard Classification (two paths, same result):                         │
│  ├─ Dashboard: src/dashboard/rules.py (classify_all)                    │
│  └─ Pipeline:  src/hazard_detection/rule_engine/orchestrator.py         │
│       ├─ CameraLocationResolver (camera name → zone type)               │
│       ├─ TrainingFolderLocationResolver (folder name → zone type)       │
│       ├─ LocationRuleLoader (16 zone rule sets from rules.py + YAML)    │
│       ├─ check_human() / check_container() / check_vehicle()            │
│       ├─ check_tel_spot() / check_tel_occupancy()                       │
│       └─ AuditLogger (logs/rule_audit.jsonl)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Frame Sources:                                                          │
│  ├─ image_data_with_synth/ (real footage + synthetic hazards)           │
│  │   └─ augmented_hazards/ + normal_operations/                         │
│  ├─ roboflow data/ (annotated training dataset, ~935 train images)      │
│  └─ Live RTSP camera (one real camera, dashboard-scoped — see below)    │
│                                                                          │
│  Configuration:                                                          │
│  ├─ config/hazard_detection.yaml (pipeline config + checkpoint path)    │
│  ├─ config/location_rules.yaml (zone rule overrides + camera mappings)  │
│  ├─ config/dashboard_map.json (folder→location mapping + pin positions) │
│  └─ config/ip_addresses.json (real camera RTSP credentials, gitignored)│
│                                                                          │
│  Storage:                                                                │
│  ├─ HazardStore (in-memory, 20 events, hazard-only, lost on restart)    │
│  ├─ live_camera_captures/ (every live-camera frame, JPEG+JSON, kept    │
│  │   forever — hazard or not; see Live Camera Capture & Archive)       │
│  │   └─ capture_log.db (SQLite index over the same frames, auto-       │
│  │       generated/rebuildable from the JSON sidecars)                 │
│  ├─ logs/rule_audit.jsonl (every rule decision, persistent)             │
│  └─ [FUTURE] PostgreSQL for persistent hazard-event history            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Inventory

| Component | Location | Purpose |
|---|---|---|
| Detection Pipeline | `src/hazard_detection/main.py` | Production camera loop (currently stub) |
| DetectionPipeline class | `src/hazard_detection/detection_pipeline.py` | Orchestrates frame→detect→analyze→dispatch per camera |
| YOLODetector | `src/hazard_detection/yolo_detector.py` | Ultralytics YOLOv12 wrapper |
| HumanDetector | `src/hazard_detection/human_detector.py` | Zone-map-based human hazard logic (legacy path) |
| ContainerAnalyzer | `src/hazard_detection/container_analyzer.py` | Temporal container hazard analysis |
| Rule Engine | `src/hazard_detection/rule_engine/` | Location-aware hazard rules (16 zone types) |
| Dashboard App | `src/dashboard/app.py` | Flask web UI + REST API |
| InferenceEngine | `src/dashboard/inference_engine.py` | Single-frame YOLO + rules wrapper |
| FrameSourceManager | `src/dashboard/frame_source.py` | Auto-cycles through dataset images |
| CheckpointResolver | `src/dashboard/checkpoint_resolver.py` | Finds the best available .pt checkpoint |
| LiveCameraCapture / LiveCaptureService | `src/dashboard/live_camera.py` | Opens a real camera's RTSP stream, bursts 5 frames, runs inference, archives every frame |
| LiveCaptureArchiver | `src/dashboard/live_camera.py` | Saves every captured frame (JPEG + JSON) to disk, hazard or not |
| CaptureDatabase | `src/dashboard/capture_db.py` | SQLite index (`frames` + `detections` tables) over the archived frames; auto-rebuilds from JSON sidecars |
| Class Taxonomy | `src/hazard_detection/rule_engine/class_taxonomy.py` | Shared 17-class / 12-class (reduced) lists |

---

## Data Flow: Current State

### Dashboard Auto-Cycle (every hour, or every 10 min in demo mode)

```
1. FrameSourceManager picks next image from roboflow data/ (or image_data_with_synth/ if available)
2. InferenceEngine.run(image, camera_id="location_N", folder_name="berth_401")
3. YOLODetector.detect(frame_sequence) → List[Detection]
4. dashboard.rules.classify_all(detections, config, camera_id) → List[HazardResult]
5. Hazard results → HazardStore (if is_hazard=True)
6. Annotated image + results → /api/cycle/current cache
7. Frontend polls /api/cycle/current → updates Live Inference display
```

### Manual Upload (user drops image on the dashboard)

```
1. POST /api/inference with image + camera_id
2. Same InferenceEngine.run() path as above
3. Results returned immediately as JSON response
4. Frontend displays annotated image + detection table
```

### Terminal Pipeline (main.py, stub mode)

```
1. _StubFrameSampler → FrameSourceManager.get_random_frame() → real image
2. YOLODetector.detect(frame_sequence with duplicated single frame)
3. HumanDetector.analyze() / ContainerAnalyzer.analyze() (existing legacy analyzers)
4. AlertDispatcher.dispatch() (log-only channel)
5. Cycle to next camera, repeat every ~8ms per camera (fast when stub)
```

### Live Camera Burst (button click, or hourly timer)

```
1. LiveCameraCapture opens rtsp://user:pass@ip:554/profileN/media.smp via cv2.VideoCapture
2. Reads 5 frames in quick succession (~150ms apart), releases the connection
3. For each frame:
   a. InferenceEngine.run(frame, camera_id) — same engine as auto-cycle/upload
   b. annotate(frame, results) — same annotator
   c. Hazard detections → HazardStore (hazard-only, same as other paths)
   d. Every frame (hazard or not) → LiveCaptureArchiver.save_frame()
        → live_camera_captures/<camera_id>/<date>/<frame_id>.jpg + .json
        → CaptureDatabase.record_frame() mirrors the same data into
          capture_log.db's frames/detections tables
4. Full burst result → /api/live-camera/status cache; frontend polls + renders
```

See [Live Camera Capture & Archive](#live-camera-capture--archive) for why every
frame (not just hazards) is kept, and how the SQLite index works.

---

## Live Camera Capture & Archive

This is the one part of the system that deliberately differs from a
"standard" hazard-detection setup: **every frame captured from the live
camera is kept, not just the ones flagged as hazards.**

### Why

Most hazard-detection systems discard non-hazard frames — only flagged
events get retained. `HazardStore` alone behaves that way here too
(in-memory, 20-event cap, hazard-only, wiped on restart). But the live
camera path adds a second, parallel retention mechanism that keeps
*everything*, so the full capture history is available later for review,
audit, or reuse (e.g. retraining on real "no hazard" frames, not just
synthetic ones).

### How

```
Camera (RTSP) → LiveCameraCapture → 5 raw frames
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
            InferenceEngine.run()   annotate()          LiveCaptureArchiver
                    │                     │              .save_frame()
                    ▼                     ▼                     │
              HazardResult[]      annotated PNG                 │
                    │                (for UI)                  │
                    ▼                                           ▼
          HazardStore.append()                    live_camera_captures/<camera_id>/<date>/
          (hazard-only, in-memory,                    <frame_id>.jpg   (raw frame, kept forever)
           capped, lost on restart)                   <frame_id>.json  (detections + hazard flag)
                                                              │
                                                              ▼
                                                  CaptureDatabase.record_frame()
                                                  capture_log.db → frames + detections tables
```

- **`frame_id`** is the frame's own filename stem (its capture timestamp,
  filesystem-safe — e.g. `2026-07-24T14-03-01.120000Z_frame0`). It's the
  primary key in both the filesystem layout and the `frames` table, so a
  database row always traces back to an exact `.jpg`/`.json` pair on disk.
- **SQLite, not a server database**: chosen specifically because it's
  file-based and portable. `capture_log.db` lives inside
  `live_camera_captures/` itself, so copying that one directory to another
  machine brings the whole queryable history with it.
- **Auto-generating / self-healing**: `capture_log.db` is created
  automatically (schema included) the first time a frame is saved — no
  manual setup. If it's ever missing or deleted,
  `sync_archive_to_db()` rebuilds it completely by walking every JSON
  sidecar already on disk. Both the live-write path and the rebuild path
  funnel through the same upsert logic (matched on `frame_id`), so
  re-running the rebuild is always safe.
- **No pruning**: nothing in this path ever deletes a file or a row. Disk
  usage grows without bound by design — that's the explicit "keep
  everything" requirement, not an oversight. Disk space should be
  monitored on whichever machine runs long-term captures.

Two dashboard endpoints expose this data:
- `GET /api/live-camera/status` — includes `archived_frame_count` (total
  rows in `frames`), so the archive's growth is visible from the same
  status poll the UI already uses.
- `GET /api/live-camera/history?limit=20&hazard_only=false` — returns
  recent frame rows (including the location fields shown in the
  dashboard's detail view) straight from `capture_log.db`.

Full setup and query instructions: **`docs/LIVE_CAMERA_SETUP.md`**.

---

## Configuration Files

| File | Purpose | Edit when... |
|---|---|---|
| `config/hazard_detection.yaml` | Main system config (cameras, thresholds, checkpoint) | Changing model, adding cameras, tuning thresholds |
| `config/location_rules.yaml` | Zone rule overrides + camera_name_overrides + camera_id_to_name | HSSE confirms a pending rule, new camera names appear |
| `config/dashboard_map.json` | Folder→location mapping + map pin positions | New dataset folders appear, pin positions need adjustment |
| `config/ip_addresses_template.json` | Committed template for live-camera RTSP credentials | Reference when setting up a new machine |
| `config/ip_addresses.json` | Real live-camera RTSP credentials (gitignored, per-machine) | Connecting a real camera on a given device — see `docs/LIVE_CAMERA_SETUP.md` |

---

## What Works Today

- YOLO inference on any image (upload or auto-cycle)
- 12-class Reduced_Class_Set defined and centralized
- 16 zone types with HSSE-confirmed rules (some pending — documented, defaulted safe)
- Location resolution from camera names (live) and folder names (dataset)
- Auto-cycling dashboard with real dataset images
- Hazard event storage and recent-events API
- Site map PNG displayed on dashboard
- Checkpoint auto-discovery (most recent best.pt)
- Full audit trail of every rule decision (JSON-lines)
- **One real live Wisenet camera** wired into the dashboard via RTSP — on-demand burst capture (button) and an hourly automatic timer, both running the same inference pipeline as dataset images
- **Every live-camera frame archived** (JPEG + JSON), hazard or not, plus a self-healing SQLite index (`capture_log.db`) queryable via `/api/live-camera/history`
- 738+ unit tests passing

---

## What's Still Stubbed

| Feature | Current State | What's Needed |
|---|---|---|
| **Live cameras (production `main.py` loop)** | `_StubFrameSampler` serves dataset images. The **dashboard** now has one real RTSP camera wired in (`src/dashboard/live_camera.py`) — that part is no longer stubbed, just scoped to the dashboard rather than the production pipeline | Wire real cameras into `FrameAcquisitionModule`/`main.py` for the multi-camera production loop (see TODO below) |
| **Camera-to-location mapping** | Folder names → map locations via JSON config; the live camera path uses a single `location_id` in `config/ip_addresses.json` | Real Ocularis camera names → CameraLocationResolver |
| **Map pin positions** | Removed (inaccurate); location list shown as cards | Visual calibration against site_map.png |
| **Dashboard rules → orchestrator** | Dashboard still uses `dashboard.rules.py` | Wire `HazardRuleOrchestrator` as opt-in (same pattern as DetectionPipeline) |
| **Hazard-event storage** | In-memory HazardStore (20 events, hazard-only, lost on restart) | PostgreSQL backend |
| **Raw frame archive** | Already persistent — `live_camera_captures/` + `capture_log.db` keep every live-camera frame indefinitely, hazard or not | None needed for this scope; a future retention/pruning policy would be an explicit, separate addition |
| **Alerting** | Log-only channel (no real notifications) | Email/SMS/Slack integration |
| **Reduced-class retraining** | Class list defined; no 12-class checkpoint yet | Run training on roboflow data/ with remapped labels |

---

## TODO: Camera Integration

**Priority: HIGH — this is the next major milestone.**

**Update:** the dashboard-scoped slice of this (one real camera, RTSP, feeding
the dashboard's existing `InferenceEngine`) is done — see
[Live Camera Capture & Archive](#live-camera-capture--archive) and
`docs/LIVE_CAMERA_SETUP.md`. What remains below is the larger,
multi-camera **production** integration into `main.py`'s continuous loop,
which is a separate and bigger undertaking (4+ cameras, `FrameAcquisitionModule`,
`HazardRuleOrchestrator` with real Ocularis names).

### Approach

The system is already designed for real cameras — the integration seam is clearly defined:

1. **Replace `_StubFrameSampler`** in `main.py` with a real `FrameSampler` backed by `FrameAcquisitionModule`. This requires:
   - Fixing `src/acquisition/frame_acquisition.py`'s broken import (`from src.models.core import SynchronizedFrameBatch` — the module now exists at `src/models/core.py`, but the import path uses `src.` prefix which requires the workspace root on `sys.path`)
   - Configuring camera sources (RTSP URLs or device IDs) in `FrameAcquisitionModule.initialize_cameras()`
   - Adding camera source configs to `config/hazard_detection.yaml`

2. **Map real Ocularis camera names** in `config/location_rules.yaml`'s `camera_id_to_name` section. Each cam_01/cam_02/etc. in `cameras.sequence` needs its full Ocularis display name (e.g. "A8 - SE PTZ - Block 1F") so the rule engine can resolve its zone type.

3. **Enable the `HazardRuleOrchestrator` in `DetectionPipeline`** by passing it as the `hazard_rule_orchestrator` constructor parameter (already supported — currently `None`, which means the old direct HumanDetector/ContainerAnalyzer path runs). Once real camera names resolve to real zone types, this becomes the production hazard-classification path.

4. **Dashboard integration**: once live cameras feed `main.py`, the dashboard could either:
   - Poll `main.py`'s hazard events via a shared database/queue, or
   - Run its own parallel inference on camera snapshots (current architecture)

   **Done for one camera:** option B is now implemented — `src/dashboard/live_camera.py`'s
   `LiveCaptureService` grabs a 5-frame burst from a real RTSP camera and runs
   the dashboard's existing `InferenceEngine` on each frame, via
   `POST /api/live-camera/capture` and an hourly timer. Extending this to
   multiple cameras would mean generalizing `RTSPCameraConfig`/
   `load_camera_config()` beyond "first camera in the JSON file only."

### Files to modify

| File | Change |
|---|---|
| `config/hazard_detection.yaml` | Add real camera RTSP URLs/device IDs |
| `config/location_rules.yaml` | Map camera IDs to real Ocularis names |
| `src/hazard_detection/main.py` | Fix `_FRAME_SAMPLER_AVAILABLE` path (acquisition import) |
| `src/acquisition/__init__.py` | Fix `from src.acquisition...` import to relative |
| `src/hazard_detection/main.py` | Pass `hazard_rule_orchestrator` + `get_camera_name` to `DetectionPipeline` |

### Prerequisite

The `src/models/core.py` module (reconstructed in this session) must be importable. Currently it requires both `.` and `src` on `PYTHONPATH`. A cleaner fix: change `src/acquisition/__init__.py` and `src/cv/flow_analyzer.py` to use relative imports (`from .frame_acquisition import ...` and `from models.core import ...`) instead of the `src.` prefix pattern. That's a 2-line fix but touches files owned by a different spec.

### Related, already done

See `docs/LIVE_CAMERA_SETUP.md` for the full how-to on the dashboard's
single-camera RTSP integration (setup on a new device, credentials file,
troubleshooting connection issues) — useful reference even though it's a
smaller scope than the multi-camera production wiring described above.

---

## TODO: Production Deployment

**Priority: MEDIUM — needed before going live, not needed for development.**

### Approach

1. **WSGI server**: Replace Flask's dev server with Gunicorn or uWSGI. The dashboard's `HazardStore` is not thread-safe — either add `threading.Lock` around deque access, or move to a real database.

2. **Database**: Replace in-memory `HazardStore` with PostgreSQL. Schema:
   - `hazard_events` table (event_id, hazard_type, camera_id, timestamp, confidence, bbox_json, annotated_image_path, location_json)
   - `rule_audit_log` table (or just keep the jsonl file — it's already append-only and grep-friendly)

3. **HTTPS + auth**: Add TLS termination (nginx reverse proxy) and basic auth or SSO for the dashboard.

4. **Systemd/Windows service**: Make `main.py` run as a service that auto-restarts on crash.

5. **Monitoring**: Health-check endpoint (`/api/status` already exists), plus external uptime monitoring.

---

## TODO: Model Improvements

**Priority: MEDIUM — improves accuracy but system works without it.**

### Approach

1. **Reduced-class retraining** (Task 12 in camera-location-hazard-rules spec):
   - Use `scripts/package_image_data_with_synth.py --reduced_classes` to build a 12-class dataset
   - Run `python scripts/train_yolo.py --data <path-to-12-class-data.yaml> --name hazard_yolo_12class`
   - Evaluate with `python scripts/evaluate_yolo.py --data <same> --checkpoint <new best.pt>`
   - Update `config/hazard_detection.yaml` checkpoint_path once eval confirms acceptable recall

2. **Tighter bounding boxes for people**: The synthetic injection script (`generate_hazard_augmentations.py`) creates boxes from the full RGBA patch rectangle, not the tight alpha mask. Fix `box_to_yolo_polygon()` to compute the bbox from non-transparent pixels after resize/placement.

3. **Small/distant person examples**: Extend `place_patch_random()`'s `scale_range` to include smaller scales (e.g. `(0.04, 0.30)` instead of `(0.12, 0.30)`) so the model learns to detect distant workers.

4. **Per-zone model fine-tuning**: Different zones have different lighting, angles, and expected objects. Consider zone-specific confidence thresholds or fine-tuned checkpoints per zone cluster.

---

## TODO: Dashboard Enhancements

**Priority: LOW — quality-of-life improvements.**

### Approach

1. **Map pin calibration**: Open the dashboard, inspect `site_map.png`'s rendered dimensions, and set accurate `x_pct`/`y_pct` values in `config/dashboard_map.json` for each of the 16 locations. Re-enable the pin overlay code in `terminal_map.js` (currently commented out, ready to uncomment).

2. **Real-time updates via WebSocket**: Replace the 30-second polling in `app.js` with a WebSocket connection. Flask-SocketIO is the simplest option. Push new cycle results to all connected clients immediately.

3. **Zoom-in views**: The `maps/` folder contains additional angle images for specific locations. Add a "click location card → show zoomed view" feature. Each location card in the sidebar could open a modal showing that area's detail image.

4. **Historical timeline**: Add a Chart.js graph showing hazard count over time (hour/day/week). Source: extend `HazardStore` to track timestamps, or read from a database once deployed.

5. **Export/report**: Add a "Download CSV" button that exports all stored hazard events for a time range.

---

## TODO: Alerting System

**Priority: LOW — needed for production, not for development/demo.**

### Approach

1. **Alert channels**: The `AlertDispatcher` in `src/hazard_detection/alert_dispatcher.py` already defines a `Protocol` interface (`AlertChannelAdapter`) with `send()` and `get_name()` methods. Implement real adapters:
   - `EmailAlertChannel` (SMTP)
   - `SlackAlertChannel` (webhook)
   - `SMSAlertChannel` (Twilio or similar)

2. **Rate limiting**: Already implemented (`rate_limit_seconds` in config, default 60s per camera+hazard_type combo). No changes needed.

3. **Severity routing**: Currently all hazards are treated equally. Add severity levels (Critical: human in prohibited zone; Warning: PPE violation; Info: container door open at non-suppressed location) and route different severities to different channels.

4. **Acknowledge/snooze**: Add a dashboard button to acknowledge an alert (suppress for N minutes). Requires persistent storage (database TODO above).

---

## Quick Reference: How to Run

```cmd
REM Dashboard (web UI):
start_webapp.bat
REM or manually:
set PYTHONPATH=.;src
python -m dashboard.app

REM Pipeline (terminal, stub mode):
set PYTHONPATH=.;src
python -m hazard_detection.main --config config/hazard_detection.yaml

REM Tests:
python -m pytest tests/unit/ -v --tb=short

REM 10-minute demo cycle (dashboard):
set DASHBOARD_CYCLE_MINUTES=10
python -m dashboard.app

REM Live camera (dashboard): copy config/ip_addresses_template.json to
REM config/ip_addresses.json, fill in real credentials, then run the
REM dashboard normally — see docs/LIVE_CAMERA_SETUP.md for full details.
copy config\ip_addresses_template.json config\ip_addresses.json
```
