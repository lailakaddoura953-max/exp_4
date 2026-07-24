# Live Camera Setup — Connecting a Real Wisenet Camera to the Dashboard

**Status:** Feature implemented, tested with no camera configured (correctly
degrades to "Not Configured"). Not yet tested against a real camera —
that's the last step on the other device.

This guide covers three things:
1. What was built (new files, changed files, how they talk to each other)
2. How the data flows from the physical camera to the dashboard UI
3. Exactly what to do on the other device to get a real live feed working

---

## Table of Contents

1. [What This Feature Does](#1-what-this-feature-does)
2. [New Files](#2-new-files)
3. [Modified Files](#3-modified-files)
4. [How It All Interacts](#4-how-it-all-interacts)
5. [Setup on the Other Device](#5-setup-on-the-other-device)
6. [Verifying It Works](#6-verifying-it-works)
7. [Configuration Reference](#7-configuration-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What This Feature Does

The dashboard can now pull frames directly from a real Wisenet IP camera
over RTSP, run them through the same YOLO hazard-detection pipeline used
everywhere else in this system, and show the results in the browser — no
dataset images involved.

Two ways a capture happens:

- **On-demand** — click "Capture 5-Frame Burst" in the dashboard's Live
  Camera panel. Good for live demonstrations.
- **Automatic** — a background timer fires the same capture every hour.
  Good for unattended/production use. (Hourly is a hard floor; see
  [Configuration Reference](#7-configuration-reference).)

Either path does the same thing: open the camera's RTSP stream, grab 5
frames in quick succession, run each through the inference engine, log
every step to the terminal, and store the result so the dashboard can
display it.

---

## 2. New Files

### `src/dashboard/live_camera.py`

The core of the feature. Three main pieces:

- **`RTSPCameraConfig`** — a small dataclass holding one camera's connection
  info (IP, port, username, password, RTSP profile) and building the actual
  `rtsp://...` URL from it.
- **`load_camera_config()`** — reads `config/ip_addresses.json` and returns
  an `RTSPCameraConfig` for the first camera listed. Returns `None` (not an
  exception) if the file is missing or malformed, so the dashboard can start
  up fine without a camera configured — it just disables the feature and
  logs a warning.
- **`LiveCameraCapture`** — opens a short-lived `cv2.VideoCapture` connection
  to the RTSP URL, reads 5 frames with a small delay between each, then
  releases the connection. Returns a `BurstCaptureResult` (frames + any
  connection error) either way — callers never get a bare `None` to check.
- **`LiveCaptureService`** — the orchestrator. Combines `LiveCameraCapture`
  with the existing `InferenceEngine` and `annotate()` helper: capture a
  burst, run inference on every frame, log detailed progress to the
  terminal for each frame, optionally store hazard events in the shared
  `HazardStore`, and return one structured result dict.

This module deliberately does **not** reuse `FrameAcquisitionModule` (the
existing 4-camera production module in `src/acquisition/`). That module
hard-requires exactly 4 cameras with IDs 0-3 — the wrong shape for "one live
test camera, burst on demand or hourly." Keeping them separate means the
4-camera module's contract stays untouched for whenever real multi-camera
production wiring happens.

- **`LiveCaptureArchiver`** — saves **every** captured frame to disk,
  hazard or not. This is the key difference from a standard hazard-only
  system: instead of discarding non-hazard frames after inference (which is
  all `HazardStore` retains — hazard events only, in-memory, capped at 20),
  every single frame from every burst gets written to
  `live_camera_captures/<camera_id>/<YYYY-MM-DD>/` as a JPEG + a JSON
  sidecar (detections, hazard flag, timestamp). The archive directory tree
  is created automatically the first time a frame is saved — nothing needs
  to be set up ahead of time, and nothing here ever deletes old files (disk
  usage grows unbounded by design, since the goal is to keep everything).

### `src/dashboard/capture_db.py`

A thin SQLite wrapper (`CaptureDatabase`) providing structured, queryable
access to the same data that's written as JPEG/JSON pairs to
`live_camera_captures/`. SQLite was chosen because it's file-based and
portable — the resulting `capture_log.db` lives right inside
`live_camera_captures/`, so copying that one directory to another machine
brings the whole queryable history with it.

Two tables:
- **`frames`** — one row per captured frame, primary-keyed by `frame_id`
  (the frame's own filename stem, e.g.
  `2026-07-24T14-03-01.120000Z_frame0` — per the explicit choice to use the
  snapshot's own timestamp-derived name as its ID, since it's already
  unique and already ties back to the exact `.jpg`/`.json` pair on disk).
  Includes the same location fields (`loc_facility`, `loc_berth`,
  `loc_crane`, `loc_camera_label`, `loc_landmark`) shown in the dashboard's
  detail view, so a query against this table alone reproduces what the UI
  already displays per capture.
- **`detections`** — one row per YOLO detection within a frame (0 or more
  per `frame_id`), foreign-keyed to `frames`.

Two ways rows get in:
1. **Live** — `LiveCaptureArchiver.save_frame()` calls
   `CaptureDatabase.record_frame()` right after writing each frame's JPEG +
   JSON sidecar.
2. **Auto-generate / backfill** — `sync_archive_to_db()` walks every JSON
   sidecar already on disk under `live_camera_captures/` and (re)populates
   the database from them. This is what makes the database self-healing:
   if `capture_log.db` doesn't exist yet (fresh machine, or copied
   `live_camera_captures/` without its `.db` file), or gets deleted, this
   rebuilds it completely from the JSON sidecars alone — no manual import
   step required. Both paths funnel through the same `record_frame()`
   upsert logic (matched on `frame_id`), so re-running the sync is always
   safe and never creates duplicates.

### `config/ip_addresses_template.json`

The template you copy to create the real (gitignored) config. Contains only
placeholder values (`CHANGE_ME` for the password) — safe to commit, which is
why it's tracked in git while `ip_addresses.json` itself is not.

```json
{
  "cameras": {
    "test_camera_01": {
      "camera_id": "live_cam_01",
      "ip": "192.168.1.100",
      "rtsp_port": 554,
      "username": "admin",
      "password": "CHANGE_ME",
      "profile": "profile2",
      "location_id": 10
    }
  }
}
```

Only the **first** camera entry in the `"cameras"` map is used today
(single live test camera scope). Additional entries are parsed but ignored
until multi-camera support is added.

---

## 3. Modified Files

### `.gitignore`

Two changes:
1. Previously ignored both `config/ip_addresses_template.json` and
   `config/ip_addresses.json`. Now only the real `ip_addresses.json` is
   ignored — the template has no real secrets in it and needs to be
   committed so it's available to copy on any machine (including the other
   device).
2. Added `live_camera_captures/` — the new frame archive directory (see
   `LiveCaptureArchiver` above). Local-only, grows unbounded, never
   committed — same treatment as `roboflow data/` and the other large local
   data directories already in this file.

### `src/dashboard/app.py`

Three additions, all isolated so they don't touch the existing dataset
auto-cycle logic:

1. **Startup wiring** — loads `config/ip_addresses.json` via
   `load_camera_config()`. If it's found *and* the YOLO model loaded
   successfully, builds a `LiveCaptureService`. Otherwise logs a warning and
   leaves live camera capture disabled (the rest of the dashboard still
   works normally).
2. **Hourly timer thread** (`_live_camera_timer_loop`) — a daemon thread
   that sleeps for the configured interval (60 minutes minimum, enforced),
   then triggers a burst automatically. Runs independently of the on-demand
   endpoint.
3. **Two new routes:**
   - `POST /api/live-camera/capture` — triggers an on-demand burst
     synchronously, returns the full result. Returns HTTP 409 if a burst is
     already running (either from this endpoint or the hourly timer), so
     two RTSP connections never open to the same camera at once.
   - `GET /api/live-camera/status` — returns whether the feature is
     configured, whether a capture is in progress, and the most recent
     burst result (from either trigger source). The dashboard polls this
     every 15 seconds.

### `src/dashboard/static/index.html`

Added a "Live Camera" section between the Stats cards and the Terminal Map:
a status badge, a "Capture 5-Frame Burst" button, an error message area,
and a grid where captured frame cards get rendered.

### `src/dashboard/static/app.js`

Added the client-side logic:
- `pollLiveCameraStatus()` — polls `/api/live-camera/status` every 15s,
  updates the badge, and renders a new result if the timestamp changed
  (picks up both on-demand and hourly-timer results transparently).
- `triggerLiveCameraCapture()` — handles the button click, POSTs to
  `/api/live-camera/capture`, renders the result (or the error, e.g. "A
  capture is already in progress").
- `renderLiveCameraResult()` — builds one card per captured frame, each
  showing the annotated image and its detection list, matching the visual
  style of the existing "Live Inference" (auto-cycle) section.

### `src/dashboard/static/styles.css`

Added styling for the new section (`.live-camera-*` classes) — a card grid
for the 5 frames, a hazard highlight border when a frame contains a hazard,
and the header/badge/button layout. Reuses existing CSS variables
(`--spacing-*`, `--color-*`, `.detection-row` styles) rather than
introducing new colors.

---

## 4. How It All Interacts

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (dashboard UI)                                          │
│  index.html — "Live Camera" section + button                     │
│  app.js — triggerLiveCameraCapture() / pollLiveCameraStatus()     │
└───────────────────────────┬───────────────────────────────────────┘
                            │ POST /api/live-camera/capture
                            │ GET  /api/live-camera/status
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Flask backend — src/dashboard/app.py                            │
│  - loads config/ip_addresses.json at startup (load_camera_config)│
│  - builds LiveCaptureService once (if config + model both OK)    │
│  - hourly timer thread calls the same service                    │
│  - both routes call _run_live_camera_burst_locked()               │
└───────────────────────────┬───────────────────────────────────────┘
                            │ .run_burst()
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/dashboard/live_camera.py — LiveCaptureService                │
│  1. LiveCameraCapture.capture_burst()                             │
│       opens rtsp://user:pass@ip:554/profile2/media.smp             │
│       reads 5 frames via cv2.VideoCapture, ~150ms apart            │
│       releases the connection                                     │
│  2. For each frame:                                               │
│       InferenceEngine.run(frame, camera_id)  ← same engine used    │
│                                                 by dataset auto-cycle│
│       annotate(frame, results)               ← same annotator     │
│       logs every step to the terminal                             │
│       stores hazard events in the shared HazardStore (hazard-only) │
│       LiveCaptureArchiver.save_frame(...)    ← EVERY frame,        │
│                                                 hazard or not       │
│  3. Returns one result dict: frames + detections + hazards_found  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              live_camera_captures/<camera_id>/<date>/
                 <timestamp>_frame<N>.jpg   (raw frame, kept forever)
                 <timestamp>_frame<N>.json  (detections + hazard flag)
                            │
                            ▼
                 Physical camera (XN-C9303RW)
                 over the wired RTSP connection
                 set up in the earlier networking steps
```

Key point: **this feature reuses the existing inference stack end to end.**
`InferenceEngine`, `annotate()`, and `HazardStore` are exactly the same
objects the dataset auto-cycle and manual upload features already use. The
only new code is "how to get a frame from a real camera" — everything
downstream of that was already built and tested.

### Data retention: keep everything, not just hazards

This is the one place this system deliberately differs from a "standard"
hazard-detection setup. Most such systems only retain flagged
hazard events and throw away everything else — that's what `HazardStore`
alone does here too (in-memory, 20-event cap, hazard-only, wiped on
restart).

`LiveCaptureArchiver` sits alongside `HazardStore`, not in place of it, and
saves **every** frame from every burst to disk — hazard or not:

```
live_camera_captures/
└── live_cam_01/                          ← camera_id
    └── 2026-07-24/                       ← capture date (UTC)
        ├── 2026-07-24T14-03-01.120000Z_frame0.jpg   ← raw frame
        ├── 2026-07-24T14-03-01.120000Z_frame0.json  ← detections + hazard flag
        ├── 2026-07-24T14-03-01.270000Z_frame1.jpg
        ├── 2026-07-24T14-03-01.270000Z_frame1.json
        └── ... (frames 2-4)
```

The directory tree is created automatically the first time a frame is
saved — you never need to manually `mkdir` anything on either device. If
`live_camera_captures/` doesn't exist yet when the first burst runs,
`LiveCaptureArchiver._ensure_dir()` creates the camera and date
subdirectories as needed.

Nothing in this path ever deletes a file. Disk usage grows without bound as
long as the hourly timer and/or the demo button keep running — that's
intentional, not a bug, but it does mean disk space should be monitored
over time on whichever device is running captures long-term. There's no
retention/pruning logic today; if you need one later (e.g. "delete anything
older than 90 days"), that would be a separate, explicit addition — it's
not something this implementation does silently.

### Querying the archive: capture_log.db

Alongside the JPEG/JSON files, every frame is also mirrored into a SQLite
database at `live_camera_captures/capture_log.db`. This is the "organized
into accessible tables" layer on top of the raw files — the same data,
queryable with plain SQL instead of having to open and parse every JSON
sidecar individually.

```
live_camera_captures/
├── capture_log.db                    ← SQLite file, portable with this folder
└── live_cam_01/
    └── 2026-07-24/
        ├── ...frame0.jpg / .json
        └── ...
```

Two tables — `frames` (one row per captured frame, including location
fields) and `detections` (one row per YOLO detection, linked to its
frame). Every frame row's primary key is its own filename stem, so you can
always cross-reference a database row back to its exact `.jpg`/`.json`
pair on disk.

Two dashboard endpoints expose this without needing a SQL client:

- `GET /api/live-camera/status` — now also returns `archived_frame_count`
  (total rows in the `frames` table), so you can see the archive growing
  over time from the same status poll the UI already uses.
- `GET /api/live-camera/history?limit=20&hazard_only=false` — returns
  recent frame rows directly from `capture_log.db`, including the location
  fields, without touching the raw JSON files at all.

If you ever need to inspect it directly (e.g. after copying
`live_camera_captures/` to another machine, or just to sanity-check the
data), any SQLite browser or the `sqlite3` CLI works:

```bash
sqlite3 live_camera_captures/capture_log.db "SELECT frame_id, camera_id, is_hazard_frame, loc_berth FROM frames ORDER BY capture_timestamp DESC LIMIT 10;"
```

If `capture_log.db` is ever missing or deleted (e.g. you only copied the
JPEG/JSON files over, or want to force a clean rebuild), it regenerates
itself automatically the next time a frame is captured — or immediately, on
demand, via:

```python
from dashboard.live_camera import sync_archive_to_db
sync_archive_to_db()  # rebuilds capture_log.db from every JSON sidecar on disk
```

---

## 5. Setup on the Other Device

Do this on whichever machine will have the camera physically wired to it
(the one you did the direct Ethernet + PoE injector setup on).

### Step 1 — Pull the latest code

```bash
git pull
```

This brings in `src/dashboard/live_camera.py`,
`config/ip_addresses_template.json`, and the updated `app.py` /
`index.html` / `app.js` / `styles.css`.

### Step 2 — Create the real credentials file

```bash
# Windows (PowerShell)
Copy-Item config\ip_addresses_template.json config\ip_addresses.json
```

Edit `config/ip_addresses.json` and fill in the real values:

```json
{
  "cameras": {
    "test_camera_01": {
      "camera_id": "live_cam_01",
      "ip": "192.168.1.100",
      "rtsp_port": 554,
      "username": "admin",
      "password": "Camera2026!Test",
      "profile": "profile2",
      "location_id": 10
    }
  }
}
```

- **`ip`** — the camera's IP address on whatever network the dashboard
  machine and camera share (the static-IP direct-connection setup from the
  networking steps, or the camera's real IP once it's on production
  network/VLAN).
- **`username` / `password`** — whatever you set in WiseNet Device Manager
  when you got the camera connected.
- **`profile`** — `profile2` is the lower-resolution sub-stream,
  recommended for frequent bursts. Use `profile1` for the full main stream
  if you need higher image quality and don't mind larger frames.
- **`location_id`** — optional; should match one of the location IDs in
  `config/dashboard_map.json` if you want this camera associated with a
  specific yard location on the map later. Not required for the feature to
  work.

**This file is gitignored — it will never get committed, and it won't sync
via `git pull`/`git push` between devices.** Each device that connects to a
real camera needs its own copy with its own credentials filled in.

### Step 3 — Restart the dashboard

```bash
set PYTHONPATH=.;src
python -m dashboard.app
```

or use `start_webapp.bat` as usual. Check the startup log for:

```
Live camera capture ready — camera_id='live_cam_01', auto_capture=True, interval_minutes=60
```

If instead you see:

```
Live camera capture disabled — no camera config found. ...
```

`config/ip_addresses.json` wasn't found or couldn't be parsed — double
check it's at `config/ip_addresses.json` (not `.json.txt` or similar) and
is valid JSON.

### Step 4 — Test the button

Open the dashboard (`http://localhost:5000`), find the "Live Camera"
section, and click "Capture 5-Frame Burst". Watch the terminal — you should
see a full log trace: RTSP connection opening, each of the 5 frames being
captured with its timestamp and resolution, then each frame's inference
results (detections, hazards) as they happen. The browser will show the 5
annotated frames as cards once the burst completes.

---

## 6. Verifying It Works

Checklist for the other device:

- [ ] `config/ip_addresses.json` exists and has real (not `CHANGE_ME`) values
- [ ] Dashboard startup log shows `Live camera capture ready`
- [ ] Live Camera badge in the browser shows "Ready" (not "Not Configured")
- [ ] Clicking "Capture 5-Frame Burst" produces 5 frame cards with images
- [ ] Terminal shows the full capture log (connection → 5x frame capture → 5x inference)
- [ ] No hazards found is fine — that just means the current view has nothing hazardous in it; the important thing is that frames arrive and get annotated
- [ ] `live_camera_captures/<camera_id>/<today's date>/` now exists on disk with 5 new `.jpg` + `.json` file pairs, even if none of the frames were hazards
- [ ] `live_camera_captures/capture_log.db` now exists, and `GET /api/live-camera/status`'s `archived_frame_count` increased by 5

If the button returns an error instead:

- **"Could not open RTSP stream..."** — camera is offline, wrong IP, wrong
  port, or a network/isolation issue (see the earlier networking
  troubleshooting steps — same root causes apply here).
- **"Connected to the camera but every frame read failed..."** — the RTSP
  session opened but no frames were readable. Try switching `profile` from
  `profile2` to `profile1` (or vice versa) in `ip_addresses.json` — some
  firmware versions are picky about which sub-stream responds over RTSP
  first.
- **HTTP 409 "A live camera capture is already in progress"** — a burst
  (on-demand or hourly) is already running against this camera; wait a few
  seconds and try again. Bursts typically take a few seconds total (5
  frames + 5 inference passes).

---

## 7. Configuration Reference

Environment variables (all optional, same pattern as the existing
`DASHBOARD_*` variables):

| Variable | Default | Purpose |
|---|---|---|
| `DASHBOARD_LIVE_CAMERA_AUTO` | `true` | Set to `false` to disable the hourly timer entirely. The on-demand button still works either way. |
| `DASHBOARD_LIVE_CAMERA_INTERVAL_MINUTES` | `60` | Change the automatic capture cadence. **Values below 60 are clamped up to 60** — this floor is intentional (explicit requirement: never run the automatic timer more often than hourly). |

Other constants (in `src/dashboard/live_camera.py`, not env-configurable —
change the source if you need different values):

| Constant | Value | Purpose |
|---|---|---|
| `BURST_SIZE` | `5` | Frames captured per burst |
| `INTER_FRAME_DELAY_SECONDS` | `0.15` | Delay between frame reads within a burst |
| `CONNECT_TIMEOUT_SECONDS` | `8.0` | How long to wait for the RTSP connection to open before giving up |

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Badge stuck on "Not Configured" | `config/ip_addresses.json` missing or invalid JSON | Copy from the template again, validate JSON syntax (e.g. `python -m json.tool config/ip_addresses.json`) |
| Badge shows "Ready" but capture always fails with connection error | Camera offline, wrong IP/port, or network isolation blocking the dashboard machine from reaching the camera | Re-check the direct wired connection / static IP setup from the networking steps; confirm the IP in `ip_addresses.json` matches what WiseNet Device Manager shows |
| Capture succeeds but frames are blank/black | Wrong RTSP profile or unsupported codec for that profile | Try switching `profile` between `profile1` and `profile2` |
| Terminal shows `InferenceEngine.run failed` for each frame | YOLO model not loaded (check dashboard startup log for `model_loaded=True`) | Fix the underlying checkpoint issue first — same as with dataset-image inference failures, unrelated to the camera itself |
| Two clicks in a row both fail with HTTP 409 | Bursts take a few seconds; UI button should be disabled automatically while one is running via `capture_in_progress` polling | If the button seems stuck disabled, refresh the page — `/api/live-camera/status` will report the true current state |
| Credentials work in a browser tab but not here | Password has special characters that need URL-encoding | Already handled — `RTSPCameraConfig.build_url()` URL-encodes username/password automatically. If it still fails, try a password without `@` or `:` characters as a sanity check |

---

## Where This Fits in the Bigger Picture

This is the first real (non-dataset) camera integration in the system.
`docs/SYSTEM_ARCHITECTURE.md`'s "TODO: Camera Integration" section describes
the longer-term production path (multi-camera `FrameAcquisitionModule`,
`main.py`'s continuous pipeline, `HazardRuleOrchestrator` with real Ocularis
camera names). This feature is a smaller, dashboard-scoped slice of that
same goal — proving the camera-to-inference path works end-to-end with one
real camera before investing in the full multi-camera production wiring.
