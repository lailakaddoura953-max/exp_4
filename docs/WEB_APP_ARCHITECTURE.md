# Web App Architecture & Design

## Overview

A two-server web application for viewing strad carrier monitoring results. The frontend renders a dashboard with camera status cards, live monitoring feed, and an inference test tool. The backend serves API endpoints for real data from the monitoring system.

---

## System Layout

```
Browser (localhost:8000)
    │
    │  Static HTML/CSS/JS served by Python HTTP server
    │
    ├── index.html          Main page structure
    ├── script.js           All frontend logic (API calls, DOM updates, modals)
    └── styles.css          Visual styling
    
    │
    │  API calls to Flask backend
    │
    ▼
Flask Backend (localhost:5000)
    │
    ├── /                            Health check
    ├── /api/inference               Upload image → classify (real or mock)
    ├── /api/model/status            Classifier info
    ├── /api/strads/recent           DB query for recent results
    ├── /api/strads/stats            Classification counts
    ├── /api/snapshot/<id>           Serve snapshot by strad ID (from DB path)
    ├── /api/live/images             Real screenshots or augmented dataset
    ├── /api/live/image/<path>       Serve image file by relative path
    ├── /api/live/active-camera-count   Total strads minus exclusions
    └── /api/live/strad-details/<id>    IP, history, critical status
    
    │
    │  Reads from:
    │
    ├── system_config.json              Configuration
    ├── config/ip_addresses.json        Strad → IP mapping
    ├── data/monitoring_state.json      Local state (check history, results, exclusions)
    ├── permanent_snapshots/            Critical photos (long-term storage)
    └── SCFootage/ or SCFootage_augmented/   Training dataset images (fallback)
```

---

## Frontend Components

### Page Sections (top to bottom)

| Section | Purpose |
|---------|---------|
| **Header** | Title, connection status indicator |
| **Stats Row** | Active strads available, FPS, features/camera |
| **Kanban Board** | Demo cards: Normal, Low Priority, Critical columns |
| **Impact Timeline** | Full scenario playback button |
| **Upload/Inference** | Drag-drop image → classify via backend |
| **Results Panel** | Probability bar, severity badge, 6-DOF pose, uncertainty |
| **Live Monitoring Feed** | Real/augmented image grid with filters |
| **Video Modal** | Scenario demo video playback |
| **Strad Detail Modal** | Per-strad info: IP, history, critical status |

### Key JavaScript Functions

```
Frontend (script.js)
├── Backend Integration
│   ├── checkBackendConnection()      Check if backend is reachable
│   ├── loadRecentStrads()            Fetch recent classifications from DB
│   └── updateKanbanWithRealData()    Populate kanban with live data
│
├── Demo Playback
│   ├── viewScenario(id)              Open video modal for scenario
│   ├── showDetails(id)               Show scenario details in modal
│   ├── closeModal()                  Close video/detail modal
│   └── loadGifFallback(file)         Fall back to GIF if MP4 fails
│
├── Inference
│   ├── runInference()                Upload image → POST /api/inference
│   ├── displayInferenceResults()     Render classification output
│   ├── downloadResults()             Export results as JSON
│   └── resetInference()              Clear and start over
│
├── Live Monitoring Feed
│   ├── loadActiveCameraCount()       GET /api/live/active-camera-count
│   ├── refreshLiveImages()           GET /api/live/images + render grid
│   ├── showStradDetails(id)          GET /api/live/strad-details/<id>
│   ├── copyIpAddress(id)             Copy camera IP to clipboard
│   └── closeStradDetailModal()       Close detail modal
│
└── Upload Handling
    ├── handleCompositeImageSelect()  File input change
    ├── handleDrop()                  Drag-and-drop
    ├── displayImagePreview()         Show thumbnail
    └── removeImage()                 Clear selection
```

---

## Backend Architecture

### File: `docs/backend/app.py`

```
Flask App
├── Startup Initialization
│   ├── Load system_config.json (via ConfigurationManager)
│   ├── Initialize DatabaseInterface (SQL Server via DSN)
│   ├── Initialize DL Classifier (SimpleClassifierWrapper or DLClassifierWrapper)
│   └── Print status (connected/disconnected for each component)
│
├── Original Endpoints (unchanged)
│   ├── GET /                     Health + connection status
│   ├── POST /api/inference       Multi-camera or single-image classification
│   ├── GET /api/model/status     Classifier type and readiness
│   ├── GET /api/strads/recent    Recent results from SQL Server
│   ├── GET /api/strads/stats     Count by severity from SQL Server
│   └── GET /api/snapshot/<id>    Serve critical snapshot from DB path
│
└── Live Monitoring Endpoints (new)
    ├── GET /api/live/images              Image list (live + augmented fallback)
    ├── GET /api/live/image/<path>        Serve image file
    ├── GET /api/live/active-camera-count Strad pool count
    └── GET /api/live/strad-details/<id>  Per-strad monitoring details
```

### Data Sources for Live Endpoints

```
/api/live/active-camera-count
    ├── IPAddressLoader (config/ip_addresses.json) → total count
    └── LocalStateStore (data/monitoring_state.json) → critical exclusions

/api/live/images
    ├── Priority 1: LocalStateStore → results with snapshot_path
    ├── Priority 2: permanent_snapshots/ directory scan
    └── Priority 3: SCFootage/ or SCFootage_augmented/ (augmented dataset)

/api/live/image/<path>
    ├── Try: permanent_snapshots/<path>
    ├── Try: temp_snapshots/<path>
    ├── Try: SCFootage/<path>
    └── Try: SCFootage_augmented/<path>

/api/live/strad-details/<id>
    ├── IPAddressLoader → IP address
    └── LocalStateStore → check history, classifications, critical info
```

---

## Image Source: SCFootage Structure

```
SCFootage/
├── misaligned_critical/
│   ├── strad_001/
│   │   └── image.png
│   ├── strad_042/
│   │   └── image.png
│   └── ...
├── misaligned_moderate/
│   ├── strad_xxx/
│   │   └── image.png
│   └── ...
└── misaligned_none/
    ├── strad_xxx/
    │   └── image.png
    └── ...
```

- Classification derived from folder name (`misaligned_critical` → `critical`)
- Strad ID extracted from subfolder name (`strad_042` → `SC042`)
- Images served using relative path as unique identifier
- One image per strad folder displayed in the gallery

---

## Frontend ↔ Backend Communication

```
Page Load:
  1. GET http://localhost:5000/           → Check backend alive
  2. GET /api/live/active-camera-count   → Update stat card
  3. GET /api/live/images?source=auto    → Populate image grid
  4. GET /api/strads/recent              → (Optional) kanban real data

User clicks "Details" on SC042:
  1. GET /api/live/strad-details/SC042   → Open modal with IP, history

User clicks "Copy IP" on SC042:
  1. GET /api/live/strad-details/SC042   → Read IP → clipboard

User changes filter dropdown:
  1. GET /api/live/images?source=X&severity=Y → Refresh grid

User uploads image for inference:
  1. POST /api/inference (multipart form) → Classify
  2. Display results in Results Panel

Image card renders:
  1. <img src="/api/live/image/misaligned_critical/strad_042/image.png">
     → Backend resolves path → serves file
```

---

## How to Run

```bash
# Terminal 1: Backend (port 5000)
cd docs/backend
python app.py

# Terminal 2: Frontend (port 8000)
cd <project_root>
python start_frontend_server.py

# Or use the batch file:
start_web_app.bat
```

Open: **http://localhost:8000**

---

## Configuration Dependencies

| Config Source | Used By | Purpose |
|---------------|---------|---------|
| `system_config.json` | Backend startup | DB connection, model path, snapshot paths |
| `config/ip_addresses.json` | `/api/live/active-camera-count`, `/api/live/strad-details` | Total strad count, IP lookup |
| `data/monitoring_state.json` | All `/api/live/*` endpoints | Check history, results, exclusions |
| `SCFootage/` or `SCFootage_augmented/` | `/api/live/images` (fallback) | Augmented dataset images |
| `permanent_snapshots/` | `/api/live/images`, `/api/live/image/` | Critical photos from cycles |

---

## Graceful Degradation

| Condition | Frontend Behavior |
|-----------|-------------------|
| Backend not running | Page loads, shows "Disconnected", demo cards still work |
| DB unavailable | Backend starts, live endpoints use local files only |
| No monitoring_state.json | Active count shows total (no exclusions), no history |
| No SCFootage folder | "No images available" message in grid |
| No permanent_snapshots | Falls back to augmented dataset |
| Classifier not loaded | Inference returns mock classification |

---

## Styling Conventions

- White card backgrounds with subtle shadow (`box-shadow: 0 1px 3px`)
- Severity color coding: 🔴 `#ef4444` / 🟡 `#f59e0b` / 🟢 `#22c55e`
- Source badges: green for "live", indigo for "augmented"
- Responsive grid: `auto-fill, minmax(280px, 1fr)`
- Card hover: lift effect (`translateY(-2px)`)
- Modal: centered overlay with backdrop

---

## Future Improvements

| Feature | Effort | Notes |
|---------|--------|-------|
| Auto-refresh live feed every N seconds | Low | Add `setInterval(refreshLiveImages, 30000)` |
| Show live feed during active cycle | Medium | WebSocket or polling during orchestrator run |
| Historical timeline view | Medium | Chart.js graph of classifications over time |
| Per-strad image history gallery | Medium | Show all photos for one strad across cycles |
| Export monitoring report | Low | Download all results as CSV/PDF |
| Dark mode toggle | Low | CSS variables already partially support it |
