/**
 * app.js — Yard Hazard Inference Dashboard
 *
 * Tasks implemented:
 *   11.1  Page-load initialisation: status poll, recent hazards, categorize_hazard_type()
 *   11.3  Detection card rendering: renderHazardCards(), "View Details" modal, map highlight
 *   11.4  Run Inference section: upload/drop zone, preview, submit, results, "Use Dataset Image"
 */

'use strict';

/* ============================================================
   UTILITY — hazard type categorisation
   ============================================================ */

/**
 * Map a hazard_reason string to one of three categories used for card
 * border colour.  This function is total (covers every possible string
 * input) and deterministic (same input always yields same output).
 *
 * @param {string} hazardReason  e.g. "misaligned_container", "ppe_violation"
 * @returns {"container"|"human"|"other"}
 */
function categorize_hazard_type(hazardReason) {
  const CONTAINER_REASONS = new Set([
    'misaligned_container',
    'water_drop_container',
    'open_container_unsecured',
    'picked_no_crane',
    'picked_person_below_crane',
    'flipped_container',
  ]);

  const HUMAN_REASONS = new Set([
    'ppe_violation',
    'human_below_crane',
    'human_detected_stub',
  ]);

  if (CONTAINER_REASONS.has(hazardReason)) return 'container';
  if (HUMAN_REASONS.has(hazardReason))    return 'human';
  return 'other';
}

/**
 * Convert a snake_case hazard reason to a human-readable Title Case string.
 * e.g. "misaligned_container" → "Misaligned Container"
 *
 * @param {string} reason
 * @returns {string}
 */
function formatHazardType(reason) {
  if (!reason) return 'No Hazard';
  return reason
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/* ============================================================
   UTILITY — safe wrappers for terminal_map.js exports
   (terminal_map.js may not be loaded in all test/dev scenarios)
   ============================================================ */

function safeHighlightCamera(cameraId) {
  if (window.terminalMap && typeof window.terminalMap.highlightCamera === 'function') {
    window.terminalMap.highlightCamera(cameraId);
  }
}

function safeClearHighlight() {
  if (window.terminalMap && typeof window.terminalMap.clearHighlight === 'function') {
    window.terminalMap.clearHighlight();
  }
}

/* ============================================================
   UTILITY — location context from CAMERA_PINS or camera_id pattern
   ============================================================ */

/**
 * Derive a LocationContext-like object from a camera_id.
 * Uses CAMERA_PINS from terminal_map.js if available; otherwise falls back
 * to the same stub mapping defined in models.py LocationContext.from_camera_id().
 *
 * @param {string} cameraId
 * @returns {{ facility: string, berth: string, crane: string, camera: string }}
 */
function getLocationContext(cameraId) {
  // Prefer live CAMERA_PINS data from terminal_map.js
  if (typeof window.CAMERA_PINS !== 'undefined' && Array.isArray(window.CAMERA_PINS)) {
    const pin = window.CAMERA_PINS.find(p => p.id === cameraId);
    if (pin) {
      return {
        facility: 'Railyard',
        berth:    pin.berth  || '',
        crane:    pin.crane  || '',
        camera:   pin.label  || cameraId,
      };
    }
  }

  // Fallback: mirror the stub mapping from LocationContext.from_camera_id()
  const STUB_MAP = {
    'cam_stub_01': { berth: 'Berth 403', crane: 'Crane 01',  camera: 'Camera 01'  },
    'cam_stub_02': { berth: 'Berth 403', crane: 'Crane 02',  camera: 'Camera 02'  },
    'cam_stub_03': { berth: 'Berth 403', crane: 'Crane 03',  camera: 'Camera 03'  },
    'cam_stub_04': { berth: 'Berth 403', crane: 'Crane 04',  camera: 'Camera 04'  },
    'cam_stub_05': { berth: 'Berth 405', crane: 'Crane 05',  camera: 'Camera 05'  },
    'cam_stub_06': { berth: 'Berth 405', crane: 'Crane 06',  camera: 'Camera 06'  },
    'cam_stub_07': { berth: 'Berth 405', crane: 'Crane 07',  camera: 'Camera 07'  },
    'cam_stub_08': { berth: 'Berth 405', crane: 'Crane 08',  camera: 'Camera 08'  },
    'cam_stub_09': { berth: 'Berth 405', crane: 'Crane 09',  camera: 'Camera 09'  },
    'cam_stub_10': { berth: 'Berth 406', crane: 'Crane 10',  camera: 'Camera 10'  },
    'cam_stub_11': { berth: 'Berth 406', crane: 'Crane 11',  camera: 'Camera 11'  },
    'cam_stub_12': { berth: 'Berth 406', crane: 'Crane 12',  camera: 'Camera 12'  },
    'cam_stub_13': { berth: 'Berth 406', crane: 'Crane 13',  camera: 'Camera 13'  },
    'cam_stub_14': { berth: 'Berth 406', crane: 'Crane 14',  camera: 'Camera 14'  },
    'cam_stub_15': { berth: '',          crane: '',           camera: 'Camera 15'  },
  };

  const entry = STUB_MAP[cameraId];
  if (entry) {
    return { facility: 'Railyard', ...entry };
  }

  // Unrecognised camera_id — best-effort fallback
  return { facility: 'Railyard', berth: '', crane: '', camera: cameraId };
}

/**
 * Build a location string from a LocationContext, skipping empty fields.
 * e.g. "📍 Railyard · Berth 403 · Crane 01 · Camera 01"
 *
 * @param {object} loc
 * @returns {string}
 */
function formatLocation(loc) {
  const parts = [loc.facility, loc.berth, loc.crane, loc.camera]
    .filter(v => v && v.trim() !== '');
  return parts.length ? parts.join(' · ') : '';
}

/* ============================================================
   11.1 — PAGE LOAD INITIALISATION
   ============================================================ */

/**
 * Poll GET /api/status and update header badge + stat cards.
 */
async function initStatus() {
  const badge = document.getElementById('connection-badge');
  const statHazardCount   = document.getElementById('stat-hazard-count');
  const statModelStatus   = document.getElementById('stat-model-status');
  const statActiveCameras = document.getElementById('stat-active-cameras');

  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    // Connection badge
    badge.textContent = 'Connected';
    badge.classList.remove('disconnected');
    badge.classList.add('connected');

    // Stat cards
    statHazardCount.textContent   = data.hazard_count  ?? '--';
    statModelStatus.textContent   = data.model_loaded  ? 'Loaded' : 'Not Loaded';
    statActiveCameras.textContent = data.active_cameras ?? (data.camera_id ? '1' : '0');
  } catch (err) {
    badge.textContent = 'Disconnected';
    badge.classList.remove('connected');
    badge.classList.add('disconnected');
    console.warn('[app.js] GET /api/status failed:', err);
  }
}

/**
 * Fetch GET /api/live/images (dataset fallback if store empty) and render cards.
 * Mirrors exp_2's live monitoring feed pattern.
 */
async function initRecentHazards() {
  try {
    const response = await fetch('/api/live/images?limit=6&source=auto');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.json();
    renderHazardCards(body.data || []);
  } catch (err) {
    console.warn('[app.js] GET /api/live/images failed:', err);
    // fallback to hazards/recent
    try {
      const r2 = await fetch('/api/hazards/recent');
      if (r2.ok) {
        const events = await r2.json();
        renderHazardCards(events);
      } else {
        renderHazardCards([]);
      }
    } catch (_) {
      renderHazardCards([]);
    }
  }
}

/* ============================================================
   11.3 — DETECTION CARD RENDERING
   ============================================================ */

/**
 * Render hazard event cards into #hazard-cards-grid.
 * Shows #no-recent-msg when the array is empty.
 *
 * @param {Array<object>} events  Array of HazardEvent objects from the API
 */
function renderHazardCards(events) {
  const grid   = document.getElementById('hazard-cards-grid');
  const noMsg  = document.getElementById('no-recent-msg');

  // Clear previous cards (but keep #no-recent-msg in the DOM)
  Array.from(grid.children).forEach(child => {
    if (child.id !== 'no-recent-msg') child.remove();
  });

  if (!events || events.length === 0) {
    noMsg.style.display = '';
    return;
  }

  noMsg.style.display = 'none';

  events.forEach(event => {
    const card = buildHazardCard(event);
    grid.appendChild(card);
  });
}

/**
 * Build a single .detection-card element from a HazardEvent.
 *
 * @param {object} event  HazardEvent from the API
 * @returns {HTMLElement}
 */
function buildHazardCard(event) {
  const isHazard    = event.is_hazard === true;
  const hazardType  = event.hazard_type || '';
  const category    = isHazard ? categorize_hazard_type(hazardType) : 'no-hazard';
  const borderClass = `border-${category}`;

  const loc      = event.location ? event.location : getLocationContext(event.camera_id || '');
  const locStr   = formatLocation(loc);
  const timeStr  = event.timestamp
    ? new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '--';
  const confStr  = (event.confidence != null)
    ? `${Math.round(event.confidence * 100)}%`
    : '--';
  const typeStr  = formatHazardType(hazardType);

  const card = document.createElement('article');
  card.className = `detection-card ${borderClass}`;
  card.setAttribute('role', 'button');
  card.setAttribute('tabindex', '0');
  card.setAttribute('aria-label', `${typeStr} — ${timeStr}`);

  // Annotated image (only when annotated_image is non-null)
  let imgHtml = '';
  if (event.annotated_image) {
    imgHtml = `<img
      class="detection-card-image"
      src="data:image/png;base64,${event.annotated_image}"
      alt="Annotated image for ${typeStr}"
      loading="lazy"
    />`;
  }

  card.innerHTML = `
    ${imgHtml}
    <div class="detection-card-meta">
      <div class="detection-card-type">${typeStr}</div>
      ${locStr ? `<div class="detection-card-location">${locStr}</div>` : ''}
      <div class="detection-card-time">${timeStr}</div>
      <div class="detection-card-confidence">Confidence: ${confStr}</div>
    </div>
    <div class="detection-card-footer">
      <button class="btn btn-secondary view-details-btn" type="button">View Details</button>
    </div>
  `;

  // "View Details" button and card body both open the modal
  const openModal = (e) => {
    e.stopPropagation();
    openDetailsModal(event);
    safeHighlightCamera(event.camera_id);
  };

  card.querySelector('.view-details-btn').addEventListener('click', openModal);
  card.addEventListener('click', openModal);
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openModal(e);
    }
  });

  return card;
}

/* ============================================================
   11.3 — VIEW DETAILS MODAL
   ============================================================ */

/**
 * Populate and show the #details-modal for a given HazardEvent.
 *
 * @param {object} event  HazardEvent from the API
 */
function openDetailsModal(event) {
  const modal       = document.getElementById('details-modal');
  const modalImg    = document.getElementById('modal-annotated-image');
  const tableBody   = document.getElementById('modal-event-table-body');

  // Annotated image
  if (event.annotated_image) {
    modalImg.src   = `data:image/png;base64,${event.annotated_image}`;
    modalImg.style.display = '';
  } else {
    modalImg.src   = '';
    modalImg.style.display = 'none';
  }

  // Event metadata table — one row per field
  const loc     = event.location ? event.location : getLocationContext(event.camera_id || '');
  const bboxStr = event.bbox
    ? `x=${event.bbox.x_center?.toFixed(3)}, y=${event.bbox.y_center?.toFixed(3)}, w=${event.bbox.width?.toFixed(3)}, h=${event.bbox.height?.toFixed(3)}`
    : '—';

  const rows = [
    ['Event ID',    event.event_id    || '—'],
    ['Hazard Type', formatHazardType(event.hazard_type || '')],
    ['Camera ID',   event.camera_id   || '—'],
    ['Timestamp',   event.timestamp
                      ? new Date(event.timestamp).toLocaleString()
                      : '—'],
    ['Confidence',  (event.confidence != null) ? `${Math.round(event.confidence * 100)}%` : '—'],
    ['Bounding Box', bboxStr],
    ['Facility',    loc.facility || '—'],
    ['Berth',       loc.berth    || '—'],
    ['Crane',       loc.crane    || '—'],
    ['Camera',      loc.camera   || '—'],
  ];

  tableBody.innerHTML = rows
    .map(([key, val]) => `
      <tr>
        <th scope="row">${key}</th>
        <td>${val}</td>
      </tr>`)
    .join('');

  // Show modal
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  // Focus the close button for accessibility
  const closeBtn = document.getElementById('modal-close-btn');
  closeBtn.focus();
}

/**
 * Close the #details-modal and clear the map highlight.
 */
function closeDetailsModal() {
  const modal = document.getElementById('details-modal');
  modal.style.display = 'none';
  document.body.style.overflow = '';
  safeClearHighlight();
}

/* ============================================================
   11.4 — RUN INFERENCE SECTION
   ============================================================ */

/** The currently selected File object (upload or test-image). */
let selectedFile = null;

/**
 * Load a File into the upload preview area.
 * Called both from file-picker/drop and "Use Dataset Image".
 *
 * @param {File} file
 */
function loadFilePreview(file) {
  selectedFile = file;

  const placeholder       = document.getElementById('drop-placeholder');
  const previewContainer  = document.getElementById('image-preview-container');
  const previewImg        = document.getElementById('image-preview');
  const runBtn            = document.getElementById('run-inference-btn');

  // Revoke previous object URL if any
  if (previewImg.dataset.objectUrl) {
    URL.revokeObjectURL(previewImg.dataset.objectUrl);
  }

  const objectUrl = URL.createObjectURL(file);
  previewImg.src            = objectUrl;
  previewImg.dataset.objectUrl = objectUrl;

  placeholder.style.display      = 'none';
  previewContainer.style.display = '';
  runBtn.disabled                = false;
}

/**
 * Reset the upload area to its initial empty state.
 */
function resetUploadArea() {
  const placeholder       = document.getElementById('drop-placeholder');
  const previewContainer  = document.getElementById('image-preview-container');
  const previewImg        = document.getElementById('image-preview');
  const fileInput         = document.getElementById('file-input');
  const runBtn            = document.getElementById('run-inference-btn');

  if (previewImg.dataset.objectUrl) {
    URL.revokeObjectURL(previewImg.dataset.objectUrl);
    delete previewImg.dataset.objectUrl;
  }

  previewImg.src                 = '';
  previewContainer.style.display = 'none';
  placeholder.style.display      = '';
  fileInput.value                = '';
  runBtn.disabled                = true;
  selectedFile                   = null;
}

/**
 * Wire up the drop zone: dragover / dragleave / drop, click-to-browse,
 * file input change, and "Remove image" button.
 */
function initUploadArea() {
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const browseLink  = document.getElementById('drop-browse-link');
  const removeBtn   = document.getElementById('remove-image-btn');

  const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];

  // Click on the drop zone (but not on the remove button) opens the picker
  dropZone.addEventListener('click', (e) => {
    if (e.target === removeBtn || removeBtn.contains(e.target)) return;
    fileInput.click();
  });

  // "click to browse" link also opens the picker
  browseLink.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // File picker selection
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file && ACCEPTED_TYPES.includes(file.type)) {
      loadFilePreview(file);
    }
  });

  // Drag & Drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (!file) return;

    if (!ACCEPTED_TYPES.includes(file.type)) {
      showInferenceError('Only JPEG and PNG files are accepted.');
      return;
    }

    loadFilePreview(file);
  });

  // Remove image button
  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploadArea();
    hideResults();
  });
}

/* ---- Inference submission ---- */

function showInferenceError(message) {
  const errorEl   = document.getElementById('inference-error');
  const errorText = document.getElementById('inference-error-text');
  errorText.textContent = message;
  errorEl.style.display = '';
}

function hideInferenceError() {
  document.getElementById('inference-error').style.display = 'none';
}

function showSpinner() {
  document.getElementById('loading-spinner').style.display = '';
}

function hideSpinner() {
  document.getElementById('loading-spinner').style.display = 'none';
}

function hideResults() {
  document.getElementById('results-area').style.display = 'none';
}

/**
 * POST image to /api/inference and render results.
 */
async function runInference() {
  if (!selectedFile) return;

  const cameraId = 'cam_stub_01';  // default stub camera

  showSpinner();
  hideInferenceError();
  hideResults();

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('camera_id', cameraId);

  try {
    const response = await fetch('/api/inference', {
      method: 'POST',
      body: formData,
    });

    hideSpinner();

    if (!response.ok) {
      let errMsg = `Server error (HTTP ${response.status})`;
      try {
        const errBody = await response.json();
        errMsg = errBody.error || errMsg;
      } catch (_) {}
      showInferenceError(errMsg);
      return;
    }

    const data = await response.json();
    renderInferenceResults(data, cameraId);
  } catch (err) {
    hideSpinner();
    showInferenceError('Network error — could not reach the server.');
    console.error('[app.js] runInference failed:', err);
  }
}

/**
 * Render the annotated image, detection table, and location strip.
 *
 * @param {object} data      Response JSON from POST /api/inference
 * @param {string} cameraId  The camera_id used in the request
 */
function renderInferenceResults(data, cameraId) {
  const resultsArea     = document.getElementById('results-area');
  const annotatedImg    = document.getElementById('annotated-image');
  const tableBody       = document.getElementById('detection-table-body');
  const noDetectionsMsg = document.getElementById('no-detections-msg');

  // Annotated image
  if (data.annotated_image) {
    annotatedImg.src           = `data:image/png;base64,${data.annotated_image}`;
    annotatedImg.style.display = '';
  } else {
    annotatedImg.src           = '';
    annotatedImg.style.display = 'none';
  }

  // Detection table
  const results = data.results || [];
  tableBody.innerHTML = '';

  if (results.length === 0) {
    noDetectionsMsg.style.display = '';
  } else {
    noDetectionsMsg.style.display = 'none';
    results.forEach(det => {
      const row = document.createElement('tr');
      row.className = det.is_hazard ? 'row-hazard' : 'row-safe';

      const statusPill = det.is_hazard
        ? '<span class="status-pill hazard">HAZARD</span>'
        : '<span class="status-pill safe">NOT HAZARD</span>';

      row.innerHTML = `
        <td>${det.class_label || '—'}</td>
        <td>${(det.confidence != null) ? Math.round(det.confidence * 100) + '%' : '—'}</td>
        <td>${statusPill}</td>
        <td>${det.hazard_reason || '—'}</td>
      `;
      tableBody.appendChild(row);
    });
  }

  // Location strip
  populateLocationStrip(cameraId);

  resultsArea.style.display = '';
}

/**
 * Fill the location strip with data derived from the camera_id.
 *
 * @param {string} cameraId
 */
function populateLocationStrip(cameraId) {
  const loc = getLocationContext(cameraId);

  const setEl = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value || '';
  };

  setEl('loc-facility', loc.facility);
  setEl('loc-berth',    loc.berth);
  setEl('loc-crane',    loc.crane);
  setEl('loc-camera',   loc.camera);

  // Hide separator dots for empty fields
  const strip = document.getElementById('location-strip');
  if (!strip) return;
  const items = strip.querySelectorAll('.location-item');
  const seps  = strip.querySelectorAll('.location-sep');

  // Show/hide separators based on whether adjacent items are populated
  let lastVisible = -1;
  items.forEach((item, i) => {
    const visible = item.textContent.trim() !== '';
    item.style.display = visible ? '' : 'none';
    if (visible) lastVisible = i;
  });

  // Hide all seps then re-enable only those between two visible items
  seps.forEach(sep => sep.style.display = 'none');
  let prevVisible = -1;
  items.forEach((item, i) => {
    if (item.style.display !== 'none') {
      if (prevVisible >= 0 && seps[prevVisible]) {
        seps[prevVisible].style.display = '';
      }
      prevVisible = i;
    }
  });
}

/* ---- "Use Dataset Image" button ---- */

/**
 * Fetch a test image from GET /api/test-image and pre-load it into the upload area.
 */
async function loadDatasetImage() {
  const btn = document.getElementById('use-dataset-btn');
  btn.disabled = true;

  hideInferenceError();

  try {
    const response = await fetch('/api/test-image');

    if (response.status === 404) {
      showInferenceError('No dataset images available.');
      return;
    }

    if (!response.ok) {
      showInferenceError(`Could not load dataset image (HTTP ${response.status}).`);
      return;
    }

    const blob = await response.blob();
    const mimeType = blob.type || 'image/jpeg';
    const file = new File([blob], 'dataset_image.jpg', { type: mimeType });

    loadFilePreview(file);
  } catch (err) {
    showInferenceError('Network error — could not fetch dataset image.');
    console.error('[app.js] loadDatasetImage failed:', err);
  } finally {
    btn.disabled = false;
  }
}

/* ============================================================
   MODAL EVENT WIRING
   ============================================================ */

function initModal() {
  const modal    = document.getElementById('details-modal');
  const closeBtn = document.getElementById('modal-close-btn');

  // Close button
  closeBtn.addEventListener('click', closeDetailsModal);

  // Clicking the overlay (but not the modal content) closes it
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeDetailsModal();
    }
  });

  // Escape key closes the modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') {
      closeDetailsModal();
    }
  });
}

/* ============================================================
   INFERENCE BUTTON WIRING
   ============================================================ */

function initInferenceButtons() {
  document.getElementById('run-inference-btn').addEventListener('click', runInference);
  document.getElementById('use-dataset-btn').addEventListener('click', loadDatasetImage);
}

/* ============================================================
   DOM CONTENT LOADED — entry point
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  // 11.1 — page-load initialisation
  initStatus();
  initRecentHazards();

  // 11.3 — modal wiring
  initModal();

  // 11.4 — upload area + inference buttons
  initUploadArea();
  initInferenceButtons();
});
