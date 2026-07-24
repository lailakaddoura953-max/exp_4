/**
 * terminal_map.js — v2
 * Renders the real terminal site map PNG with numbered location pin overlays.
 * Pins are positioned using percentage-based coordinates from /api/map/config
 * (backed by config/dashboard_map.json) so they work at any display resolution.
 *
 * Exports: window.terminalMap.highlightLocation(locationId)
 *          window.terminalMap.clearHighlight()
 */

"use strict";

// ============================================================
// STATE
// ============================================================

let _mapConfig = null;   // loaded from /api/map/config
let _activePin = null;   // currently highlighted location ID

// ============================================================
// INITIALIZATION
// ============================================================

async function initTerminalMap() {
  const container = document.getElementById("terminal-map-container");
  if (!container) {
    console.warn("terminal_map.js: #terminal-map-container not found.");
    return;
  }

  // Load map config (pin positions + location names)
  try {
    const resp = await fetch("/api/map/config");
    if (resp.ok) {
      _mapConfig = await resp.json();
    } else {
      console.warn("terminal_map.js: /api/map/config returned", resp.status, "— using fallback.");
      _mapConfig = getFallbackConfig();
    }
  } catch (err) {
    console.warn("terminal_map.js: failed to fetch /api/map/config —", err.message);
    _mapConfig = getFallbackConfig();
  }

  renderMap(container);
}

// ============================================================
// RENDER
// ============================================================

function renderMap(container) {
  container.innerHTML = "";

  // Map wrapper — relative positioning anchor for pin overlays
  const wrapper = document.createElement("div");
  wrapper.className = "map-wrapper";
  wrapper.style.position = "relative";
  wrapper.style.display = "inline-block";
  wrapper.style.width = "100%";

  // Site map image
  const img = document.createElement("img");
  img.src = "/static/site_map.png";
  img.alt = "Terminal site map";
  img.className = "site-map-image";
  img.style.width = "100%";
  img.style.height = "auto";
  img.style.display = "block";
  img.style.borderRadius = "8px";
  wrapper.appendChild(img);

  // Location pins — REMOVED for now. Pin positions need to be visually
  // calibrated against the actual site_map.png before they can be rendered
  // correctly. The map image renders without pins until positions are confirmed.
  // To re-enable: uncomment the block below and correct x_pct/y_pct values
  // in config/dashboard_map.json to match the real PNG pixel layout.
  //
  // const locations = _mapConfig.locations || [];
  // for (const loc of locations) { ... pin rendering ... }

  container.appendChild(wrapper);

  // Location list — right-hand column in kanban-card style
  const listColumn = document.createElement("div");
  listColumn.className = "map-location-list";

  const listHeading = document.createElement("h3");
  listHeading.className = "location-list-heading";
  listHeading.textContent = "Locations";
  listColumn.appendChild(listHeading);

  const locations = _mapConfig.locations || [];
  for (const loc of locations) {
    const card = document.createElement("div");
    card.className = "location-card";
    card.id = `loc-card-${loc.id}`;
    card.innerHTML = `<span class="loc-card-number">${loc.id}</span><span class="loc-card-name">${loc.name}</span>`;
    listColumn.appendChild(card);
  }

  // Grid wrapper: map left, location list right
  const grid = document.createElement("div");
  grid.className = "map-grid";
  grid.appendChild(wrapper);
  grid.appendChild(listColumn);

  container.appendChild(grid);
}

// ============================================================
// PUBLIC API
// ============================================================

/**
 * Highlight a map pin by location ID (1-16).
 * Turns the pin red with a pulse effect to indicate active hazard.
 * @param {number} locationId
 */
function highlightLocation(locationId) {
  clearHighlight();

  const pin = document.getElementById(`map-pin-${locationId}`);
  if (!pin) return;

  pin.style.backgroundColor = "#ef4444";
  pin.style.border = "2px solid #7f1d1d";
  pin.style.transform = "translate(-50%, -50%) scale(1.3)";
  pin.style.boxShadow = "0 0 12px rgba(239, 68, 68, 0.7)";
  pin.style.zIndex = "20";
  _activePin = locationId;
}

/**
 * Clear any active pin highlight, returning all pins to default state.
 */
function clearHighlight() {
  if (_activePin !== null) {
    const pin = document.getElementById(`map-pin-${_activePin}`);
    if (pin) {
      pin.style.backgroundColor = "#3b82f6";
      pin.style.border = "2px solid #1e3a5f";
      pin.style.transform = "translate(-50%, -50%)";
      pin.style.boxShadow = "none";
      pin.style.zIndex = "10";
    }
    _activePin = null;
  }
}

// ============================================================
// FALLBACK CONFIG (if /api/map/config is unavailable)
// ============================================================

function getFallbackConfig() {
  return {
    locations: [
      { id: 1,  name: "In-Gate",                    x_pct: 88, y_pct: 48 },
      { id: 2,  name: "Out-Gate",                   x_pct: 42, y_pct: 72 },
      { id: 3,  name: "Admin Building",             x_pct: 30, y_pct: 60 },
      { id: 4,  name: "Reefer Racks",               x_pct: 10, y_pct: 42 },
      { id: 5,  name: "Asset Management",           x_pct: 28, y_pct: 82 },
      { id: 6,  name: "Rail",                       x_pct: 55, y_pct: 52 },
      { id: 7,  name: "Genset Drop-off / Pick-Up",  x_pct: 33, y_pct: 72 },
      { id: 8,  name: "Clean Truck Express Lane",   x_pct: 90, y_pct: 42 },
      { id: 9,  name: "Truck Exchange Lane",        x_pct: 58, y_pct: 45 },
      { id: 10, name: "Highline",                   x_pct: 18, y_pct: 12 },
      { id: 11, name: "Container Exchange Gates",   x_pct:  5, y_pct: 55 },
      { id: 12, name: "Roadability",                x_pct: 22, y_pct: 88 },
      { id: 13, name: "Vendor Gate",                x_pct: 38, y_pct: 58 },
      { id: 14, name: "Truck Exchange Lanes Path",  x_pct: 68, y_pct: 45 },
      { id: 15, name: "Yard Layout",                x_pct: 42, y_pct: 28 },
      { id: 16, name: "Ship-to-Shore Cranes",       x_pct: 58, y_pct: 12 },
    ],
  };
}

// ============================================================
// AUTO-INIT
// ============================================================

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTerminalMap);
} else {
  initTerminalMap();
}

// ============================================================
// GLOBAL NAMESPACE EXPORT
// ============================================================
window.terminalMap = { highlightLocation, clearHighlight };
