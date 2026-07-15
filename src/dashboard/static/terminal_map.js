/**
 * terminal_map.js
 * SVG terminal schematic for the Yard Hazard Inference Dashboard.
 * Programmatically builds and inserts an inline SVG into #terminal-map-container.
 * Exports: highlightCamera(cameraId), clearHighlight()
 *
 * SVG viewport: 800 × 500 units
 *
 * TODO (map-swap-1): Replace SVG base-layer with geo-referenced terminal
 *   image once official map is approved. Use maps/ntr_*.png and
 *   maps/Screenshot 2026-07-08 124802.png as visual references for layout.
 *   Interactive reference: https://3d.stowlog.com/apmt-pier-400/main
 *
 * TODO (map-swap-2): Update CAMERA_PINS and LOCATION_PINS x,y coordinates
 *   to match the new base image pixel space (or convert to lat/lng if using
 *   a tile layer).
 *
 * TODO (camera-mapping-1): Implement LocationContext.from_camera_id() with
 *   real camera registry data (berth, crane, named location, GPS) once
 *   cameras are connected and registered.
 *
 * TODO (camera-mapping-2): Verify crane count and berth distribution with
 *   yard operations. Current stub places cranes along the entire berth face
 *   without confirmed per-berth breakdown.
 *
 * TODO (camera-mapping-3): Camera 15 berth/crane assignment is unknown —
 *   currently placed at gate area (Truck Exchange Lane). Update once confirmed.
 *
 * TODO (location-17): Confirm the name of location 17 — visible near the
 *   admin complex in reference images but not fully labelled in the legend.
 *
 * TODO (traffic-lanes): Verify TEL'S Zone A-E order (west-to-east or
 *   east-to-west) and exact positions against finalised map.
 *
 * TODO (map-landmark-1): Add any remaining yard landmark names to
 *   LocationContext.landmark field and to LOCATION_PINS once yard map
 *   is finalised with operations team.
 *
 * TODO (location-expand): Expand LocationContext berth list beyond
 *   403/405/406 if additional berths are added to scope.
 */

"use strict";

// ============================================================
// CONFIGURATION DATA
// ============================================================

/**
 * Container stack fill colours — randomised per yard block, decorative only.
 * NOT linked to hazard status.
 */
const CONTAINER_COLOURS = ["#3b82f6", "#16a34a", "#ea580c", "#ca8a04", "#f3f4f6"];

// ============================================================
// ⚠️  TODO (camera-mapping): Replace CAMERA_PINS with real
//     camera-to-location data once the camera registry is set
//     up. Each camera should report its own location metadata
//     when connected (berth, crane number, GPS coords or pixel
//     coords relative to a real map image).
//     Reference: LocationContext.from_camera_id() in models.py
// ============================================================
/** @type {Array<{id:string,label:string,berth:string,crane:string,location:string,x:number,y:number}>} */
const CAMERA_PINS = [
  // { id, label, berth, crane, location, x, y }  ← x,y: SVG viewport units (0–800 x 0–500)
  // Berth 403 cameras — western quay face
  { id: "cam_stub_01", label: "Cam 01", berth: "Berth 403", crane: "Crane 01", location: "Ship-to-Shore Cranes", x: 80,  y: 320 },
  { id: "cam_stub_02", label: "Cam 02", berth: "Berth 403", crane: "Crane 02", location: "Ship-to-Shore Cranes", x: 110, y: 330 },
  { id: "cam_stub_03", label: "Cam 03", berth: "Berth 403", crane: "Crane 03", location: "Ship-to-Shore Cranes", x: 140, y: 340 },
  { id: "cam_stub_04", label: "Cam 04", berth: "Berth 403", crane: "Crane 04", location: "Ship-to-Shore Cranes", x: 170, y: 350 },
  // Berth 405 cameras — central quay face
  { id: "cam_stub_05", label: "Cam 05", berth: "Berth 405", crane: "Crane 05", location: "Ship-to-Shore Cranes", x: 300, y: 290 },
  { id: "cam_stub_06", label: "Cam 06", berth: "Berth 405", crane: "Crane 06", location: "Ship-to-Shore Cranes", x: 340, y: 280 },
  { id: "cam_stub_07", label: "Cam 07", berth: "Berth 405", crane: "Crane 07", location: "Ship-to-Shore Cranes", x: 380, y: 270 },
  { id: "cam_stub_08", label: "Cam 08", berth: "Berth 405", crane: "Crane 08", location: "Ship-to-Shore Cranes", x: 420, y: 260 },
  { id: "cam_stub_09", label: "Cam 09", berth: "Berth 405", crane: "Crane 09", location: "Ship-to-Shore Cranes", x: 460, y: 250 },
  // Berth 406 cameras — eastern quay face
  { id: "cam_stub_10", label: "Cam 10", berth: "Berth 406", crane: "Crane 10", location: "Ship-to-Shore Cranes", x: 540, y: 230 },
  { id: "cam_stub_11", label: "Cam 11", berth: "Berth 406", crane: "Crane 11", location: "Ship-to-Shore Cranes", x: 580, y: 220 },
  { id: "cam_stub_12", label: "Cam 12", berth: "Berth 406", crane: "Crane 12", location: "Ship-to-Shore Cranes", x: 620, y: 210 },
  { id: "cam_stub_13", label: "Cam 13", berth: "Berth 406", crane: "Crane 13", location: "Ship-to-Shore Cranes", x: 660, y: 200 },
  { id: "cam_stub_14", label: "Cam 14", berth: "Berth 406", crane: "Crane 14", location: "Ship-to-Shore Cranes", x: 700, y: 190 },
  // Gate area camera — landside
  // TODO (camera-mapping-3): Camera 15 berth/crane assignment unknown — covers gate area only
  { id: "cam_stub_15", label: "Cam 15", berth: "",          crane: "",          location: "Truck Exchange Lane",  x: 200, y: 430 },
];

// Coordinates are stub approximations based on map reference images.
// TODO (location-pins): verify all x,y positions against finalised map.
/** @type {Array<{id:number,name:string,x:number,y:number}>} */
const LOCATION_PINS = [
  { id: 1,  name: "In-Gate",                    x: 210, y: 450 },
  { id: 2,  name: "Out-Gate",                   x: 230, y: 440 },
  { id: 3,  name: "Admin Building",             x: 250, y: 430 },
  { id: 4,  name: "Reefer Racks",               x: 120, y: 380 },
  { id: 5,  name: "Asset Management",           x: 150, y: 410 },
  { id: 6,  name: "Rail",                       x: 400, y: 390 },
  { id: 7,  name: "Genset Drop-off / Pick-Up",  x: 220, y: 445 },
  { id: 8,  name: "Clean Truck Express Lane",   x: 730, y: 370 },
  { id: 9,  name: "Truck Exchange Lane",        x: 500, y: 385 },
  { id: 10, name: "Highline",                   x: 290, y: 145 },
  { id: 11, name: "Container Exchange Gates",   x: 260, y: 415 },
  { id: 12, name: "Roadability",                x: 190, y: 460 },
  { id: 13, name: "Vendor Gate",                x: 245, y: 435 },
  { id: 14, name: "Truck Exchange Lanes Path",  x: 600, y: 380 },
  { id: 15, name: "Yard Layout",                x: 450, y: 200 },
  { id: 16, name: "Ship-to-Shore Cranes",       x: 650, y: 155 },
  // TODO (location-17): confirm name for location 17 — visible near admin
  //   complex in reference images but not fully labelled in legend
  { id: 17, name: "Reefer / Support Area",      x: 255, y: 425 },
];

/** @type {Array<{id:string,label:string,x:number,y:number,width:number,height:number}>} */
const TRAFFIC_LANES = [
  { id: "tels-e", label: "TEL'S Zone E", x: 100, y: 355, width: 120, height: 18 },
  { id: "tels-d", label: "TEL'S Zone D", x: 220, y: 355, width: 120, height: 18 },
  { id: "tels-c", label: "TEL'S Zone C", x: 340, y: 355, width: 120, height: 18 },
  { id: "tels-b", label: "TEL'S Zone B", x: 460, y: 355, width: 120, height: 18 },
  // TODO (traffic-lanes): verify zone order (E→A west-to-east or east-to-west)
  //   and exact x positions against finalised map.
  { id: "tels-a", label: "TEL'S Zone A", x: 580, y: 355, width: 120, height: 18 },
];

// ============================================================
// SVG HELPERS
// ============================================================

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Create an SVG element with optional attributes.
 * @param {string} tag
 * @param {Object} attrs
 * @returns {SVGElement}
 */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

/**
 * Append children to a parent element.
 * @param {Element} parent
 * @param {...Element} children
 */
function append(parent, ...children) {
  for (const c of children) parent.appendChild(c);
}

/**
 * Pick a random element from an array.
 * @template T
 * @param {T[]} arr
 * @returns {T}
 */
function randomPick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ============================================================
// GROUP BUILDERS
// ============================================================

/**
 * Base layer — L-shaped terminal outline, water fill, land fill, Navy Way road.
 * // TODO (map-swap): replace with geo-referenced image once approved. See maps/ntr_*.png
 * @returns {SVGGElement}
 */
function buildBaseLayer() {
  const g = svgEl("g", { id: "base-layer" });

  // Water background — covers the full viewport
  append(g, svgEl("rect", { x: 0, y: 0, width: 800, height: 500, fill: "#bfdbfe" }));

  // L-shaped land / yard surface (warm yellow, navy blue outline)
  // Upper/northeast block: main container yard (x=60 to 780, y=120 to 380)
  // Lower/southwest block: gate & admin area (x=60 to 300, y=380 to 495)
  const landPath = "M60,120 L780,120 L780,380 L300,380 L300,495 L60,495 Z";
  append(g, svgEl("path", {
    d: landPath,
    fill: "#fbbf24",
    stroke: "#1e3a5f",
    "stroke-width": "3",
  }));

  // Navy Way road — diagonal grey band across the lower-left area (~160,400)
  append(g, svgEl("rect", {
    x: 60, y: 380, width: 240, height: 115,
    fill: "#6b7280",
  }));

  return g;
}

/**
 * Berths group — quay edge strip and berth zone labels.
 * @returns {SVGGElement}
 */
function buildBerths() {
  const g = svgEl("g", { id: "berths" });

  // Quay edge strip — dark grey concrete apron spanning the berth face
  append(g, svgEl("rect", {
    x: 60, y: 315, width: 700, height: 15,
    fill: "#374151",
  }));

  // Berth labels
  const berths = [
    { label: "Berth 403", x: 120 },
    { label: "Berth 405", x: 380 },
    { label: "Berth 406", x: 600 },
  ];
  for (const b of berths) {
    const txt = svgEl("text", {
      x: b.x, y: 312,
      fill: "#ffffff",
      "font-size": "11",
      "font-family": "sans-serif",
      "font-weight": "bold",
      "text-anchor": "middle",
    });
    txt.textContent = b.label;
    append(g, txt);
  }

  return g;
}

/**
 * Cranes group — 14 vertical tick marks + labels C01–C14.
 * // TODO (crane-positions): verify with yard operations
 * @returns {SVGGElement}
 */
function buildCranes() {
  const g = svgEl("g", { id: "cranes" });

  const startX = 80;
  const endX   = 750;
  const step   = (endX - startX) / 13; // 13 gaps between 14 cranes
  const topY   = 290;
  const botY   = 330;

  for (let i = 0; i < 14; i++) {
    const x = Math.round(startX + i * step);
    const label = `C${String(i + 1).padStart(2, "0")}`;

    // Vertical tick mark
    append(g, svgEl("line", {
      x1: x, y1: topY, x2: x, y2: botY,
      stroke: "#3b82f6",
      "stroke-width": "3",
    }));

    // Label above the tick
    const txt = svgEl("text", {
      x, y: topY - 4,
      fill: "#3b82f6",
      "font-size": "9",
      "font-family": "sans-serif",
      "text-anchor": "middle",
    });
    txt.textContent = label;
    append(g, txt);
  }

  return g;
}

/**
 * Yard blocks group — 8 container stack blocks in 2 rows × 4 cols.
 * Fill colour is randomised from CONTAINER_COLOURS per block.
 * @returns {SVGGElement}
 */
function buildYardBlocks() {
  const g = svgEl("g", { id: "yard-blocks" });

  // 2 rows × 4 cols layout inside the main yard (y≈160–310)
  const blockW  = 150;
  const blockH  = 65;
  const padX    = 20;
  const startX  = 75;
  const rowY    = [160, 240];

  let blockNum = 1;
  for (const y of rowY) {
    for (let col = 0; col < 4; col++) {
      const x    = startX + col * (blockW + padX);
      const fill = randomPick(CONTAINER_COLOURS);

      // Block rectangle
      append(g, svgEl("rect", {
        x, y,
        width: blockW,
        height: blockH,
        fill,
        stroke: "#1f2937",
        "stroke-width": "1.5",
        rx: "3",
      }));

      // Block label
      const txt = svgEl("text", {
        x: x + blockW / 2,
        y: y + blockH / 2 + 4,
        fill: "#ffffff",
        "font-size": "11",
        "font-family": "sans-serif",
        "font-weight": "bold",
        "text-anchor": "middle",
        "paint-order": "stroke",
        stroke: "#000000",
        "stroke-width": "2",
      });
      txt.textContent = `Yard Block ${blockNum}`;
      append(g, txt);

      blockNum++;
    }
  }

  return g;
}

/**
 * Traffic lanes group — 5 semi-transparent green bands (TEL'S Zone A–E).
 * @returns {SVGGElement}
 */
function buildTrafficLanes() {
  const g = svgEl("g", { id: "traffic-lanes" });

  for (const lane of TRAFFIC_LANES) {
    append(g, svgEl("rect", {
      id: lane.id,
      x: lane.x,
      y: lane.y,
      width: lane.width,
      height: lane.height,
      fill: "#16a34a",
      opacity: "0.55",
    }));

    const txt = svgEl("text", {
      x: lane.x + lane.width / 2,
      y: lane.y + 13,
      fill: "#ffffff",
      "font-size": "9",
      "font-family": "sans-serif",
      "text-anchor": "middle",
      "font-weight": "bold",
    });
    txt.textContent = lane.label;
    append(g, txt);
  }

  return g;
}

/**
 * Rail group — horizontal rail corridor stripe at y≈390–400.
 * @returns {SVGGElement}
 */
function buildRail() {
  const g = svgEl("g", { id: "rail" });

  // Rail corridor base — dark charcoal
  append(g, svgEl("rect", {
    x: 300, y: 375, width: 480, height: 20,
    fill: "#1f2937",
  }));

  // Red rail line (left track)
  append(g, svgEl("line", {
    x1: 300, y1: 380, x2: 780, y2: 380,
    stroke: "#ef4444", "stroke-width": "2",
  }));

  // Blue rail line (right track)
  append(g, svgEl("line", {
    x1: 300, y1: 390, x2: 780, y2: 390,
    stroke: "#3b82f6", "stroke-width": "2",
  }));

  return g;
}

/**
 * Navy Way diagonal road label.
 * @returns {SVGGElement}
 */
function buildNavyWay() {
  const g = svgEl("g", { id: "navy-way" });

  const txt = svgEl("text", {
    x: 160, y: 400,
    fill: "#ffffff",
    "font-size": "12",
    "font-family": "sans-serif",
    "font-weight": "bold",
    transform: "rotate(-20, 160, 400)",
  });
  txt.textContent = "Navy Way";
  append(g, txt);

  return g;
}

/**
 * Gate complex group — light grey admin/gate area lower-left.
 * @returns {SVGGElement}
 */
function buildGateComplex() {
  const g = svgEl("g", { id: "gate-complex" });

  append(g, svgEl("rect", {
    x: 60, y: 400, width: 220, height: 90,
    fill: "#d1d5db",
    stroke: "#6b7280",
    "stroke-width": "1.5",
  }));

  const lbl = svgEl("text", {
    x: 170, y: 448,
    fill: "#374151",
    "font-size": "10",
    "font-family": "sans-serif",
    "font-weight": "bold",
    "text-anchor": "middle",
  });
  lbl.textContent = "Gate & Admin Complex";
  append(g, lbl);

  return g;
}

/**
 * Named locations group — 17 numbered circle + SVG <title> tooltip pins.
 * @returns {SVGGElement}
 */
function buildNamedLocations() {
  const g = svgEl("g", { id: "named-locations" });

  for (const pin of LOCATION_PINS) {
    const circle = svgEl("circle", {
      cx: pin.x,
      cy: pin.y,
      r: "10",
      fill: "#60a5fa",
      stroke: "#1e3a5f",
      "stroke-width": "1.5",
      style: "cursor: pointer;",
    });

    // SVG <title> provides native browser hover tooltip
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${pin.id}. ${pin.name}`;
    circle.appendChild(title);
    append(g, circle);

    // Number label inside circle
    const txt = svgEl("text", {
      x: pin.x,
      y: pin.y + 4,
      fill: "#ffffff",
      "font-size": "9",
      "font-family": "sans-serif",
      "font-weight": "bold",
      "text-anchor": "middle",
      "pointer-events": "none",
    });
    txt.textContent = String(pin.id);
    append(g, txt);
  }

  return g;
}

/**
 * Camera pins group — 15 amber circles + text labels.
 * @returns {SVGGElement}
 */
function buildCameraPins() {
  const g = svgEl("g", { id: "camera-pins" });

  for (const pin of CAMERA_PINS) {
    const circle = svgEl("circle", {
      cx: pin.x,
      cy: pin.y,
      r: "7",
      fill: "#f59e0b",
      stroke: "#92400e",
      "stroke-width": "1.5",
      "data-camera-id": pin.id,
      style: "cursor: pointer;",
    });

    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${pin.label} — ${pin.berth}${pin.crane ? " · " + pin.crane : ""} · ${pin.location}`;
    circle.appendChild(title);
    append(g, circle);

    // Short label below pin
    const txt = svgEl("text", {
      x: pin.x,
      y: pin.y + 18,
      fill: "#92400e",
      "font-size": "8",
      "font-family": "sans-serif",
      "text-anchor": "middle",
      "pointer-events": "none",
    });
    txt.textContent = pin.label;
    append(g, txt);
  }

  return g;
}

/**
 * Highlight ring group — single hidden pulse circle.
 * @returns {SVGGElement}
 */
function buildHighlightRing() {
  const g = svgEl("g", { id: "highlight-ring" });

  const ring = svgEl("circle", {
    class: "highlight-ring hidden",
    cx: "0",
    cy: "0",
    r: "20",
  });
  append(g, ring);

  // Highlight label — camera ID + berth/crane info
  const lbl = svgEl("text", {
    id: "highlight-label",
    x: "0",
    y: "-28",
    fill: "#ffffff",
    "font-size": "10",
    "font-family": "sans-serif",
    "font-weight": "bold",
    "text-anchor": "middle",
    class: "highlight-label hidden",
    "paint-order": "stroke",
    stroke: "#000000",
    "stroke-width": "2",
  });
  append(g, lbl);

  return g;
}

/**
 * Compact map legend — 6 entries, pure SVG.
 * @returns {SVGGElement}
 */
function buildLegend() {
  const g = svgEl("g", { id: "map-legend" });

  // Background panel
  append(g, svgEl("rect", {
    x: "590", y: "405", width: "200", height: "90",
    fill: "rgba(17,24,39,0.85)", rx: "4",
    stroke: "#374151", "stroke-width": "1",
  }));

  const entries = [
    { colour: "#3b82f6",  label: "Crane" },
    { colour: "#ea580c",  label: "Containers" },
    { colour: "#f3f4f6",  label: "Reefers" },
    { colour: "#16a34a",  label: "Traffic Lane" },
    { colour: "#f59e0b",  label: "Camera" },
    { stroke: "#ffffff",  label: "Active Highlight" },
  ];

  entries.forEach((e, i) => {
    const y = 418 + i * 13;
    if (e.stroke) {
      append(g, svgEl("circle", {
        cx: "604", cy: y, r: "5",
        fill: "none", stroke: e.stroke, "stroke-width": "1.5",
      }));
    } else {
      append(g, svgEl("rect", {
        x: "599", y: y - 5, width: "10", height: "10",
        fill: e.colour, rx: "2",
      }));
    }
    const txt = svgEl("text", {
      x: "617", y: y + 4,
      fill: "#f9fafb", "font-size": "9", "font-family": "sans-serif",
    });
    txt.textContent = e.label;
    append(g, txt);
  });

  return g;
}

// ============================================================
// INLINE STYLES
// ============================================================

/**
 * Build and inject the inline <style> block for the map widget.
 * @returns {HTMLStyleElement}
 */
function buildStyles() {
  const style = document.createElement("style");
  style.textContent = `
    /* Map container */
    #terminal-map-container {
      position: relative;
      width: 100%;
      max-width: 820px;
      margin: 0 auto;
    }

    /* WIP disclaimer banner */
    .map-wip-banner {
      background: #b45309;
      color: #fff;
      font-size: 12px;
      font-weight: bold;
      padding: 6px 12px;
      border-radius: 4px 4px 0 0;
      text-align: center;
      letter-spacing: 0.02em;
    }

    /* Highlight ring */
    .highlight-ring {
      stroke: white;
      stroke-width: 3;
      fill: none;
    }
    .highlight-ring.hidden {
      display: none;
    }
    .highlight-label.hidden {
      display: none;
    }

    /* Pulse animation */
    @keyframes pulse-ring {
      0%   { r: 20; opacity: 1; }
      100% { r: 35; opacity: 0; }
    }
    .highlight-ring:not(.hidden) {
      animation: pulse-ring 1.2s ease-out infinite;
    }

    /* Map legend box */
    .map-legend-box {
      background: rgba(17,24,39,0.85);
      border: 1px solid #374151;
      border-radius: 4px;
      padding: 5px 8px;
      font-size: 10px;
      color: #f9fafb;
      font-family: sans-serif;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 5px;
      margin-bottom: 2px;
    }
    .legend-swatch {
      display: inline-block;
      width: 14px;
      height: 10px;
      border-radius: 2px;
      flex-shrink: 0;
    }

    /* Traffic lane CSS (also applied via attribute for extra specificity) */
    .traffic-lane {
      fill: #16a34a;
      opacity: 0.55;
    }
  `;
  return style;
}

// ============================================================
// MAIN RENDER
// ============================================================

/**
 * Build the full SVG schematic and insert it into #terminal-map-container.
 * Also prepends the WIP disclaimer banner.
 */
function renderTerminalMap() {
  const container = document.getElementById("terminal-map-container");
  if (!container) {
    console.warn("terminal_map.js: #terminal-map-container not found.");
    return;
  }

  // Inject styles
  document.head.appendChild(buildStyles());

  // WIP disclaimer banner — prepended before the SVG
  const banner = document.createElement("div");
  banner.className = "map-wip-banner";
  banner.textContent =
    "NOT OFFICIAL LOCATIONS — WORK IN PROGRESS. " +
    "Camera positions are placeholder stubs and do not reflect real equipment.";
  container.prepend(banner);

  // Build SVG root
  const svg = svgEl("svg", {
    viewBox: "0 0 800 500",
    width: "800",
    height: "500",
    xmlns: SVG_NS,
    "aria-label": "Terminal schematic map",
    role: "img",
    style: "display:block;width:100%;height:auto;",
  });

  // Layer order matters — append in z-index order (back to front)
  append(svg,
    buildBaseLayer(),       // 1. background water + land
    buildBerths(),          // 2. quay edge strip + berth labels
    buildCranes(),          // 3. 14 crane tick marks
    buildYardBlocks(),      // 4. 8 container stack blocks
    buildTrafficLanes(),    // 5. 5 TEL'S Zone bands
    buildRail(),            // 6. rail corridor
    buildNavyWay(),         // 7. Navy Way diagonal label
    buildGateComplex(),     // 8. gate & admin rectangle
    buildNamedLocations(),  // 9. 17 numbered location pins
    buildCameraPins(),      // 10. 15 amber camera pins
    buildHighlightRing(),   // 11. pulse highlight ring (top layer)
    buildLegend(),          // 12. compact legend overlay
  );

  container.appendChild(svg);
}

// Auto-render when the DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderTerminalMap);
} else {
  renderTerminalMap();
}

// ============================================================
// PUBLIC API — Task 10.2
// ============================================================

/**
 * Move the highlight ring to the camera pin matching cameraId,
 * show it, and start the pulse animation.
 *
 * If no matching pin is found the ring stays hidden; no error is thrown.
 *
 * Exported as window.terminalMap.highlightCamera for non-module scripts.
 * Also exported via ES module export for module-aware bundlers/tests.
 *
 * @param {string} cameraId  — e.g. "cam_stub_01"
 */
function highlightCamera(cameraId) {
  const pin = CAMERA_PINS.find((p) => p.id === cameraId);

  const ring  = document.querySelector(".highlight-ring");
  const label = document.getElementById("highlight-label");

  if (!ring) return; // SVG not rendered yet

  if (!pin) {
    // No matching pin — keep ring hidden
    ring.classList.add("hidden");
    if (label) label.classList.add("hidden");
    return;
  }

  // Move ring to pin position
  ring.setAttribute("cx", String(pin.x));
  ring.setAttribute("cy", String(pin.y));

  // Show ring and restart animation (re-attach element to re-trigger keyframes)
  ring.classList.remove("hidden");

  // Update label text showing camera ID + berth/crane
  if (label) {
    label.setAttribute("x", String(pin.x));
    label.setAttribute("y", String(pin.y - 28));
    label.textContent = `${pin.label}${pin.berth ? " · " + pin.berth : ""}${pin.crane ? " · " + pin.crane : ""}`;
    label.classList.remove("hidden");
  }
}

/**
 * Hide the highlight ring and clear the label.
 */
function clearHighlight() {
  const ring  = document.querySelector(".highlight-ring");
  const label = document.getElementById("highlight-label");

  if (ring)  ring.classList.add("hidden");
  if (label) label.classList.add("hidden");
}

// ============================================================
// GLOBAL NAMESPACE EXPORT
// Expose API on window.terminalMap so non-module scripts (app.js)
// can call window.terminalMap.highlightCamera(id) and
// window.terminalMap.clearHighlight() regardless of module loading order.
// ============================================================
window.terminalMap = { highlightCamera, clearHighlight };
