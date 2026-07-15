"""
Annotation Review Tool
======================
A keyboard-driven OpenCV viewer for stepping through the review queue
produced by auto_annotate.py.

For each image in the review queue it shows:
  - The image with polygon overlays and confidence scores
  - A panel listing each detected object, its class name, and confidence
  - Keyboard controls to accept, reject, or relabel each annotation

Keyboard controls (shown on screen):
    A         — Accept ALL annotations on this image → move to auto_accepted/
    R         — Reject ALL annotations on this image → move to rejected/
    SPACE     — Accept current annotation, move to next
    X         — Reject current annotation (remove from label file)
    1-9 / 0   — Relabel current annotation to class ID (0=10, 1=1, …)
    N         — Skip to next image without saving changes
    S         — Save current state and move to next image
    Q / ESC   — Quit and save progress

Output:
    Accepted images + corrected labels → review_queue/../auto_accepted/
    Rejected images (no labels)        → review_queue/../rejected/
    Progress is saved after each image so you can resume any time.

Usage:
    python scripts/annotation/review_annotations.py \\
        --review_dir image_data_annotated/review_queue

    # Resume from where you left off (already-reviewed images are skipped):
    python scripts/annotation/review_annotations.py \\
        --review_dir image_data_annotated/review_queue --resume
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Class names (must match data.yaml order)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Boat - With Cargo",         # 0
    "Container - Misaligned",    # 1
    "Container - Open",          # 2
    "Container - Picked",        # 3
    "Container - Reefer",        # 4
    "Container - Water Drop",    # 5
    "Container - Separate",      # 6
    "Container - Stacked",       # 7
    "Crane",                     # 8
    "Human",                     # 9
    "Human - No Safety Clothes", # 10
    "Truck - No Container",      # 11
    "Truck - With Container",    # 12
    "Vehicle",                   # 13
    "Yard - Dropoff zone",       # 14
    "Yard - No People",          # 15
    "Yard - Operation Zone",     # 16
]

# Colours per class (BGR) — cycling through a palette for readability
PALETTE = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150,  50), (150, 200,  50), ( 50, 150, 200),
    (255, 180,   0), (  0, 180, 255), (180,   0, 255),
    (200,  80,  80), ( 80, 200,  80), ( 80,  80, 200),
    (220, 220,  50), ( 50, 220, 220),
]

SAFETY_CLASSES = {2, 9, 10}  # highlight these with a warning colour


def class_colour(class_id: int) -> tuple[int, int, int]:
    if class_id in SAFETY_CLASSES:
        return (0, 60, 255)  # bright red-orange for safety classes
    return PALETTE[class_id % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# Label file I/O
# ─────────────────────────────────────────────────────────────────────────────

def parse_label_line(line: str) -> Optional[tuple[int, list[tuple[float, float]]]]:
    """Parse one YOLO polygon line. Returns (class_id, [(x,y),...]) or None."""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            return None
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        return class_id, points
    except ValueError:
        return None


def load_labels(label_path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Load all annotations from a YOLO polygon label file."""
    if not label_path.exists():
        return []
    annotations = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(line)
        if parsed:
            annotations.append(parsed)
    return annotations


def write_labels(label_path: Path, annotations: list[tuple[int, list[tuple[float, float]]]]) -> None:
    """Write annotations to a YOLO polygon label file."""
    lines = []
    for class_id, points in annotations:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{class_id} {coords}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_meta(meta_path: Path) -> list[dict]:
    """Load the sidecar .meta.json confidence data written by auto_annotate.py."""
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(progress_path: Path) -> set[str]:
    """Load set of already-reviewed image stems from the progress file."""
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return set(data.get("reviewed", []))
    except Exception:
        return set()


def save_progress(progress_path: Path, reviewed: set[str]) -> None:
    progress_path.write_text(
        json.dumps({"reviewed": sorted(reviewed)}, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_polygon_overlay(
    canvas: np.ndarray,
    points_norm: list[tuple[float, float]],
    colour: tuple[int, int, int],
    alpha: float = 0.25,
    border_thickness: int = 2,
    is_selected: bool = False,
) -> np.ndarray:
    """Draw a filled semi-transparent polygon + border onto canvas (in-place copy)."""
    h, w = canvas.shape[:2]
    pts_px = np.array(
        [(int(x * w), int(y * h)) for x, y in points_norm], dtype=np.int32
    )
    if len(pts_px) < 3:
        return canvas

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_px], colour)
    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # Border — thicker and brighter when selected
    border_colour = (255, 255, 255) if is_selected else colour
    thickness = border_thickness + 2 if is_selected else border_thickness
    cv2.polylines(canvas, [pts_px], isClosed=True, color=border_colour, thickness=thickness)

    return canvas


def draw_label_badge(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    font_scale: float = 0.45,
) -> None:
    """Draw a filled rectangle badge with text at position (x, y)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Background rect
    cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 4, y + baseline), colour, cv2.FILLED)
    # Text in white or black depending on brightness
    brightness = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
    text_colour = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(canvas, text, (x, y), font, font_scale, text_colour, thickness, cv2.LINE_AA)


def build_info_panel(
    annotations: list[tuple[int, list[tuple[float, float]]]],
    meta: list[dict],
    selected_idx: int,
    img_idx: int,
    total_imgs: int,
    panel_w: int = 320,
    panel_h: int = 480,
) -> np.ndarray:
    """
    Build the right-hand info panel showing image progress,
    annotation list, keyboard shortcuts.
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20

    def text(msg, ypos, scale=0.42, colour=(200, 200, 200), bold=False):
        thickness = 2 if bold else 1
        cv2.putText(panel, msg, (10, ypos), font, scale, colour, thickness, cv2.LINE_AA)

    # Header
    text(f"Image {img_idx + 1} / {total_imgs}", y, scale=0.5, colour=(255, 255, 100), bold=True)
    y += 22
    text(f"{len(annotations)} annotation(s)", y, scale=0.42)
    y += 22

    # Separator
    cv2.line(panel, (0, y), (panel_w, y), (80, 80, 80), 1)
    y += 12

    # Annotation list
    for i, (cid, pts) in enumerate(annotations):
        is_sel = (i == selected_idx)
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        colour = class_colour(cid)
        bg_col = (70, 70, 70) if is_sel else (40, 40, 40)

        # Row background
        cv2.rectangle(panel, (4, y - 12), (panel_w - 4, y + 8), bg_col, cv2.FILLED)

        # Colour swatch
        cv2.rectangle(panel, (8, y - 8), (20, y + 4), colour, cv2.FILLED)

        # Class name + confidence
        score_str = ""
        if i < len(meta):
            score_str = f"  {meta[i].get('score', 0):.2f}"
        prefix = "► " if is_sel else "  "
        label = f"{prefix}[{cid}] {name[:18]}{score_str}"
        text_col = (255, 255, 255) if is_sel else (180, 180, 180)
        text(label, y, scale=0.38, colour=text_col)
        y += 20

        if y > panel_h - 120:
            text(f"  ... (+{len(annotations) - i - 1} more)", y, scale=0.38)
            break

    y = panel_h - 115
    cv2.line(panel, (0, y), (panel_w, y), (80, 80, 80), 1)
    y += 14

    # Keyboard guide
    controls = [
        ("SPACE",  "Accept current"),
        ("X",      "Reject current"),
        ("A",      "Accept ALL"),
        ("R",      "Reject ALL"),
        ("1-9/0",  "Relabel current (0=10)"),
        ("S",      "Save & next"),
        ("N",      "Skip (no save)"),
        ("Q/ESC",  "Quit"),
    ]
    text("Controls:", y, scale=0.4, colour=(150, 200, 255), bold=True)
    y += 16
    for key, action in controls:
        text(f"  {key:<8} {action}", y, scale=0.36, colour=(160, 160, 160))
        y += 14

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Review window — composites image + overlays + info panel
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_MAX_W = 1100   # max width of the image portion of the display
DISPLAY_MAX_H = 720    # max height of the image portion
PANEL_W       = 320


def build_display(
    image: np.ndarray,
    annotations: list[tuple[int, list[tuple[float, float]]]],
    meta: list[dict],
    selected_idx: int,
    rejected_set: set[int],
    img_idx: int,
    total_imgs: int,
) -> np.ndarray:
    """Compose the full display frame: resized image with overlays + side panel."""
    orig_h, orig_w = image.shape[:2]

    # Resize image to fit display area
    scale = min(DISPLAY_MAX_W / orig_w, DISPLAY_MAX_H / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    canvas = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Draw each annotation
    for i, (cid, pts) in enumerate(annotations):
        if i in rejected_set:
            continue  # don't show rejected annotations
        colour = class_colour(cid)
        is_sel = (i == selected_idx)
        canvas = draw_polygon_overlay(canvas, pts, colour, alpha=0.22, is_selected=is_sel)

        # Label badge near polygon centroid
        if pts:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            px = int(cx * disp_w)
            py = int(cy * disp_h)
            name = CLASS_NAMES[cid][:12] if cid < len(CLASS_NAMES) else f"cls{cid}"
            score = meta[i].get("score", 0) if i < len(meta) else 0
            badge = f"[{i}] {name} {score:.2f}"
            draw_label_badge(canvas, badge, (px, py), colour)

    # Pad canvas height to match panel height if needed
    panel_h = max(DISPLAY_MAX_H, disp_h)
    if disp_h < panel_h:
        pad = np.zeros((panel_h - disp_h, disp_w, 3), dtype=np.uint8)
        canvas = np.vstack([canvas, pad])

    # Build info panel
    panel = build_info_panel(
        annotations, meta, selected_idx,
        img_idx, total_imgs,
        panel_w=PANEL_W, panel_h=panel_h,
    )

    # Concatenate side by side
    display = np.hstack([canvas, panel])
    return display


# ─────────────────────────────────────────────────────────────────────────────
# Single-image review loop
# ─────────────────────────────────────────────────────────────────────────────

# Return codes from review_single_image
SAVE_AND_NEXT = "save_next"
SKIP          = "skip"
QUIT          = "quit"


def review_single_image(
    img_path: Path,
    annotations: list[tuple[int, list[tuple[float, float]]]],
    meta: list[dict],
    img_idx: int,
    total_imgs: int,
) -> tuple[str, list[tuple[int, list[tuple[float, float]]]]]:
    """
    Show one image in the review window and collect user decisions.
    Returns (action, final_annotations) where action is one of the
    SAVE_AND_NEXT / SKIP / QUIT constants.
    final_annotations excludes any rejected entries.
    """
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [warn] Cannot read image: {img_path}")
        return SKIP, annotations

    # Working copies — we mutate these as the user makes decisions
    current_annotations = list(annotations)   # list of (class_id, polygon)
    rejected_indices: set[int] = set()
    selected_idx = 0  # which annotation is currently highlighted

    window_name = "Annotation Review — Q to quit, S to save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_MAX_W + PANEL_W, DISPLAY_MAX_H)

    action = SAVE_AND_NEXT

    while True:
        # Clamp selected_idx to valid range
        n = len(current_annotations)
        if n == 0:
            selected_idx = 0
        else:
            selected_idx = max(0, min(selected_idx, n - 1))

        frame = build_display(
            image, current_annotations, meta,
            selected_idx, rejected_indices,
            img_idx, total_imgs,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC
            action = QUIT
            break

        elif key == ord('s'):              # Save & next
            action = SAVE_AND_NEXT
            break

        elif key == ord('n'):              # Skip without saving
            action = SKIP
            break

        elif key == ord('a'):              # Accept ALL
            rejected_indices.clear()
            action = SAVE_AND_NEXT
            break

        elif key == ord('r'):              # Reject ALL
            rejected_indices = set(range(len(current_annotations)))
            action = SAVE_AND_NEXT
            break

        elif key == ord(' '):              # Accept current, move to next
            if selected_idx in rejected_indices:
                rejected_indices.discard(selected_idx)
            selected_idx = min(selected_idx + 1, max(0, n - 1))

        elif key == ord('x'):              # Reject current
            rejected_indices.add(selected_idx)
            selected_idx = min(selected_idx + 1, max(0, n - 1))

        elif key == 82 or key == ord('k'): # Up arrow or K — previous annotation
            selected_idx = max(0, selected_idx - 1)

        elif key == 84 or key == ord('j'): # Down arrow or J — next annotation
            selected_idx = min(n - 1, selected_idx + 1)

        elif chr(key) in "0123456789":     # Relabel current annotation
            new_class = int(chr(key))
            if new_class == 0:
                new_class = 10  # 0 maps to class 10
            if 0 <= new_class < len(CLASS_NAMES) and n > 0:
                cid, pts = current_annotations[selected_idx]
                current_annotations[selected_idx] = (new_class, pts)
                # Also update meta score placeholder so panel refreshes cleanly
                if selected_idx < len(meta):
                    meta[selected_idx] = dict(meta[selected_idx])
                    meta[selected_idx]["class_id"] = new_class

    cv2.destroyAllWindows()

    # Build final annotation list (exclude rejected)
    final = [
        ann for i, ann in enumerate(current_annotations)
        if i not in rejected_indices
    ]
    return action, final


# ─────────────────────────────────────────────────────────────────────────────
# Output routing
# ─────────────────────────────────────────────────────────────────────────────

def route_output(
    img_path: Path,
    label_path: Path,
    meta_path: Path,
    final_annotations: list[tuple[int, list[tuple[float, float]]]],
    review_root: Path,
    action: str,
) -> None:
    """
    Save accepted images to auto_accepted/ (sibling of review_queue/).
    Empty-annotation images go to rejected/.
    """
    # auto_accepted/ and rejected/ are siblings of review_queue/
    accepted_root = review_root.parent / "auto_accepted"
    rejected_root = review_root.parent / "rejected"

    try:
        rel = img_path.relative_to(review_root)
    except ValueError:
        rel = Path(img_path.name)

    if final_annotations:
        dest_dir = accepted_root / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_img = dest_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        write_labels(dest_img.with_suffix(".txt"), final_annotations)
    else:
        # All annotations were rejected — move to rejected/
        dest_dir = rejected_root / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)

    # Clean up review queue files once processed
    for f in [img_path, label_path, meta_path]:
        if f and f.exists():
            try:
                f.unlink()
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Main review loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    review_root = Path(args.review_dir)
    if not review_root.exists():
        print(f"ERROR: --review_dir does not exist: {review_root}")
        print("Run auto_annotate.py first to populate the review queue.")
        sys.exit(1)

    # Discover all images in the review queue
    all_images: list[Path] = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"):
        all_images.extend(review_root.rglob(ext))
    all_images = sorted(all_images)

    if not all_images:
        print(f"No images found in review queue: {review_root}")
        print("Either auto_annotate.py hasn't run yet, or the queue is empty.")
        sys.exit(0)

    # Progress tracking
    progress_path = review_root.parent / ".review_progress.json"
    reviewed: set[str] = set()
    if args.resume:
        reviewed = load_progress(progress_path)
        print(f"Resuming: {len(reviewed)} images already reviewed")

    pending = [p for p in all_images if p.stem not in reviewed]

    if not pending:
        print("All images in the review queue have already been reviewed.")
        print("Use --resume=False to re-review them.")
        sys.exit(0)

    print(f"\n=== Annotation Review Tool ===")
    print(f"  Review queue : {review_root}")
    print(f"  Total images : {len(all_images)}")
    print(f"  Pending      : {len(pending)}")
    print(f"\n  Controls: SPACE=accept  X=reject  A=accept all  R=reject all")
    print(f"           S=save&next  N=skip  1-9/0=relabel  Q/ESC=quit\n")
    input("  Press Enter to start reviewing...")

    stats = {"accepted": 0, "rejected_all": 0, "skipped": 0}

    for img_idx, img_path in enumerate(pending):
        label_path = img_path.with_suffix(".txt")
        meta_path  = img_path.parent / (img_path.stem + ".meta.json")

        annotations = load_labels(label_path)
        meta        = load_meta(meta_path)

        if not annotations:
            # No labels — nothing to review, move to rejected
            route_output(img_path, label_path, meta_path, [], review_root, SAVE_AND_NEXT)
            reviewed.add(img_path.stem)
            save_progress(progress_path, reviewed)
            continue

        action, final = review_single_image(
            img_path, annotations, meta, img_idx, len(pending)
        )

        if action == QUIT:
            print(f"\n  Quitting — progress saved ({len(reviewed)} reviewed so far)")
            save_progress(progress_path, reviewed)
            break

        if action == SKIP:
            stats["skipped"] += 1
            continue

        # SAVE_AND_NEXT — commit the result
        route_output(img_path, label_path, meta_path, final, review_root, action)
        reviewed.add(img_path.stem)
        save_progress(progress_path, reviewed)

        if final:
            stats["accepted"] += 1
        else:
            stats["rejected_all"] += 1

    print(f"\n=== Review complete ===")
    print(f"  Accepted (with labels) : {stats['accepted']}")
    print(f"  Fully rejected         : {stats['rejected_all']}")
    print(f"  Skipped                : {stats['skipped']}")
    print(f"  Total reviewed         : {len(reviewed)}")
    accepted_root = review_root.parent / "auto_accepted"
    print(f"\n  Accepted labels ready for training: {accepted_root}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Review the annotation queue produced by auto_annotate.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--review_dir", type=str,
        default="image_data_annotated/review_queue",
        help="Path to the review_queue directory produced by auto_annotate.py",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip images that have already been reviewed in a previous session",
    )
    p.add_argument(
        "--no_resume", dest="resume", action="store_false",
        help="Re-review all images, ignoring previous progress",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
