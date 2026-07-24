"""
Capture Database — SQLite archive of live-camera frame metadata.

SQLite was chosen specifically because it's file-based and portable: the
resulting .db file lives right next to the JPEG/JSON files it indexes
(inside live_camera_captures/), so copying that one directory to another
machine brings the whole queryable history with it — no separate database
server to install or configure.

Two ways data gets into this database:
1. Live — LiveCaptureArchiver.save_frame() calls CaptureDatabase.record_frame()
   immediately after writing each frame's JPEG + JSON sidecar to disk.
2. Backfill/rebuild — CaptureDatabase.sync_from_archive() walks every JSON
   sidecar already on disk and (re)populates the database from them. This
   is what makes the database "auto-generate" from existing files: if
   live_camera_captures/ was copied from another machine without its .db
   file, or the .db file was deleted, running sync_from_archive() rebuilds
   it completely from the JSON sidecars alone. Both the live path and the
   backfill path funnel through the same record_frame() upsert logic, so
   re-running sync_from_archive() is always safe (idempotent — matches on
   frame_id, never creates duplicates).

Schema
------
frames       — one row per captured frame, keyed by frame_id (the frame's
               filename stem, e.g. "2026-07-24T14-03-01.120000Z_frame0" —
               the same identifier already used for the .jpg/.json pair on
               disk, per the user's choice to use the snapshot's own
               timestamp-derived name as its ID). Includes the location
               fields shown in the dashboard's detail view (facility,
               berth, crane, camera label, landmark) so a single query
               reproduces what the UI already displays per capture.
detections   — one row per YOLO detection within a frame (0 or more per
               frame_id), foreign-keyed to frames.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazard_detection.diagnostics import get_logger

logger = get_logger("capture_db")

# Database filename, created inside the same archive_root directory that
# LiveCaptureArchiver writes JPEG/JSON files to (e.g.
# "live_camera_captures/capture_log.db").
DEFAULT_DB_FILENAME = "capture_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id            TEXT PRIMARY KEY,
    camera_id           TEXT NOT NULL,
    capture_timestamp   TEXT NOT NULL,
    is_hazard_frame     INTEGER NOT NULL,
    image_path          TEXT,
    sidecar_path        TEXT,
    loc_facility        TEXT,
    loc_berth           TEXT,
    loc_crane           TEXT,
    loc_camera_label    TEXT,
    loc_landmark        TEXT,
    recorded_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_camera_id ON frames(camera_id);
CREATE INDEX IF NOT EXISTS idx_frames_capture_timestamp ON frames(capture_timestamp);
CREATE INDEX IF NOT EXISTS idx_frames_is_hazard ON frames(is_hazard_frame);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id        TEXT NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    class_label     TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox_x_center   REAL,
    bbox_y_center   REAL,
    bbox_width      REAL,
    bbox_height     REAL,
    is_hazard       INTEGER NOT NULL,
    hazard_reason   TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_is_hazard ON detections(is_hazard);
"""


class CaptureDatabase:
    """
    Thin SQLite wrapper around the frames/detections tables described above.

    One instance typically lives for the lifetime of the dashboard process
    (owned by LiveCaptureArchiver), but it's also safe to construct a
    short-lived instance purely to run sync_from_archive() from a script.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        # Auto-create the parent directory (e.g. live_camera_captures/) if
        # it doesn't exist yet — mirrors LiveCaptureArchiver's lazy
        # directory creation, so the database never requires manual setup.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        # check_same_thread=False because the dashboard's hourly timer
        # thread and Flask request threads may both write; the lock above
        # serializes all actual access so this is safe.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        logger.info("CaptureDatabase: schema ready at '%s'", self._db_path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record_frame(
        self,
        frame_id: str,
        camera_id: str,
        capture_timestamp: str,
        is_hazard_frame: bool,
        image_path: Optional[str],
        sidecar_path: Optional[str],
        detections: List[Dict[str, Any]],
    ) -> None:
        """
        Insert or replace one frame row plus all of its detection rows.

        Upserts on frame_id, so calling this again for the same frame_id
        (e.g. during a sync_from_archive() rebuild) simply overwrites the
        previous row rather than creating a duplicate.
        """
        from dashboard.models import LocationContext

        loc = LocationContext.from_camera_id(camera_id)
        recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO frames (
                    frame_id, camera_id, capture_timestamp, is_hazard_frame,
                    image_path, sidecar_path,
                    loc_facility, loc_berth, loc_crane, loc_camera_label, loc_landmark,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(frame_id) DO UPDATE SET
                    camera_id=excluded.camera_id,
                    capture_timestamp=excluded.capture_timestamp,
                    is_hazard_frame=excluded.is_hazard_frame,
                    image_path=excluded.image_path,
                    sidecar_path=excluded.sidecar_path,
                    loc_facility=excluded.loc_facility,
                    loc_berth=excluded.loc_berth,
                    loc_crane=excluded.loc_crane,
                    loc_camera_label=excluded.loc_camera_label,
                    loc_landmark=excluded.loc_landmark,
                    recorded_at=excluded.recorded_at
                """,
                (
                    frame_id, camera_id, capture_timestamp, int(is_hazard_frame),
                    image_path, sidecar_path,
                    loc.facility, loc.berth, loc.crane, loc.camera, loc.landmark,
                    recorded_at,
                ),
            )

            cur.execute("DELETE FROM detections WHERE frame_id = ?", (frame_id,))

            rows = []
            for det in detections:
                bbox = det.get("bbox") or {}
                rows.append((
                    frame_id,
                    det.get("class_label", ""),
                    float(det.get("confidence", 0.0)),
                    bbox.get("x_center"),
                    bbox.get("y_center"),
                    bbox.get("width"),
                    bbox.get("height"),
                    int(bool(det.get("is_hazard", False))),
                    det.get("hazard_reason", ""),
                ))
            if rows:
                cur.executemany(
                    """
                    INSERT INTO detections (
                        frame_id, class_label, confidence,
                        bbox_x_center, bbox_y_center, bbox_width, bbox_height,
                        is_hazard, hazard_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            self._conn.commit()

    # ------------------------------------------------------------------
    # Backfill / rebuild from disk
    # ------------------------------------------------------------------

    def sync_from_archive(self, archive_root: Path | str) -> Dict[str, int]:
        """
        Walk every JSON sidecar under archive_root and (re)populate the
        database from them via record_frame().

        This is what lets the database "auto-generate" — point it at an
        existing live_camera_captures/ directory (e.g. copied from another
        machine, or after deleting capture_log.db to force a rebuild) and
        every frame that has a JSON sidecar gets indexed.

        Returns {"inserted": N, "errors": N} — N files skipped due to a
        parse error or missing expected fields are counted as errors and
        logged, without stopping the rest of the sync.
        """
        root = Path(archive_root)
        inserted = 0
        errors = 0

        if not root.is_dir():
            logger.warning(
                "CaptureDatabase.sync_from_archive: '%s' does not exist — nothing to sync.",
                root,
            )
            return {"inserted": 0, "errors": 0}

        for json_path in sorted(root.rglob("*.json")):
            if json_path.name == DEFAULT_DB_FILENAME:
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "CaptureDatabase.sync_from_archive: failed to read '%s': %s",
                    json_path, exc,
                )
                errors += 1
                continue

            try:
                frame_id = json_path.stem
                camera_id = data.get("camera_id") or json_path.parent.parent.name
                capture_timestamp = data.get("capture_timestamp", "")
                is_hazard_frame = bool(data.get("is_hazard_frame", False))
                detections = data.get("detections", [])

                image_path = json_path.with_suffix(".jpg")
                if not image_path.is_file():
                    logger.warning(
                        "CaptureDatabase.sync_from_archive: sidecar '%s' has no "
                        "matching image at '%s' — recording metadata anyway.",
                        json_path, image_path,
                    )
                    image_path_str = None
                else:
                    image_path_str = str(image_path)

                self.record_frame(
                    frame_id=frame_id,
                    camera_id=camera_id,
                    capture_timestamp=capture_timestamp,
                    is_hazard_frame=is_hazard_frame,
                    image_path=image_path_str,
                    sidecar_path=str(json_path),
                    detections=detections,
                )
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CaptureDatabase.sync_from_archive: failed to record frame "
                    "from '%s': %s: %s",
                    json_path, type(exc).__name__, exc,
                )
                errors += 1

        logger.info(
            "CaptureDatabase.sync_from_archive: synced %d frame(s) from '%s' (%d error(s)).",
            inserted, root, errors,
        )
        return {"inserted": inserted, "errors": errors}

    # ------------------------------------------------------------------
    # Reads — convenience query helpers
    # ------------------------------------------------------------------

    def fetch_frame(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """Return one frame's full record (fields + its detections), or None."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,))
            row = cur.fetchone()
            if row is None:
                return None
            columns = [d[0] for d in cur.description]
            frame = dict(zip(columns, row))

            cur.execute("SELECT * FROM detections WHERE frame_id = ?", (frame_id,))
            det_columns = [d[0] for d in cur.description]
            frame["detections"] = [dict(zip(det_columns, r)) for r in cur.fetchall()]
            return frame

    def fetch_recent_frames(
        self,
        limit: int = 20,
        camera_id: Optional[str] = None,
        hazard_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent frame rows (no detections attached — use
        fetch_frame() for a specific frame's full detection list), newest
        first by capture_timestamp.
        """
        query = "SELECT * FROM frames WHERE 1=1"
        params: List[Any] = []
        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if hazard_only:
            query += " AND is_hazard_frame = 1"
        query += " ORDER BY capture_timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def count_frames(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM frames")
            return cur.fetchone()[0]
