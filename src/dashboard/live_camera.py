"""
Live Camera Capture — RTSP burst acquisition for real Wisenet cameras.

Opens a short-lived RTSP connection (via OpenCV, same pattern as
FrameAcquisitionModule) to a live camera, grabs a fixed-size burst of
frames in quick succession, and runs each frame through the existing
InferenceEngine — the same engine already used for dataset auto-cycle
and manual upload.

This module intentionally does NOT reuse FrameAcquisitionModule directly:
that module is hard-coded to require exactly 4 cameras (camera_id 0-3),
which is the wrong shape for "one live test camera, burst on demand or
hourly". A dedicated single-camera RTSP burst capture is a much better
fit and keeps FrameAcquisitionModule's 4-camera contract untouched for
its intended production use.

Credentials live in config/ip_addresses.json (gitignored — never
committed). config/ip_addresses_template.json documents the expected
shape for setup on a new machine.

Design notes:
- Connections are opened fresh for each burst and released immediately
  afterward, rather than held open between bursts. For an hourly/on-demand
  cadence this is simpler and more robust than managing a long-lived
  RTSP session (Wisenet cameras will happily accept a new short session
  per burst; keeping one open for an hour risks silent stream death going
  undetected between uses).
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from hazard_detection.diagnostics import get_logger

logger = get_logger("live_camera")

# Default location of the (gitignored) real credentials file.
DEFAULT_IP_ADDRESSES_PATH = "config/ip_addresses.json"

# Number of frames captured per burst (Requirement from user: "5 snapshots
# in a row"). Kept as a module constant rather than user-configurable to
# match the explicit ask.
BURST_SIZE = 5

# Minimum delay between consecutive frame reads within a burst. RTSP
# streams typically run at >=15fps; a small sleep avoids hammering
# cv2.VideoCapture.read() faster than the camera can deliver new frames.
INTER_FRAME_DELAY_SECONDS = 0.15

# Per-connection timeout for opening the RTSP stream before giving up.
CONNECT_TIMEOUT_SECONDS = 8.0

# Hourly floor for the scheduled auto-capture cadence. Per explicit
# instruction, the production timer must never run more frequently than
# hourly (60 minutes) — a lower value is clamped up to this floor rather
# than honored, since sub-hourly bursts on a single live camera were
# specifically ruled out.
MIN_AUTO_CAPTURE_INTERVAL_MINUTES = 60

# Default root directory for archived live-camera frames. Unlike
# HazardStore (in-memory, hazard-only, capped at 20 events), every frame
# from every burst is written here regardless of whether it was flagged
# as a hazard — this system intentionally keeps all captured data rather
# than discarding non-hazard frames, unlike a "standard" hazard-only
# retention approach.
DEFAULT_ARCHIVE_DIR = "live_camera_captures"


@dataclass
class RTSPCameraConfig:
    """Connection details for one RTSP-capable camera."""

    camera_id: str
    ip: str
    rtsp_port: int = 554
    username: str = ""
    password: str = ""
    profile: str = "profile2"
    location_id: Optional[int] = None

    def build_url(self) -> str:
        """
        Build the RTSP URL for this camera.

        Format follows Hanwha Wisenet's documented convention:
            rtsp://<user>:<password>@<ip>:<port>/<profile>/media.smp

        Username/password are URL-encoded so special characters in the
        password (recommended by Hanwha's complexity rules — e.g. '!')
        don't break URL parsing.
        """
        user = urllib.parse.quote(self.username, safe="")
        pwd = urllib.parse.quote(self.password, safe="")
        auth = f"{user}:{pwd}@" if user or pwd else ""
        return f"rtsp://{auth}{self.ip}:{self.rtsp_port}/{self.profile}/media.smp"

    def safe_repr(self) -> str:
        """Return a connection description with the password redacted, for logging."""
        return f"rtsp://{self.username}:***@{self.ip}:{self.rtsp_port}/{self.profile}/media.smp"


def load_camera_config(
    path: str = DEFAULT_IP_ADDRESSES_PATH,
) -> Optional[RTSPCameraConfig]:
    """
    Load the first camera entry from config/ip_addresses.json.

    Returns None (rather than raising) when the file is missing or
    malformed, so the dashboard can start in degraded mode — mirroring
    the existing model_loaded / checkpoint-missing pattern in app.py.

    Only the first camera in the "cameras" mapping is used today; this
    matches the current single-live-test-camera scope.
    """
    config_path = Path(path)
    if not config_path.is_file():
        logger.warning(
            "Live camera config not found at '%s'. Copy "
            "config/ip_addresses_template.json to config/ip_addresses.json "
            "and fill in real credentials to enable live camera capture.",
            path,
        )
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load live camera config '%s': %s", path, exc)
        return None

    cameras = raw.get("cameras", {})
    if not cameras:
        logger.warning("Live camera config '%s' has no cameras defined.", path)
        return None

    first_key = next(iter(cameras))
    entry = cameras[first_key]

    try:
        return RTSPCameraConfig(
            camera_id=entry.get("camera_id", first_key),
            ip=entry["ip"],
            rtsp_port=int(entry.get("rtsp_port", 554)),
            username=entry.get("username", ""),
            password=entry.get("password", ""),
            profile=entry.get("profile", "profile2"),
            location_id=entry.get("location_id"),
        )
    except KeyError as exc:
        logger.error(
            "Live camera config entry '%s' is missing required field: %s",
            first_key,
            exc,
        )
        return None


class LiveCameraCapture:
    """
    Opens a short-lived RTSP connection and captures a fixed-size burst
    of frames from a single camera.
    """

    def __init__(
        self,
        config: RTSPCameraConfig,
        burst_size: int = BURST_SIZE,
        connect_timeout_seconds: float = CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self._config = config
        self._burst_size = burst_size
        self._connect_timeout_seconds = connect_timeout_seconds

    @property
    def config(self) -> RTSPCameraConfig:
        return self._config

    def capture_burst(self) -> "BurstCaptureResult":
        """
        Open the RTSP stream, capture ``burst_size`` frames in quick
        succession, then release the connection.

        Returns a BurstCaptureResult — always, even on failure — so
        callers get a structured record of what happened (connection
        error, partial burst, or full success) rather than a bare None.
        """
        camera_id = self._config.camera_id
        url = self._config.build_url()

        logger.info(
            "Opening RTSP connection for camera '%s' -> %s",
            camera_id,
            self._config.safe_repr(),
        )

        cap = cv2.VideoCapture(url)
        # Reduce internal buffering so .read() returns the *latest* frame
        # rather than a stale queued one — important for a burst capture
        # where we want 5 genuinely-new snapshots, not 5 copies of frame 1.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 — not all backends support this
            pass

        connect_start = time.perf_counter()
        opened = cap.isOpened()
        while not opened and (time.perf_counter() - connect_start) < self._connect_timeout_seconds:
            time.sleep(0.2)
            opened = cap.isOpened()

        if not opened:
            elapsed = time.perf_counter() - connect_start
            logger.error(
                "Failed to open RTSP stream for camera '%s' after %.1fs "
                "(connection timeout / auth failure / camera offline).",
                camera_id,
                elapsed,
            )
            cap.release()
            return BurstCaptureResult(
                camera_id=camera_id,
                frames=[],
                capture_timestamps=[],
                connection_error=(
                    f"Could not open RTSP stream at {self._config.safe_repr()}. "
                    "Check camera power/network connection and credentials."
                ),
            )

        logger.info(
            "RTSP connection established for camera '%s'. Capturing %d-frame burst...",
            camera_id,
            self._burst_size,
        )

        frames: List[np.ndarray] = []
        timestamps: List[str] = []

        for i in range(self._burst_size):
            ret, frame = cap.read()
            capture_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            if ret and frame is not None:
                frames.append(frame)
                timestamps.append(capture_time)
                logger.info(
                    "Camera '%s': captured frame %d/%d (%dx%d) at %s",
                    camera_id,
                    i + 1,
                    self._burst_size,
                    frame.shape[1],
                    frame.shape[0],
                    capture_time,
                )
            else:
                logger.warning(
                    "Camera '%s': frame %d/%d read failed — stream may have "
                    "dropped mid-burst.",
                    camera_id,
                    i + 1,
                    self._burst_size,
                )

            if i < self._burst_size - 1:
                time.sleep(INTER_FRAME_DELAY_SECONDS)

        cap.release()

        logger.info(
            "RTSP connection closed for camera '%s'. Captured %d/%d frames.",
            camera_id,
            len(frames),
            self._burst_size,
        )

        connection_error = None
        if not frames:
            connection_error = (
                "Connected to the camera but every frame read failed. "
                "The stream may be unstable or the profile/codec unsupported."
            )

        return BurstCaptureResult(
            camera_id=camera_id,
            frames=frames,
            capture_timestamps=timestamps,
            connection_error=connection_error,
        )


@dataclass
class BurstCaptureResult:
    """Raw result of a single RTSP burst capture attempt (pre-inference)."""

    camera_id: str
    frames: List[np.ndarray]
    capture_timestamps: List[str]
    connection_error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.connection_error is None and len(self.frames) > 0


def sync_archive_to_db(archive_root: str = DEFAULT_ARCHIVE_DIR) -> Dict[str, int]:
    """
    Rebuild/backfill capture_log.db from whatever JSON sidecars already
    exist on disk under archive_root.

    This is the "auto-generate" half of the SQLite integration: rather
    than requiring a manual import step, the database and its tables are
    created automatically (see CaptureDatabase._init_schema()) and then
    populated from every JSON sidecar file already present — covering the
    case where the DB file doesn't exist yet, was deleted, or the
    live_camera_captures/ directory was copied over from another machine.
    Safe to call repeatedly (each frame is upserted by its frame_id, so
    re-running never creates duplicates).

    Returns a dict: {"inserted": N, "errors": N}.
    """
    from dashboard.capture_db import CaptureDatabase, DEFAULT_DB_FILENAME

    db_path = Path(archive_root) / DEFAULT_DB_FILENAME
    db = CaptureDatabase(db_path)
    return db.sync_from_archive(archive_root)


class LiveCaptureArchiver:
    """
    Persists every captured frame to disk, unconditionally — hazard or not.

    This is the key difference from a "standard" hazard-detection system:
    those typically discard non-hazard frames and only retain flagged
    events. Here, every frame from every burst is written to disk so the
    full capture history is available later (e.g. for retraining, audit,
    or reviewing what a "no hazard" reading actually looked like).

    Directory layout (created lazily, on first save — see _ensure_dir()):
        <archive_root>/<camera_id>/<YYYY-MM-DD>/
            <burst_timestamp>_frame<N>.jpg   — the raw captured frame
            <burst_timestamp>_frame<N>.json  — sidecar: detections, hazard
                                                flags, confidence, timestamps

    Nothing here ever deletes or overwrites existing files — disk usage
    will grow without bound over time. That's intentional per the "keep
    everything" requirement; monitor available disk space accordingly.
    """

    def __init__(self, archive_root: str = DEFAULT_ARCHIVE_DIR) -> None:
        self._archive_root = Path(archive_root)
        # Lazily constructed on first save_frame() call — see _get_db().
        # Deferred rather than built here so simply constructing an
        # archiver (e.g. in a test) never touches disk until a frame is
        # actually saved.
        self._db: "CaptureDatabase | None" = None

    @property
    def archive_root(self) -> Path:
        return self._archive_root

    def _get_db(self) -> "CaptureDatabase":
        """
        Return the CaptureDatabase for this archive, constructing it (and
        its parent directory, and its schema) on first use.

        The .db file lives inside archive_root itself
        (<archive_root>/capture_log.db) so the SQLite file travels with the
        JPEG/JSON files it indexes if this directory is copied elsewhere.
        """
        if self._db is None:
            from dashboard.capture_db import CaptureDatabase, DEFAULT_DB_FILENAME

            self._db = CaptureDatabase(self._archive_root / DEFAULT_DB_FILENAME)
        return self._db

    def _ensure_dir(self, camera_id: str, day: str) -> Path:
        """
        Return the directory for this camera_id + day, creating it (and
        any missing parents, including the archive root itself) if it
        doesn't already exist yet.

        This is what makes the archive directory appear automatically the
        first time a burst is captured, rather than requiring it to be
        created manually ahead of time.
        """
        day_dir = self._archive_root / camera_id / day
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir

    def save_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        capture_timestamp: str,
        frame_index: int,
        detections: List[Dict[str, Any]],
        is_hazard_frame: bool,
    ) -> Optional[str]:
        """
        Save one frame (JPEG) + its metadata (JSON sidecar) to disk.

        Called once per captured frame, regardless of whether that frame
        contains a hazard — every frame is written the same way.

        Args:
            camera_id: Camera identifier, used as a subdirectory name.
            frame: Raw BGR frame as captured (not the annotated version —
                   this preserves the original pixels for future reuse,
                   e.g. retraining).
            capture_timestamp: ISO 8601 UTC timestamp string for this frame.
            frame_index: 0-based index of this frame within its burst.
            detections: List of detection dicts (from HazardResult.to_dict())
                        for this frame.
            is_hazard_frame: True if any detection in this frame was a hazard.

        Returns:
            Path to the saved JPEG as a string, or None if saving failed
            (logged as a warning; never raises — a disk write failure
            should not interrupt the capture/inference pipeline).
        """
        # Directory day-bucket derived from the frame's own capture time,
        # not wall-clock "now" — keeps a burst that straddles midnight UTC
        # filed under the day each frame actually belongs to.
        try:
            day = capture_timestamp[:10]  # "YYYY-MM-DD" prefix of ISO 8601
        except Exception:  # noqa: BLE001
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            day_dir = self._ensure_dir(camera_id, day)
        except OSError as exc:
            logger.warning(
                "LiveCaptureArchiver: failed to create archive directory "
                "for camera '%s': %s. Frame not saved.",
                camera_id,
                exc,
            )
            return None

        # Filesystem-safe timestamp for the filename (colons aren't valid
        # on Windows paths).
        safe_ts = capture_timestamp.replace(":", "-")
        base_name = f"{safe_ts}_frame{frame_index}"
        image_path = day_dir / f"{base_name}.jpg"
        sidecar_path = day_dir / f"{base_name}.json"

        try:
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning(
                    "LiveCaptureArchiver: cv2.imencode failed for camera "
                    "'%s' frame %d — frame not saved.",
                    camera_id,
                    frame_index,
                )
                return None
            image_path.write_bytes(buffer.tobytes())
        except OSError as exc:
            logger.warning(
                "LiveCaptureArchiver: failed to write frame image for "
                "camera '%s' frame %d: %s",
                camera_id,
                frame_index,
                exc,
            )
            return None

        try:
            sidecar = {
                "camera_id": camera_id,
                "frame_index": frame_index,
                "capture_timestamp": capture_timestamp,
                "is_hazard_frame": is_hazard_frame,
                "detections": detections,
            }
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "LiveCaptureArchiver: failed to write metadata sidecar for "
                "camera '%s' frame %d: %s (image was still saved)",
                camera_id,
                frame_index,
                exc,
            )

        # Mirror the same frame + detections into capture_log.db, keyed by
        # the frame's own filename stem (base_name) — per the user's
        # choice to use the snapshot's timestamp-derived name as its ID,
        # since it's already unique and already ties back to the exact
        # .jpg/.json pair on disk. A DB write failure never blocks the
        # file-based archive above; it's logged and the JPEG/JSON files
        # (the source of truth) are already safely on disk either way.
        try:
            self._get_db().record_frame(
                frame_id=base_name,
                camera_id=camera_id,
                capture_timestamp=capture_timestamp,
                is_hazard_frame=is_hazard_frame,
                image_path=str(image_path),
                sidecar_path=str(sidecar_path),
                detections=detections,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LiveCaptureArchiver: failed to record frame '%s' in "
                "capture_log.db: %s: %s",
                base_name,
                type(exc).__name__,
                exc,
            )

        logger.info(
            "LiveCaptureArchiver: saved frame -> %s%s",
            image_path,
            " [hazard]" if is_hazard_frame else "",
        )
        return str(image_path)


class LiveCaptureService:
    """
    Ties LiveCameraCapture together with the existing InferenceEngine and
    annotate() helper: capture a burst, run inference on every frame, log
    each step to the terminal, and produce a single structured result
    dict consumed by both the Flask API and (optionally) HazardStore.

    This is the "camera to computer vision label" seam the terminal logs
    and the dashboard's /api/live-camera/status endpoint both draw from.
    """

    def __init__(
        self,
        camera_config: RTSPCameraConfig,
        inference_engine: Any,
        hazard_store: Any = None,
        burst_size: int = BURST_SIZE,
        archiver: Optional["LiveCaptureArchiver"] = None,
    ) -> None:
        self._camera_config = camera_config
        self._inference_engine = inference_engine
        self._hazard_store = hazard_store
        self._capture = LiveCameraCapture(camera_config, burst_size=burst_size)
        # Defaults to DEFAULT_ARCHIVE_DIR ("live_camera_captures/") when not
        # explicitly provided — every burst run through this service saves
        # all its frames to disk, hazard or not (see LiveCaptureArchiver).
        self._archiver = archiver or LiveCaptureArchiver()

    def run_burst(self) -> Dict[str, Any]:
        """
        Execute one full burst-capture-and-infer cycle.

        Returns a JSON-serialisable dict:
            {
              "camera_id": str,
              "timestamp": iso8601,
              "success": bool,
              "connection_error": str | None,
              "frames": [
                  {"frame_index", "capture_timestamp", "annotated_image",
                   "detections": [...]},
                  ...
              ],
              "hazards_found": int,
            }
        """
        from dashboard.annotator import annotate  # local import avoids any cycle risk

        camera_id = self._camera_config.camera_id
        run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "=" * 60,
        )
        logger.info(
            "LIVE CAMERA BURST — camera_id='%s', triggered at %s",
            camera_id,
            run_timestamp,
        )
        logger.info("=" * 60)

        burst = self._capture.capture_burst()

        if not burst.success:
            logger.error(
                "Live camera burst failed for '%s': %s",
                camera_id,
                burst.connection_error,
            )
            return {
                "camera_id": camera_id,
                "location_id": self._camera_config.location_id,
                "timestamp": run_timestamp,
                "success": False,
                "connection_error": burst.connection_error,
                "frames": [],
                "hazards_found": 0,
            }

        frame_results: List[Dict[str, Any]] = []
        hazards_found = 0

        for idx, (frame, capture_ts) in enumerate(
            zip(burst.frames, burst.capture_timestamps)
        ):
            logger.info(
                "Camera '%s' frame %d/%d -> running InferenceEngine...",
                camera_id,
                idx + 1,
                len(burst.frames),
            )

            try:
                results = self._inference_engine.run(frame, camera_id=camera_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "InferenceEngine.run failed for camera '%s' frame %d: %s: %s",
                    camera_id,
                    idx + 1,
                    type(exc).__name__,
                    exc,
                )
                results = []

            frame_hazards = [r for r in results if r.is_hazard]
            hazards_found += len(frame_hazards)

            if results:
                logger.info(
                    "Camera '%s' frame %d/%d -> %d detection(s), %d hazard(s): %s",
                    camera_id,
                    idx + 1,
                    len(burst.frames),
                    len(results),
                    len(frame_hazards),
                    ", ".join(
                        f"{r.class_label}({r.confidence:.0%})"
                        + (" [HAZARD]" if r.is_hazard else "")
                        for r in results
                    ),
                )
            else:
                logger.info(
                    "Camera '%s' frame %d/%d -> no detections above threshold",
                    camera_id,
                    idx + 1,
                    len(burst.frames),
                )

            try:
                annotated_b64 = annotate(frame, results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("annotate() failed for frame %d: %s", idx + 1, exc)
                annotated_b64 = None

            # Persist hazard events, same as the dashboard auto-cycle path
            if self._hazard_store is not None:
                for r in frame_hazards:
                    self._store_hazard_event(r, camera_id, capture_ts, annotated_b64)

            # Archive the raw frame to disk unconditionally — hazard or
            # not. This is the "keep everything" behavior: unlike
            # HazardStore above (hazard-only, in-memory, capped), every
            # frame from every burst lands on disk here. The archive
            # directory is created automatically on first use if it
            # doesn't exist yet (see LiveCaptureArchiver._ensure_dir()).
            try:
                self._archiver.save_frame(
                    camera_id=camera_id,
                    frame=frame,
                    capture_timestamp=capture_ts,
                    frame_index=idx,
                    detections=[r.to_dict() for r in results],
                    is_hazard_frame=len(frame_hazards) > 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Archiving failed for camera '%s' frame %d: %s: %s",
                    camera_id,
                    idx + 1,
                    type(exc).__name__,
                    exc,
                )

            frame_results.append(
                {
                    "frame_index": idx,
                    "capture_timestamp": capture_ts,
                    "annotated_image": annotated_b64,
                    "detections": [r.to_dict() for r in results],
                }
            )

        logger.info(
            "Live camera burst complete for '%s': %d frame(s) processed, "
            "%d hazard(s) total.",
            camera_id,
            len(frame_results),
            hazards_found,
        )
        logger.info("=" * 60)

        return {
            "camera_id": camera_id,
            "location_id": self._camera_config.location_id,
            "timestamp": run_timestamp,
            "success": True,
            "connection_error": None,
            "frames": frame_results,
            "hazards_found": hazards_found,
        }

    def _store_hazard_event(
        self,
        result: Any,
        camera_id: str,
        capture_ts: str,
        annotated_b64: Optional[str],
    ) -> None:
        """Append a HazardEvent to the shared HazardStore for a hazardous result."""
        import uuid

        from dashboard.models import HazardEvent, LocationContext

        try:
            event = HazardEvent(
                event_id=str(uuid.uuid4()),
                hazard_type=result.hazard_reason,
                camera_id=camera_id,
                timestamp=capture_ts,
                confidence=result.confidence,
                bbox=result.bbox,
                annotated_image=annotated_b64,
                location=LocationContext.from_camera_id(camera_id),
            )
            self._hazard_store.append(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to store hazard event for camera '%s': %s", camera_id, exc
            )
