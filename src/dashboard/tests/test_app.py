"""
Route tests for src/dashboard/app.py.

Uses the Flask test client with the InferenceEngine and CameraStub patched so
tests run without a real checkpoint file or Roboflow image directory.

Coverage:
  POST /api/inference   — missing image, bad image, successful inference,
                          model-not-loaded (500)
  GET  /api/hazards/recent — empty store, populated store
  GET  /api/status         — all required keys present
  GET  /api/test-image     — image returned, 404 when no image available
"""

from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid JPEG as raw bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (100, 150, 200)  # solid BGR colour
    success, buf = cv2.imencode(".jpg", img)
    assert success, "cv2.imencode failed in test helper"
    return buf.tobytes()


def _make_test_image_array(width: int = 64, height: int = 64) -> np.ndarray:
    """Return a small BGR image array."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (50, 100, 150)
    return img


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


def _make_mock_hazard_result(
    is_hazard: bool = False,
    hazard_reason: str = "",
    class_label: str = "Container - Stacked",
    confidence: float = 0.75,
) -> MagicMock:
    """Return a MagicMock that looks like a HazardResult."""
    from hazard_detection.models import BBox

    bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1)
    result = MagicMock()
    result.is_hazard = is_hazard
    result.hazard_reason = hazard_reason
    result.class_label = class_label
    result.confidence = confidence
    result.bbox = bbox
    result.camera_id = "cam_stub_01"
    result.to_dict.return_value = {
        "class_label": class_label,
        "confidence": confidence,
        "bbox": {
            "x_center": bbox.x_center,
            "y_center": bbox.y_center,
            "width": bbox.width,
            "height": bbox.height,
        },
        "is_hazard": is_hazard,
        "hazard_reason": hazard_reason,
        "camera_id": "cam_stub_01",
    }
    return result


# ---------------------------------------------------------------------------
# Base test class — patches singletons before importing app
# ---------------------------------------------------------------------------


class AppTestBase(unittest.TestCase):
    """
    Base class that patches the module-level singletons so every test starts
    from a clean state without touching the real InferenceEngine or filesystem.
    """

    def setUp(self) -> None:
        # We patch at the module level AFTER the module is already imported,
        # so we target `dashboard.app.<name>` directly.
        import dashboard.app as app_module

        self.app_module = app_module

        # Save originals so we can restore after each test
        self._orig_engine = app_module.inference_engine
        self._orig_model_loaded = app_module.model_loaded
        self._orig_store = app_module.hazard_store
        self._orig_camera = app_module.camera_stub

        # Replace singletons with mocks
        self.mock_engine = MagicMock()
        self.mock_engine.run.return_value = []

        from dashboard.hazard_store import HazardStore

        self.real_store = HazardStore(capacity=20)

        self.mock_camera = MagicMock()
        self.mock_camera.get_camera_id.return_value = "cam_stub_01"
        self.mock_camera.get_test_image.return_value = None

        app_module.inference_engine = self.mock_engine
        app_module.model_loaded = True
        app_module.hazard_store = self.real_store
        app_module.camera_stub = self.mock_camera

        # Flask test client
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        # Restore originals
        self.app_module.inference_engine = self._orig_engine
        self.app_module.model_loaded = self._orig_model_loaded
        self.app_module.hazard_store = self._orig_store
        self.app_module.camera_stub = self._orig_camera


# ---------------------------------------------------------------------------
# POST /api/inference tests
# ---------------------------------------------------------------------------


class TestInferenceRoute(AppTestBase):

    def test_missing_image_returns_400(self):
        """POST without 'image' field → 400 {"error": "No image provided"}."""
        resp = self.client.post("/api/inference")
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertEqual(body["error"], "No image provided")

    def test_invalid_image_bytes_returns_400(self):
        """POST with non-image bytes → 400 {"error": "Invalid or unreadable image"}."""
        resp = self.client.post(
            "/api/inference",
            data={"image": (io.BytesIO(b"not an image"), "test.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertEqual(body["error"], "Invalid or unreadable image")

    def test_model_not_loaded_returns_500(self):
        """POST when model_loaded=False → 500 {"error": "Model not loaded"}."""
        self.app_module.inference_engine = None
        self.app_module.model_loaded = False

        jpeg = _make_jpeg_bytes()
        resp = self.client.post(
            "/api/inference",
            data={"image": (io.BytesIO(jpeg), "test.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 500)
        body = resp.get_json()
        self.assertEqual(body["error"], "Model not loaded")

    def test_successful_inference_no_hazards(self):
        """
        POST with valid JPEG, engine returns no hazards →
        200, results=[], annotated_image key present.
        """
        self.mock_engine.run.return_value = []

        # Patch annotate to return a dummy base64 string
        dummy_b64 = base64.b64encode(b"fakeimage").decode()
        with patch("dashboard.app.annotate", return_value=dummy_b64):
            jpeg = _make_jpeg_bytes()
            resp = self.client.post(
                "/api/inference",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("results", body)
        self.assertIn("annotated_image", body)
        self.assertEqual(body["results"], [])

    def test_successful_inference_with_hazard_stores_event(self):
        """
        POST with valid JPEG, engine returns one hazard result →
        event is appended to hazard_store.
        """
        mock_result = _make_mock_hazard_result(
            is_hazard=True,
            hazard_reason="ppe_violation",
            class_label="Human - No Safety Clothes",
            confidence=0.91,
        )
        self.mock_engine.run.return_value = [mock_result]

        dummy_b64 = base64.b64encode(b"fakeimage").decode()
        with patch("dashboard.app.annotate", return_value=dummy_b64):
            jpeg = _make_jpeg_bytes()
            resp = self.client.post(
                "/api/inference",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(len(body["results"]), 1)
        self.assertTrue(body["results"][0]["is_hazard"])

        # Event stored in hazard_store
        self.assertEqual(self.real_store.count(), 1)
        event = self.real_store.get_recent(1)[0]
        self.assertEqual(event.hazard_type, "ppe_violation")
        self.assertEqual(event.camera_id, "cam_stub_01")

    def test_annotation_failure_sets_null_annotated_image(self):
        """
        If annotate() raises, annotated_image in response is null.
        """
        self.mock_engine.run.return_value = []

        with patch("dashboard.app.annotate", side_effect=RuntimeError("boom")):
            jpeg = _make_jpeg_bytes()
            resp = self.client.post(
                "/api/inference",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsNone(body["annotated_image"])

    def test_custom_camera_id_forwarded_to_engine(self):
        """
        camera_id form field is passed to inference_engine.run().
        """
        self.mock_engine.run.return_value = []

        with patch("dashboard.app.annotate", return_value=None):
            jpeg = _make_jpeg_bytes()
            self.client.post(
                "/api/inference",
                data={
                    "image": (io.BytesIO(jpeg), "test.jpg"),
                    "camera_id": "cam_stub_07",
                },
                content_type="multipart/form-data",
            )

        call_args = self.mock_engine.run.call_args
        # Second positional arg is camera_id
        self.assertEqual(call_args[0][1], "cam_stub_07")

    def test_engine_exception_returns_500(self):
        """
        If inference_engine.run() raises, route returns HTTP 500.
        """
        self.mock_engine.run.side_effect = RuntimeError("engine exploded")

        jpeg = _make_jpeg_bytes()
        resp = self.client.post(
            "/api/inference",
            data={"image": (io.BytesIO(jpeg), "test.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(resp.status_code, 500)
        body = resp.get_json()
        self.assertIn("error", body)

    def test_non_hazard_results_not_stored(self):
        """
        Results with is_hazard=False are returned but not stored in HazardStore.
        """
        mock_result = _make_mock_hazard_result(is_hazard=False)
        self.mock_engine.run.return_value = [mock_result]

        with patch("dashboard.app.annotate", return_value=None):
            jpeg = _make_jpeg_bytes()
            self.client.post(
                "/api/inference",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )

        self.assertEqual(self.real_store.count(), 0)


# ---------------------------------------------------------------------------
# GET /api/hazards/recent tests
# ---------------------------------------------------------------------------


class TestHazardsRecentRoute(AppTestBase):

    def test_empty_store_returns_empty_list(self):
        """GET /api/hazards/recent when store is empty → 200, []."""
        resp = self.client.get("/api/hazards/recent")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, list)
        self.assertEqual(body, [])

    def test_returns_at_most_3_events(self):
        """
        GET /api/hazards/recent returns at most 3 events even if store has more.
        """
        from dashboard.models import HazardEvent, LocationContext
        from hazard_detection.models import BBox

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1)
        for i in range(5):
            event = HazardEvent(
                event_id=f"event-{i}",
                hazard_type="ppe_violation",
                camera_id="cam_stub_01",
                timestamp="2025-01-01T00:00:00Z",
                confidence=0.9,
                bbox=bbox,
                annotated_image=None,
                location=LocationContext.from_camera_id("cam_stub_01"),
            )
            self.real_store.append(event)

        resp = self.client.get("/api/hazards/recent")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIsInstance(body, list)
        self.assertLessEqual(len(body), 3)

    def test_returns_newest_first(self):
        """
        Events are returned newest-first (last appended = first in list).
        """
        from dashboard.models import HazardEvent, LocationContext
        from hazard_detection.models import BBox

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1)
        for i in range(3):
            event = HazardEvent(
                event_id=f"event-{i}",
                hazard_type=f"type_{i}",
                camera_id="cam_stub_01",
                timestamp="2025-01-01T00:00:00Z",
                confidence=0.8,
                bbox=bbox,
                annotated_image=None,
                location=LocationContext.from_camera_id("cam_stub_01"),
            )
            self.real_store.append(event)

        resp = self.client.get("/api/hazards/recent")
        body = resp.get_json()
        # Newest (last appended) should be first
        self.assertEqual(body[0]["hazard_type"], "type_2")
        self.assertEqual(body[1]["hazard_type"], "type_1")
        self.assertEqual(body[2]["hazard_type"], "type_0")


# ---------------------------------------------------------------------------
# GET /api/status tests
# ---------------------------------------------------------------------------


class TestStatusRoute(AppTestBase):

    def test_status_returns_200_with_required_keys(self):
        """GET /api/status → 200 with status, model_loaded, hazard_count, camera_id."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "running")
        self.assertIn("model_loaded", body)
        self.assertIn("hazard_count", body)
        self.assertIn("camera_id", body)

    def test_status_model_loaded_true_when_engine_available(self):
        """model_loaded is True when inference_engine is not None."""
        self.app_module.model_loaded = True
        resp = self.client.get("/api/status")
        body = resp.get_json()
        self.assertTrue(body["model_loaded"])

    def test_status_model_loaded_false_when_engine_none(self):
        """model_loaded is False when inference_engine is None."""
        self.app_module.model_loaded = False
        resp = self.client.get("/api/status")
        body = resp.get_json()
        self.assertFalse(body["model_loaded"])

    def test_status_hazard_count_reflects_store(self):
        """hazard_count in status response matches hazard_store.count()."""
        from dashboard.models import HazardEvent, LocationContext
        from hazard_detection.models import BBox

        bbox = BBox(x_center=0.5, y_center=0.5, width=0.2, height=0.1)
        for i in range(2):
            self.real_store.append(
                HazardEvent(
                    event_id=f"e-{i}",
                    hazard_type="ppe_violation",
                    camera_id="cam_stub_01",
                    timestamp="2025-01-01T00:00:00Z",
                    confidence=0.9,
                    bbox=bbox,
                    annotated_image=None,
                    location=LocationContext.from_camera_id("cam_stub_01"),
                )
            )

        resp = self.client.get("/api/status")
        body = resp.get_json()
        self.assertEqual(body["hazard_count"], 2)

    def test_status_camera_id_from_camera_stub(self):
        """camera_id in response comes from camera_stub.get_camera_id()."""
        self.mock_camera.get_camera_id.return_value = "cam_stub_07"
        resp = self.client.get("/api/status")
        body = resp.get_json()
        self.assertEqual(body["camera_id"], "cam_stub_07")


# ---------------------------------------------------------------------------
# GET /api/test-image tests
# ---------------------------------------------------------------------------


class TestTestImageRoute(AppTestBase):

    def test_returns_jpeg_when_image_available(self):
        """GET /api/test-image returns image/jpeg when CameraStub returns an image."""
        self.mock_camera.get_test_image.return_value = _make_test_image_array()
        resp = self.client.get("/api/test-image")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("image/jpeg", resp.content_type)
        # Response data should be non-empty bytes that decode as a valid JPEG
        data = resp.data
        self.assertGreater(len(data), 0)
        arr = np.frombuffer(data, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)

    def test_returns_404_when_no_image(self):
        """GET /api/test-image returns 404 when both primary and fallback CameraStub return None."""
        self.mock_camera.get_test_image.return_value = None
        # The fallback path also creates a CameraStub; patch CameraStub so the
        # fallback instance also returns None, simulating no images available.
        fallback_mock = MagicMock()
        fallback_mock.get_camera_id.return_value = "cam_stub_01"
        fallback_mock.get_test_image.return_value = None
        with patch("dashboard.app.CameraStub", return_value=fallback_mock):
            resp = self.client.get("/api/test-image")
        self.assertEqual(resp.status_code, 404)
        body = resp.get_json()
        self.assertIn("error", body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
