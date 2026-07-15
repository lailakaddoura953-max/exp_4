"""
Unit tests for CameraStub (src/dashboard/camera_stub.py).

Requirements covered: 9.1, 9.2, 9.4, 9.5, 15.1, 15.2

Tests use a small synthetic JPEG written to a temporary directory so they
run without any external dataset and without mocking cv2.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from dashboard.camera_stub import CameraStub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jpeg(directory: str, filename: str = "test.jpg") -> str:
    """Write a tiny valid BGR JPEG to *directory* and return its path."""
    path = os.path.join(directory, filename)
    # 10×10 green image — small but fully valid
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (0, 200, 0)  # BGR green
    cv2.imwrite(path, img)
    return path


def _write_png(directory: str, filename: str = "test.png") -> str:
    """Write a tiny valid BGR PNG to *directory* and return its path."""
    path = os.path.join(directory, filename)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (200, 0, 0)  # BGR blue
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# get_camera_id — Requirement 9.1
# ---------------------------------------------------------------------------

class TestGetCameraId:
    def test_default_camera_id(self):
        """Default camera_id is 'cam_stub_01' (Req 9.1)."""
        stub = CameraStub()
        assert stub.get_camera_id() == "cam_stub_01"

    def test_custom_camera_id(self):
        """Configured camera_id is returned verbatim (Req 9.1)."""
        stub = CameraStub(camera_id="cam_test_99")
        assert stub.get_camera_id() == "cam_test_99"

    def test_camera_id_is_string(self):
        """get_camera_id always returns a str (Req 9.1)."""
        stub = CameraStub(camera_id="my_cam")
        result = stub.get_camera_id()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_test_image — no path set — Requirement 9.5 / 15.2
# ---------------------------------------------------------------------------

class TestGetTestImageNoPath:
    def test_none_when_no_path(self):
        """Returns None silently when test_image_path is None (Req 9.5, 15.2)."""
        stub = CameraStub()
        assert stub.get_test_image() is None

    def test_none_when_explicit_none(self):
        """Returns None when test_image_path is explicitly None."""
        stub = CameraStub(camera_id="cam_stub_01", test_image_path=None)
        assert stub.get_test_image() is None

    def test_no_warning_logged_when_no_path(self, caplog):
        """No WARNING is logged when no path is configured (Req 9.5)."""
        import logging
        stub = CameraStub()
        with caplog.at_level(logging.WARNING, logger="dashboard.camera_stub"):
            stub.get_test_image()
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# get_test_image — file mode — Requirements 9.2, 9.5
# ---------------------------------------------------------------------------

class TestGetTestImageFileMode:
    def test_returns_ndarray_for_valid_jpeg(self):
        """Returns np.ndarray for a valid JPEG file (Req 9.2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_jpeg(tmpdir)
            stub = CameraStub(test_image_path=path)
            result = stub.get_test_image()
            assert isinstance(result, np.ndarray)
            assert result.ndim == 3  # H x W x C

    def test_returns_ndarray_for_valid_png(self):
        """Returns np.ndarray for a valid PNG file (Req 9.2)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_png(tmpdir)
            stub = CameraStub(test_image_path=path)
            result = stub.get_test_image()
            assert isinstance(result, np.ndarray)
            assert result.ndim == 3

    def test_returns_none_for_nonexistent_file(self, caplog):
        """Returns None and logs WARNING for a missing file (Req 9.5)."""
        import logging
        stub = CameraStub(test_image_path="/nonexistent/path/image.jpg")
        with caplog.at_level(logging.WARNING, logger="dashboard.camera_stub"):
            result = stub.get_test_image()
        assert result is None
        assert any("WARNING" in r.levelname or r.levelno >= logging.WARNING
                   for r in caplog.records)

    def test_returns_none_for_corrupt_file(self, caplog, tmp_path):
        """Returns None and logs WARNING when cv2.imread cannot decode the file (Req 9.5)."""
        import logging
        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"this is not an image")
        stub = CameraStub(test_image_path=str(corrupt))
        with caplog.at_level(logging.WARNING, logger="dashboard.camera_stub"):
            result = stub.get_test_image()
        assert result is None
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_does_not_raise_on_unreadable_file(self):
        """Never raises an exception on read failure (Req 9.5)."""
        stub = CameraStub(test_image_path="/nonexistent/image.jpg")
        try:
            result = stub.get_test_image()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"get_test_image raised an exception: {exc}")
        assert result is None


# ---------------------------------------------------------------------------
# get_test_image — directory mode — Requirements 9.2, 9.5, 15.1
# ---------------------------------------------------------------------------

class TestGetTestImageDirectoryMode:
    def test_returns_ndarray_from_directory_with_jpeg(self):
        """Returns np.ndarray when directory contains a JPEG (Req 15.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_jpeg(tmpdir)
            stub = CameraStub(test_image_path=tmpdir)
            result = stub.get_test_image()
            assert isinstance(result, np.ndarray)

    def test_returns_ndarray_from_directory_with_png(self):
        """Returns np.ndarray when directory contains a PNG (Req 15.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_png(tmpdir)
            stub = CameraStub(test_image_path=tmpdir)
            result = stub.get_test_image()
            assert isinstance(result, np.ndarray)

    def test_returns_none_and_warns_for_empty_directory(self, caplog):
        """Returns None + WARNING when directory has no eligible images (Req 9.5)."""
        import logging
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = CameraStub(test_image_path=tmpdir)
            with caplog.at_level(logging.WARNING, logger="dashboard.camera_stub"):
                result = stub.get_test_image()
        assert result is None
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_random_selection_from_multiple_images(self):
        """Randomly picks from available images; result is always a valid ndarray (Req 15.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_jpeg(tmpdir, "img1.jpg")
            _write_jpeg(tmpdir, "img2.jpg")
            _write_png(tmpdir, "img3.png")
            stub = CameraStub(test_image_path=tmpdir)
            for _ in range(10):
                result = stub.get_test_image()
                assert isinstance(result, np.ndarray), \
                    "Expected np.ndarray from directory with multiple images"

    def test_does_not_raise_for_empty_directory(self):
        """Never raises even when the directory is empty (Req 9.5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = CameraStub(test_image_path=tmpdir)
            try:
                result = stub.get_test_image()
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"get_test_image raised an exception: {exc}")
            assert result is None

    def test_nonexistent_directory_logs_warning(self, caplog):
        """Returns None + WARNING when path doesn't exist (Req 9.5)."""
        import logging
        stub = CameraStub(test_image_path="/nonexistent/directory/")
        with caplog.at_level(logging.WARNING, logger="dashboard.camera_stub"):
            result = stub.get_test_image()
        assert result is None
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Interface contract — Requirement 9.4
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    def test_get_camera_id_returns_str(self):
        """get_camera_id() -> str (Req 9.4 contract)."""
        stub = CameraStub()
        assert isinstance(stub.get_camera_id(), str)

    def test_get_test_image_returns_ndarray_or_none(self):
        """get_test_image() -> Optional[np.ndarray] (Req 9.4 contract)."""
        stub = CameraStub()
        result = stub.get_test_image()
        assert result is None or isinstance(result, np.ndarray)

    def test_has_get_camera_id(self):
        """CameraStub exposes get_camera_id (documented interface, Req 9.4)."""
        stub = CameraStub()
        assert callable(getattr(stub, "get_camera_id", None))

    def test_has_get_test_image(self):
        """CameraStub exposes get_test_image (Req 9.2)."""
        stub = CameraStub()
        assert callable(getattr(stub, "get_test_image", None))

    def test_source_file_contains_interface_contract_comment(self):
        """
        The source file must contain a clearly marked comment block documenting
        the real camera interface: get_camera_id, get_frame, is_connected (Req 9.4).
        """
        import inspect
        import dashboard.camera_stub as module
        source = inspect.getsource(module)
        # All three required method signatures must appear in the comment block
        assert "get_camera_id" in source
        assert "get_frame" in source
        assert "is_connected" in source
