"""
Unit tests for src/dashboard/frame_source.py (FrameSourceManager).

Requirements covered: 1.1, 1.3, 1.7, 3.5, 3.6
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dashboard.frame_source import (
    FrameSourceManager,
    _resolve_location,
    load_map_config,
)


def _write_fake_image(path: Path) -> None:
    """Write a minimal valid PNG-ish file that cv2.imread can actually decode."""
    import numpy as np
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def synth_fixture(tmp_path) -> Path:
    """Build a small mock image_data_with_synth/ tree."""
    root = tmp_path / "image_data_with_synth"

    _write_fake_image(root / "augmented_hazards" / "berth_401" / "day" / "img_001.PNG")
    _write_fake_image(root / "augmented_hazards" / "TEL_1-35" / "night" / "img_002.PNG")
    _write_fake_image(root / "normal_operations" / "augmented_normal" / "berth_405_and_knuckle" / "day" / "img_003.PNG")
    _write_fake_image(root / "normal_operations" / "auto_accepted" / "E2_C_PTZ_Pedestal" / "img_004.PNG")
    _write_fake_image(root / "normal_operations" / "auto_accepted" / "unknown_location_x" / "img_005.PNG")

    return root


@pytest.fixture
def fallback_fixture(tmp_path) -> Path:
    """Build a small mock roboflow data/ tree."""
    root = tmp_path / "roboflow data"
    _write_fake_image(root / "test" / "images" / "test_001.jpg")
    _write_fake_image(root / "train" / "images" / "train_001.jpg")
    return root


@pytest.fixture
def map_config() -> dict:
    return {
        "folder_to_location": {
            "berth_401": 10,
            "berth_405_and_knuckle": 16,
            "E2_C_PTZ_Pedestal": 8,
            "E10_East_PTZ_Wall_st": 8,
            "TEL_1-35": 9,
            "TEL_144": 9,
        },
        "default_location_range": [1, 16],
    }


class TestSynthSourceActive:
    def test_discovers_all_buckets(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        assert mgr.is_using_synth() is True
        assert mgr._frame_count() == 5

    def test_confirmed_folders_map_correctly(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        # Get all frames and check their locations
        locations_found = set()
        for i in range(mgr._frame_count()):
            frame = mgr._load_frame(i)
            if frame and frame.folder_name in map_config["folder_to_location"]:
                assert frame.map_location == map_config["folder_to_location"][frame.folder_name]
                locations_found.add(frame.folder_name)
        assert "berth_401" in locations_found
        assert "TEL_1-35" in locations_found

    def test_unmapped_folder_gets_deterministic_location(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        # Find the unmapped folder frame
        for i in range(mgr._frame_count()):
            frame = mgr._load_frame(i)
            if frame and frame.folder_name == "unknown_location_x":
                loc1 = frame.map_location
                break
        else:
            pytest.fail("Unmapped folder frame not found")

        # Same seed, same folder → same location
        assert 1 <= loc1 <= 16
        assert loc1 == _resolve_location("unknown_location_x", {}, [1, 16])

    def test_is_synthetic_true_for_synth_frames(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        frame = mgr.get_current_frame()
        assert frame is not None
        assert frame.is_synthetic is True

    def test_disclaimer_present_when_synth(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        assert "not live footage" in mgr.source_disclaimer


class TestFallbackActive:
    def test_synth_missing_falls_back(self, tmp_path, fallback_fixture, map_config):
        nonexistent = tmp_path / "does_not_exist"
        mgr = FrameSourceManager(
            synth_dir=nonexistent,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        assert mgr.is_using_synth() is False
        assert mgr._frame_count() == 2

    def test_fallback_frames_not_synthetic(self, tmp_path, fallback_fixture, map_config):
        nonexistent = tmp_path / "does_not_exist"
        mgr = FrameSourceManager(
            synth_dir=nonexistent,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        frame = mgr.get_current_frame()
        assert frame is not None
        assert frame.is_synthetic is False

    def test_disclaimer_empty_for_fallback(self, tmp_path, fallback_fixture, map_config):
        nonexistent = tmp_path / "does_not_exist"
        mgr = FrameSourceManager(
            synth_dir=nonexistent,
            fallback_dir=fallback_fixture,
            map_config=map_config,
        )
        assert mgr.source_disclaimer == ""


class TestCycleAdvance:
    def test_advance_now_changes_frame(self, synth_fixture, fallback_fixture, map_config):
        mgr = FrameSourceManager(
            synth_dir=synth_fixture,
            fallback_dir=fallback_fixture,
            map_config=map_config,
            cycle_interval_seconds=9999,  # won't auto-advance
        )
        frame1 = mgr.get_current_frame()
        mgr.advance_now()
        frame2 = mgr.get_current_frame()
        # With 5 images in queue, advancing should (almost certainly) give a different frame
        # unless we're extremely unlucky with shuffle — but deterministic seed makes this reliable
        assert frame1 is not None and frame2 is not None
        # They should differ (with seed=42 and 5 items, consecutive queue positions differ)
        assert frame1.source_path != frame2.source_path or mgr._frame_count() == 1


class TestLoadMapConfig:
    def test_loads_existing_json(self, tmp_path):
        import json
        cfg_path = tmp_path / "dashboard_map.json"
        data = {"folder_to_location": {"test": 5}, "default_location_range": [1, 16]}
        cfg_path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_map_config(cfg_path)
        assert loaded["folder_to_location"]["test"] == 5

    def test_missing_file_returns_default(self, tmp_path):
        loaded = load_map_config(tmp_path / "nonexistent.json")
        assert "berth_401" in loaded["folder_to_location"]
