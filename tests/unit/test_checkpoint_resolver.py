"""
Unit tests for src/dashboard/checkpoint_resolver.py.

Requirements covered: 4.1, 4.2, 4.3, 4.4
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dashboard.checkpoint_resolver import CheckpointResolver


@pytest.fixture
def mock_runs(tmp_path) -> Path:
    """Create a fake runs/train/ structure with two checkpoints."""
    # Older checkpoint
    old_dir = tmp_path / "runs" / "train" / "old_run" / "weights"
    old_dir.mkdir(parents=True)
    old_best = old_dir / "best.pt"
    old_best.write_bytes(b"old_model_data")
    # Set mtime to 1 hour ago
    old_mtime = time.time() - 3600
    os.utime(old_best, (old_mtime, old_mtime))

    # Newer checkpoint
    new_dir = tmp_path / "runs" / "train" / "new_run" / "weights"
    new_dir.mkdir(parents=True)
    new_best = new_dir / "best.pt"
    new_best.write_bytes(b"new_model_data")
    # mtime = now (default)

    return tmp_path


class TestConfigOverride:
    def test_config_path_exists_uses_it(self, tmp_path):
        ckpt = tmp_path / "my_model.pt"
        ckpt.write_bytes(b"data")

        resolver = CheckpointResolver(
            config_path=str(ckpt),
            discovery_pattern="nonexistent_pattern_*",
        )
        assert resolver.resolve() == ckpt
        assert resolver.source == "config"

    def test_config_path_missing_falls_through(self, mock_runs):
        pattern = str(mock_runs / "runs" / "train" / "*" / "weights" / "best.pt")
        resolver = CheckpointResolver(
            config_path="/nonexistent/path/model.pt",
            discovery_pattern=pattern,
        )
        assert resolver.resolve() is not None
        assert resolver.source == "auto-discovered"


class TestAutoDiscovery:
    def test_picks_most_recent_by_mtime(self, mock_runs):
        pattern = str(mock_runs / "runs" / "train" / "*" / "weights" / "best.pt")
        resolver = CheckpointResolver(
            config_path=None,
            discovery_pattern=pattern,
        )
        resolved = resolver.resolve()
        assert resolved is not None
        assert "new_run" in str(resolved)
        assert resolver.source == "auto-discovered"

    def test_single_checkpoint_found(self, tmp_path):
        d = tmp_path / "runs" / "train" / "only_run" / "weights"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")

        pattern = str(tmp_path / "runs" / "train" / "*" / "weights" / "best.pt")
        resolver = CheckpointResolver(config_path=None, discovery_pattern=pattern)
        assert resolver.resolve() is not None
        assert resolver.source == "auto-discovered"


class TestNothingFound:
    def test_no_config_no_discovery_returns_none(self, tmp_path):
        resolver = CheckpointResolver(
            config_path=None,
            discovery_pattern=str(tmp_path / "nothing_here_*"),
        )
        assert resolver.resolve() is None
        assert resolver.source == "none"

    def test_config_doesnt_exist_and_no_discovery(self, tmp_path):
        resolver = CheckpointResolver(
            config_path="/fake/path.pt",
            discovery_pattern=str(tmp_path / "nothing_here_*"),
        )
        assert resolver.resolve() is None
        assert resolver.source == "none"
