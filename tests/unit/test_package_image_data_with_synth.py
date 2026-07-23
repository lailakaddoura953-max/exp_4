"""
Unit tests for scripts/package_image_data_with_synth.py.

Builds a small mock image_data_with_synth/-shaped fixture (including an
auto_accepted/ bucket with a DIFFERENT depth than augmented_hazards/'s
<location>/<day|night>/ shape, and one image with no matching label file)
and verifies the packaging script discovers pairs correctly, splits them,
never touches image_data_with_synth/ itself, and both class-list modes
(full vs. reduced) produce correct output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import package_image_data_with_synth as pkg  # noqa: E402
from hazard_detection.rule_engine.class_taxonomy import (  # noqa: E402
    FULL_CLASS_NAMES,
    REDUCED_CLASS_SET,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG-ish placeholder bytes


def _write_label(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def synth_fixture(tmp_path) -> Path:
    """
    Build:
      augmented_hazards/berth_401/day/img_001.PNG + label (class 9, Human)
      augmented_hazards/berth_401/night/img_002.PNG + label (class 2, Container - Open)
      normal_operations/augmented_normal/berth_401/day/img_003.PNG + label (class 9, Human)
      normal_operations/auto_accepted/berth_401/img_004.PNG + label (class 4, Container - Reefer)
          -- NOTE: no day/night subfolder here, deliberately, to confirm
          the script doesn't assume a fixed depth.
      normal_operations/auto_accepted/berth_401/img_005.PNG  -- NO matching
          label file, to confirm it's skipped and counted, not silently
          dropped or crashing.
    """
    root = tmp_path / "image_data_with_synth"

    _write_image(root / "augmented_hazards" / "berth_401" / "day" / "img_001.PNG")
    _write_label(root / "augmented_hazards" / "berth_401" / "day" / "img_001.txt",
                 ["9 0.5 0.5 0.1 0.2"])

    _write_image(root / "augmented_hazards" / "berth_401" / "night" / "img_002.PNG")
    _write_label(root / "augmented_hazards" / "berth_401" / "night" / "img_002.txt",
                 ["2 0.3 0.4 0.2 0.15"])

    _write_image(root / "normal_operations" / "augmented_normal" / "berth_401" / "day" / "img_003.PNG")
    _write_label(root / "normal_operations" / "augmented_normal" / "berth_401" / "day" / "img_003.txt",
                 ["9 0.6 0.6 0.1 0.2"])

    _write_image(root / "normal_operations" / "auto_accepted" / "berth_401" / "img_004.PNG")
    _write_label(root / "normal_operations" / "auto_accepted" / "berth_401" / "img_004.txt",
                 ["4 0.5 0.5 0.3 0.3"])

    _write_image(root / "normal_operations" / "auto_accepted" / "berth_401" / "img_005.PNG")
    # deliberately no matching img_005.txt

    return root


class TestDiscoverPairs:
    def test_finds_all_valid_pairs_across_all_three_buckets(self, synth_fixture):
        pairs, skipped = pkg.discover_pairs(synth_fixture)
        assert len(pairs) == 4
        assert skipped == 1

    def test_handles_auto_accepted_different_depth(self, synth_fixture):
        pairs, _ = pkg.discover_pairs(synth_fixture)
        buckets_found = {bucket for _, _, bucket in pairs}
        assert "normal_operations/auto_accepted" in buckets_found

    def test_does_not_duplicate_on_case_insensitive_glob(self, synth_fixture):
        pairs, _ = pkg.discover_pairs(synth_fixture)
        image_paths = [str(p[0].resolve()) for p in pairs]
        assert len(image_paths) == len(set(image_paths))


class TestBuildSplitFullClasses:
    def test_produces_correct_split_sizes_and_data_yaml(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output"
        data_yaml_path, report = pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.25,
            test_fraction=0.0,
            seed=1,
            reduced_classes=False,
            dry_run=False,
        )

        assert data_yaml_path.exists()
        total = report["split_counts"]["train"] + report["split_counts"]["val"] + report["split_counts"]["test"]
        assert total == 4
        assert report["skipped_no_matching_label"] == 1

        yaml_text = data_yaml_path.read_text(encoding="utf-8")
        assert "nc: 17" in yaml_text
        assert "Container - Reefer" in yaml_text  # full taxonomy, not dropped

    def test_never_writes_into_source_directory(self, synth_fixture, tmp_path):
        before = sorted(str(p) for p in synth_fixture.rglob("*"))
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=tmp_path / "output",
            val_fraction=0.25,
            test_fraction=0.0,
            seed=1,
            reduced_classes=False,
            dry_run=False,
        )
        after = sorted(str(p) for p in synth_fixture.rglob("*"))
        assert before == after

    def test_dry_run_writes_nothing(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_dry"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.25,
            test_fraction=0.0,
            seed=1,
            reduced_classes=False,
            dry_run=True,
        )
        assert not output_dir.exists()

    def test_labels_copied_unchanged_full_mode(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=1,
            reduced_classes=False,
            dry_run=False,
        )
        all_label_lines = []
        for label_file in sorted((output_dir / "train" / "labels").glob("*.txt")):
            all_label_lines.extend(label_file.read_text(encoding="utf-8").strip().splitlines())
        # Class 4 (Container - Reefer) must still be present unmodified.
        assert any(line.startswith("4 ") for line in all_label_lines)


class TestBuildSplitReducedClasses:
    def test_dropped_class_label_becomes_empty_not_deleted(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_reduced"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=1,
            reduced_classes=True,
            dry_run=False,
        )
        # img_004's only label was class 4 (Container - Reefer, dropped).
        # Its image file must still exist; its label file must exist but
        # be empty (background image), not be missing.
        label_files = sorted((output_dir / "train" / "labels").glob("*.txt"))
        image_files = sorted((output_dir / "train" / "images").glob("*"))
        assert len(label_files) == len(image_files) == 4

        empty_labels = [f for f in label_files if f.read_text(encoding="utf-8").strip() == ""]
        assert len(empty_labels) == 1

    def test_kept_class_indices_remapped_correctly(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_reduced2"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=1,
            reduced_classes=True,
            dry_run=False,
        )
        # class 9 (Human) in full taxonomy -> index 5 in REDUCED_CLASS_SET
        expected_human_index = REDUCED_CLASS_SET.index("Human")
        # class 2 (Container - Open) -> its reduced index
        expected_open_index = REDUCED_CLASS_SET.index("Container - Open")

        all_lines = []
        for label_file in (output_dir / "train" / "labels").glob("*.txt"):
            all_lines.extend(label_file.read_text(encoding="utf-8").strip().splitlines())
        all_lines = [l for l in all_lines if l]

        class_ids_present = {int(line.split()[0]) for line in all_lines}
        assert expected_human_index in class_ids_present
        assert expected_open_index in class_ids_present
        # The dropped class's ORIGINAL index (4) must never appear, nor
        # any index >= 12 (Reduced_Class_Set size).
        assert all(0 <= cid < 12 for cid in class_ids_present)

    def test_data_yaml_has_twelve_classes(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_reduced3"
        data_yaml_path, _ = pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=1,
            reduced_classes=True,
            dry_run=False,
        )
        yaml_text = data_yaml_path.read_text(encoding="utf-8")
        assert "nc: 12" in yaml_text
        assert "Container - Reefer" not in yaml_text

    def test_coordinates_never_recalculated(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_reduced4"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.0,
            test_fraction=0.0,
            seed=1,
            reduced_classes=True,
            dry_run=False,
        )
        all_lines = []
        for label_file in (output_dir / "train" / "labels").glob("*.txt"):
            all_lines.extend(label_file.read_text(encoding="utf-8").strip().splitlines())
        all_lines = [l for l in all_lines if l]
        coordinate_sets = {tuple(line.split()[1:]) for line in all_lines}
        # Original coordinate tuples from the fixture, byte-for-byte.
        assert ("0.5", "0.5", "0.1", "0.2") in coordinate_sets  # img_001
        assert ("0.3", "0.4", "0.2", "0.15") in coordinate_sets  # img_002


class TestTrainTestSplit:
    def test_test_fraction_produces_test_split(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_test_split"
        _, report = pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=1,
            reduced_classes=False,
            dry_run=False,
        )
        assert report["split_counts"]["test"] >= 1
        assert (output_dir / "test" / "images").exists()

    def test_no_test_fraction_omits_test_dir(self, synth_fixture, tmp_path):
        output_dir = tmp_path / "output_no_test"
        pkg.build_split(
            synth_dir=synth_fixture,
            output_dir=output_dir,
            val_fraction=0.25,
            test_fraction=0.0,
            seed=1,
            reduced_classes=False,
            dry_run=False,
        )
        assert not (output_dir / "test").exists()
