"""Step 3 - Augment train and test splits of the Roboflow YOLO dataset.

Calls generate_synthetic_data.py with sensible defaults.
Run from the project root:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --copies 5
    python scripts/augment_dataset.py --dry-run
"""
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description="Augment train and test splits.")
parser.add_argument("--dataset", default="roboflow data")
parser.add_argument("--copies", type=int, default=3)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

cmd = [
    sys.executable, "generate_synthetic_data.py",
    "--dataset", args.dataset,
    "--splits", "train", "test",
    "--copies", str(args.copies),
]
if args.dry_run:
    cmd.append("--dry-run")

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
