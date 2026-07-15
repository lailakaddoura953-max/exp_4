"""Quick inspection of roboflow label files to understand annotation format."""
import os

label_dir = r'roboflow data\train\labels'  # relative to project root
target_classes = {'2': 'Container-Open', '9': 'Human', '10': 'Human-NoSafety'}
found = {c: [] for c in target_classes}

for fname in os.listdir(label_dir):
    if not fname.endswith('.txt'):
        continue
    with open(os.path.join(label_dir, fname)) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cid = parts[0]
            if cid in target_classes and len(found[cid]) < 3:
                found[cid].append((fname, line.strip(), len(parts)))

for cid, items in found.items():
    for fname, line, num_vals in items:
        print(f'CLASS {cid} ({target_classes[cid]}) | FILE: {fname} | num_values: {num_vals}')
        print(f'  {line}')
    if not items:
        print(f'CLASS {cid} ({target_classes[cid]}): NOT FOUND in train labels')
